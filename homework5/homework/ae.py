import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary tensor from (H, W, C) to (C, H, W) format.
    This allows us to switch from trnasformer-style channel-last to pytorch-style channel-first
    images. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    The opposite of hwc_to_chw. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    """
    Takes an image tensor of the shape (B, H, W, 3) and patchifies it into
    an embedding tensor of the shape (B, H//patch_size, W//patch_size, latent_dim).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, 3) an image tensor dtype=float normalized to -1 ... 1

        return: (B, H//patch_size, W//patch_size, latent_dim) a patchified embedding tensor
        """
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))


class UnpatchifyLinear(torch.nn.Module):
    """
    Takes an embedding tensor of the shape (B, w, h, latent_dim) and reconstructs
    an image tensor of the shape (B, w * patch_size, h * patch_size, 3).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, w, h, latent_dim) an embedding tensor

        return: (B, H * patch_size, W * patch_size, 3) a image tensor
        """
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck),
        where h = H // patch_size, w = W // patch_size and bottleneck is the size of the
        AutoEncoders bottleneck.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, h, w, bottleneck) into an image (B, H, W, 3),
        We will train the auto-encoder such that decode(encode(x)) ~= x.
        """


import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchAutoEncoder(nn.Module):
    """
    Patch-level AutoEncoder implementation.
    Splits an image into patches, encodes each patch into a latent vector,
    and reconstructs the full image from these encoded features.
    """

    class PatchEncoder(nn.Module):
        """
        Encoder that maps input patches to latent representations.
        """
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, bottleneck, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(bottleneck, bottleneck, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(bottleneck, latent_dim, kernel_size=3, stride=1, padding=1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

    class PatchDecoder(nn.Module):
        """
        Decoder that reconstructs image patches from latent features.
        """
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Conv2d(latent_dim, bottleneck, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(bottleneck, bottleneck, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(bottleneck, 3, kernel_size=3, stride=1, padding=1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(x)

    def __init__(self, patch_size: int = 25, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck)
        self.decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts an image into non-overlapping patches.
        Args:
            x: (B, 3, H, W)
        Returns:
            patches: (B * num_patches, 3, P, P)
        """
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, "Image dimensions must be divisible by patch size"
        h, w = H // P, W // P
        x = x.reshape(B, C, h, P, w, P)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, h, w, C, P, P)
        patches = x.reshape(B * h * w, C, P, P)
        return patches, (h, w)

    def unpatchify(self, patches: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Reconstructs the full image from patches.
        Args:
            patches: (B * h * w, 3, P, P)
        Returns:
            image: (B, 3, H, W)
        """
        B_hw = patches.shape[0]
        B = B_hw // (h * w)
        C, P = patches.shape[1], self.patch_size
        patches = patches.reshape(B, h, w, C, P, P)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        image = patches.reshape(B, C, h * P, w * P)
        return image

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass: reconstructs the input image.
        Returns:
            recon: reconstructed image
            aux: dictionary with auxiliary losses (optional)
        """
        patches, (h, w) = self.patchify(x)
        z = self.encoder(patches)
        recon_patches = self.decoder(z)
        recon = self.unpatchify(recon_patches, h, w)

        # Optional auxiliary visualization metric
        mse_loss = F.mse_loss(recon, x)
        return recon, {"reconstruction_loss": mse_loss}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        patches, _ = self.patchify(x)
        return self.encoder(patches)

    def decode(self, z: torch.Tensor, h: int, w: int) -> torch.Tensor:
        recon_patches = self.decoder(z)
        return self.unpatchify(recon_patches, h, w)

