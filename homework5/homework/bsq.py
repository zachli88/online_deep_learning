import abc

import torch

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    # Written with the help of ChatGPT
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self.codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim

        self.proj_down = torch.nn.Linear(embedding_dim, codebook_bits)
        self.proj_up = torch.nn.Linear(codebook_bits, embedding_dim)

    # Written with the help of ChatGPT
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        z = self.proj_down(x)  # (..., codebook_bits)

        # L2 normalization along last dim
        norm = z.norm(dim=-1, keepdim=True) + 1e-8
        z = z / norm

        # Differentiable binarization to -1 / 1
        z_bin = diff_sign(z)
        return z_bin

    # Written with the help of ChatGPT
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        return self.proj_up(x)

    # Written with the help of ChatGPT
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # Written with the help of ChatGPT
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    # Written with the help of ChatGPT
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    # Written with the help of ChatGPT
    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    # Written with the help of ChatGPT
    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    # Written with the help of ChatGPT
    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim, bottleneck=latent_dim)
        self.codebook_bits = codebook_bits

        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=self.bottleneck)

    # Written with the help of ChatGPT
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        z = super().encode(x)
        tokens = self.bsq.encode_index(z)  
        return tokens

    # Written with the help of ChatGPT
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bsq.decode_index(x)
        img = super().decode(z)
        return img

    # Written with the help of ChatGPT
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = super().encode(x)
        z_q = self.bsq(z)
        return z_q

    # Written with the help of ChatGPT
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return super().decode(x)

    # Written with the help of ChatGPT
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        z = super().encode(x)

        z_code = self.bsq.encode(z)
        z_q = self.bsq.decode(z_code)

        recon = super().decode(z_q)

        with torch.no_grad():
            tokens = self.bsq._code_to_index(z_code)
            cnt = torch.bincount(
                tokens.flatten(),
                minlength=2 ** self.codebook_bits
            ).float()

            extra = {
                "cb0": (cnt == 0).float().mean(),
                "cb2": (cnt <= 2).float().mean(),
            }

        return recon, {k: v.detach() for k, v in extra.items()}
