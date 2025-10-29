from pathlib import Path
import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

import torch
import torch.nn as nn


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        """
        Args:
            n_track (int): number of points on each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        in_dim = n_track * 4           # (x, y) for left/right track boundaries
        out_dim = n_waypoints * 2      # (x, y) per predicted waypoint

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.LayerNorm(dims[i + 1]))
        self.net = nn.Sequential(*layers)

        # Xavier initialization for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left (torch.Tensor): (B, n_track, 2)
            track_right (torch.Tensor): (B, n_track, 2)

        Returns:
            torch.Tensor: predicted waypoints (B, n_waypoints, 2)
        """
        B, n_track, _ = track_left.shape
        assert n_track == self.n_track, f"expected n_track={self.n_track}, got {n_track}"

        fused = torch.cat([track_left, track_right], dim=-1).reshape(B, -1)
        out = self.net(fused)
        return out.reshape(B, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        self.track_embed = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_track * 2, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_head = nn.Sequential(
            nn.Linear(d_model * n_track * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        B = track_left.shape[0]
        x = torch.cat([track_left, track_right], dim=1)
        x = self.track_embed(x) + self.pos_embed
        x = self.encoder(x)
        x = x.reshape(B, -1)
        out = self.mlp_head(x)
        return out.view(B, self.n_waypoints, 2)

class PatchEmbedding(nn.Module):
    def __init__(self, h: int = 96, w: int = 128, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.h = h
        self.w = w
        self.patch_size = patch_size
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        return self.projection(x)


# -------------------------
# Transformer Block
# -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.drop1(attn_out)
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x

class ViTPlanner(nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        h: int = 96,
        w: int = 128,
        patch_size: int = 8,
        embed_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.patch_embed = PatchEmbedding(h, w, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)
        out = self.head(x)
        return out.view(x.size(0), self.n_waypoints, 2)

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "vit_planner": ViTPlanner,
}


def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> torch.nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)
    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(f"Failed to load {model_path.name}") from e
    if calculate_model_size_mb(m) > 20:
        raise AssertionError(f"{model_name} is too large")
    return m


def save_model(model: torch.nn.Module) -> str:
    model_name = next((n for n, m in MODEL_FACTORY.items() if type(model) is m), None)
    if model_name is None:
        raise ValueError(f"Model type {type(model)} not supported")
    out_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), out_path)
    return out_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
