"""
Usage:
    python3 -m homework.train_planner --model_name mlp_planner
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2025,
    num_workers: int = 2,
    device_str: str = None,
    **kwargs,
):
    print("Starting training...")

    # Device setup
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("CUDA not detected — using CPU.")

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Logging setup
    run_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = tb.SummaryWriter(run_dir)

    # Choose transform pipeline based on model type
    transform_pipeline = "default" if "vit" in model_name else "state_only"

    # Load data
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    # Load model
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # Loss function — masked smooth L1
    def masked_smooth_l1(preds, targets, mask):
        mask = mask.unsqueeze(-1).float()
        diff = torch.abs(preds - targets)
        loss = torch.where(diff < 1.0, 0.5 * diff ** 2, diff - 0.5)
        loss = loss * mask
        denom = mask.sum()
        return loss.sum() / denom if denom.item() > 0 else loss.sum() * 0.0

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        train_metric = PlannerMetric()
        train_losses = []

        # -------------------- Training --------------------
        for batch in train_loader:
            if "vit" in model_name:
                x = batch["image"].to(device)
            else:
                x = (batch["track_left"].to(device), batch["track_right"].to(device))

            y = batch["waypoints"].to(device)
            mask = batch["waypoints_mask"].to(device)

            preds = model(x) if "vit" in model_name else model(*x)
            loss = masked_smooth_l1(preds, y, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_metric.add(preds, y, mask)
            train_losses.append(loss.item())

            writer.add_scalar("train/step_loss", loss.item(), global_step)
            global_step += 1

        train_stats = train_metric.compute()
        avg_train_loss = float(np.mean(train_losses))

        # -------------------- Validation --------------------
        model.eval()
        val_metric = PlannerMetric()
        val_losses = []
        with torch.inference_mode():
            for batch in val_loader:
                if "vit" in model_name:
                    x = batch["image"].to(device)
                else:
                    x = (batch["track_left"].to(device), batch["track_right"].to(device))

                y = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)
                preds = model(x) if "vit" in model_name else model(*x)

                vloss = masked_smooth_l1(preds, y, mask)
                val_losses.append(vloss.item())
                val_metric.add(preds, y, mask)

        val_stats = val_metric.compute()
        avg_val_loss = float(np.mean(val_losses))

        # -------------------- Logging --------------------
        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
        writer.add_scalar("epoch/val_loss", avg_val_loss, epoch)
        writer.add_scalar("epoch/train_longitudinal", train_stats["longitudinal_error"], epoch)
        writer.add_scalar("epoch/train_lateral", train_stats["lateral_error"], epoch)
        writer.add_scalar("epoch/val_longitudinal", val_stats["longitudinal_error"], epoch)
        writer.add_scalar("epoch/val_lateral", val_stats["lateral_error"], epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(
                f"[{model_name}] Epoch {epoch + 1:02d}/{num_epoch:02d} "
                f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
                f"train_lat={train_stats['lateral_error']:.3f} val_lat={val_stats['lateral_error']:.3f} "
                f"train_long={train_stats['longitudinal_error']:.3f} val_long={val_stats['longitudinal_error']:.3f}"
            )

    # Save final model
    save_model(model)
    ckpt_path = run_dir / f"{model_name}.th"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Training complete. Model checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train planner model")
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device_str", type=str, default=None)
    args = parser.parse_args()

    train(**vars(args))
