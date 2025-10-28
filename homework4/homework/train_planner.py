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
    batch_size: int = 64,
    seed: int = 2025,
    **kwargs,
):
    print("Time to train")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size)
    val_data = load_data("drive_data/val", shuffle=False, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        metric = PlannerMetric()
        train_losses = []

        for batch in train_data:
            batch = {k: v.to(device) for k, v in batch.items()}

            preds = model(**batch)
            loss = criterion(preds, batch["waypoints"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric.add(preds, batch["waypoints"], batch["waypoints_mask"])
            train_losses.append(loss.item())
            global_step += 1

        train_stats = metric.compute()

        model.eval()
        val_metric = PlannerMetric()
        val_losses = []
        with torch.inference_mode():
            for batch in val_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = model(**batch)
                loss = criterion(preds, batch["waypoints"])
                val_metric.add(preds, batch["waypoints"], batch["waypoints_mask"])
                val_losses.append(loss.item())

        val_stats = val_metric.compute()

        mean_train_loss = np.mean(train_losses)
        mean_val_loss = np.mean(val_losses)

        logger.add_scalar("train/loss", mean_train_loss, global_step)
        logger.add_scalar("val/loss", mean_val_loss, global_step)
        logger.add_scalar("train/lateral_error", train_stats["lateral_error"], global_step)
        logger.add_scalar("train/longitudinal_error", train_stats["longitudinal_error"], global_step)
        logger.add_scalar("val/lateral_error", val_stats["lateral_error"], global_step)
        logger.add_scalar("val/longitudinal_error", val_stats["longitudinal_error"], global_step)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d}/{num_epoch:03d} "
                f"train_loss={mean_train_loss:.4f} val_loss={mean_val_loss:.4f} "
                f"train_lat={train_stats['lateral_error']:.3f} val_lat={val_stats['lateral_error']:.3f} "
                f"train_long={train_stats['longitudinal_error']:.3f} val_long={val_stats['longitudinal_error']:.3f}"
            )

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train planner model")

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2025)

    args = parser.parse_args()
    train(**vars(args))
