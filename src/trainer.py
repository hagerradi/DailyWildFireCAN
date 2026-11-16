"""Training loop with validation, checkpointing and optional Comet logging."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.builder import forward_batch, resolve_device
from src.config import TrainingConfig
from src.metrics import BinaryMetrics

if TYPE_CHECKING:
    from src.logger import CometLogger


class Trainer:
    """Trains a model, keeps the checkpoint with the lowest validation loss and resumes after preemption.

    Two checkpoints are written to ``config.checkpoint_dir`` after every epoch:
    ``latest_checkpoint.pt`` (used to resume automatically if it exists when the
    trainer is created) and ``best_checkpoint.pt`` (lowest validation loss). Both
    contain ``model_state_dict``, ``optimizer_state_dict``, ``scheduler_state_dict``,
    ``epoch`` and ``best_val_loss``.

    The scheduler is stepped after every batch.

    Args:
        model: Model returning ``(B, 1, H, W)`` logits.
        optimizer: Optimizer.
        scheduler: Learning-rate scheduler.
        loss_fn: Loss taking ``(logits, labels)``.
        config: Training configuration.
        logger: Optional Comet logger.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_fn: nn.Module,
        config: TrainingConfig,
        logger: CometLogger | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.logger = logger
        self.metrics = BinaryMetrics(self.device)

        if logger is not None:
            logger.log_params({
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "device": str(self.device),
            })

        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        latest = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest.exists():
            self.start_epoch = self.load_checkpoint(latest) + 1
            print(f"Resuming from {latest} at epoch {self.start_epoch} (best val loss {self.best_val_loss:.4f})")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train for ``config.num_epochs`` epochs, validating after each one."""
        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_metrics = self.evaluate(val_loader, prefix="Val_")
            print(f"Epoch [{epoch}/{self.config.num_epochs}] train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_auc_pr={val_metrics['Val_AUC_PR']:.4f}")

            if self.logger is not None:
                self.logger.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss,
                     "learning_rate": self.optimizer.param_groups[0]["lr"], **val_metrics},
                    step=epoch, epoch=epoch,
                )

            if val_loss < self.best_val_loss:
                print(f"Validation loss improved {self.best_val_loss:.4f} -> {val_loss:.4f}; saving best checkpoint.")
                self.best_val_loss = val_loss
                self.save_checkpoint(self.checkpoint_dir / "best_checkpoint.pt", epoch)
            self.save_checkpoint(self.checkpoint_dir / "latest_checkpoint.pt", epoch)

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one training epoch and return the mean batch loss."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar, start=1):
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            loss = self.loss_fn(forward_batch(self.model, batch, self.device), labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix(loss=f"{total_loss / batch_idx:.4f}")
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "") -> tuple[float, dict[str, float]]:
        """Return the mean loss and the metrics of :class:`src.metrics.BinaryMetrics` on ``loader``."""
        self.model.eval()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Evaluating {prefix.rstrip('_')}".strip(), leave=False):
            labels = batch["label"].to(self.device)
            logits = forward_batch(self.model, batch, self.device)
            total_loss += self.loss_fn(logits, labels).item()
            self.metrics.update(logits, labels)
        return total_loss / len(loader), self.metrics.compute(prefix=prefix)

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "best_val_loss": self.best_val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Restore model, optimizer and scheduler states; returns the checkpoint epoch."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return checkpoint["epoch"]
