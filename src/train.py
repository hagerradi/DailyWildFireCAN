"""Builds every training component from a :class:`~src.config.Config` and runs the training loop."""
from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from src.builder import build_dataloaders, build_model, resolve_device
from src.config import Config
from src.losses import build_loss
from src.trainer import Trainer
from src.utils import seed_everything


def build_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, steps_per_epoch: int):
    """Per-step schedule: one epoch of linear warm-up (10% -> 100% of the LR), then cosine decay."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch if num_epochs > 1 else max(1, total_steps // 10)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def train(config: Config) -> tuple[Trainer, dict[str, DataLoader | None]]:
    """Train ``config.model.architecture`` on the train split, selecting on the validation loss.

    Uses AdamW, the warm-up + cosine schedule of :func:`build_scheduler` and the
    focal + Dice loss of :func:`src.losses.build_loss`. Checkpoints are written to
    ``config.training.checkpoint_dir``.

    Returns:
        The trainer and the dataloaders of every split (keyed by split name).
    """
    seed_everything(config.seed)
    device = resolve_device(config.training.device)

    loaders = build_dataloaders(config)
    if loaders["train"] is None or loaders["val"] is None:
        raise FileNotFoundError("Training needs non-empty 'train' and 'val' splits; check the samples folder.")

    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )
    scheduler = build_scheduler(optimizer, config.training.num_epochs, len(loaders["train"]))

    logger = None
    if config.comet.enabled:
        from src.logger import CometLogger

        cc = config.comet
        logger = CometLogger(
            project_name=cc.project_name,
            workspace=cc.workspace,
            experiment_name=f"{cc.experiment_name}-{int(time.time())}",
            experiment_tags=cc.experiment_tags or None,
        )

    trainer = Trainer(model, optimizer, scheduler, build_loss(), config.training, logger)
    print(f"Architecture: {config.model.architecture} | device: {trainer.device} | "
          f"train batches: {len(loaders['train'])} | val batches: {len(loaders['val'])}")
    trainer.fit(loaders["train"], loaders["val"])
    return trainer, loaders


def test(trainer: Trainer, loaders: dict[str, DataLoader | None], checkpoint_name: str = "best_checkpoint.pt") -> dict:
    """Reload the best checkpoint and evaluate it on every non-empty test split.

    Returns:
        ``{split: metrics}`` (metrics include ``loss``).
    """
    from src.data import SPLIT_LABELS, TEST_SPLITS

    checkpoint = trainer.checkpoint_dir / checkpoint_name
    if not checkpoint.exists():
        print(f"No {checkpoint} found; skipping the test evaluation.")
        return {}
    epoch = trainer.load_checkpoint(checkpoint)
    print(f"\nTesting {checkpoint} (epoch {epoch})")

    results = {}
    for split in TEST_SPLITS:
        if loaders.get(split) is None:
            continue
        loss, metrics = trainer.evaluate(loaders[split])
        results[split] = {"loss": loss, **metrics}
        print(f"[{SPLIT_LABELS[split]}] loss={loss:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if trainer.logger is not None:
            trainer.logger.log_metrics({f"{split}_{k}": v for k, v in results[split].items()})
    return results
