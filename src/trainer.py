"""
Trainer: encapsulates the training and validation loops.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm
import matplotlib.pyplot as plt
import uuid
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchmetrics.segmentation import MeanIoU, DiceScore
from torchmetrics.classification import (
    BinaryJaccardIndex,    # Binary IoU
    BinaryF1Score,         # Binary Dice
    BinaryPrecision, 
    BinaryRecall, 
    BinaryAveragePrecision,
    BinaryPrecisionRecallCurve
)
from sklearn.metrics import auc

from src.config import TrainingConfig

if TYPE_CHECKING:
    from src.logger import CometLogger


class Trainer:
    """Manages the training loop, validation, and checkpointing.

    Args:
        model: PyTorch model to train.
        optimizer: Optimizer instance.
        scheduler: Scheduler instance.
        loss_fn: Loss function (criterion).
        config: Training-specific configuration.
        logger: Optional :class:`~src.logger.CometLogger` instance.  When
            provided, hyper-parameters are logged at construction time and
            train/val losses are logged after every epoch.
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
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.logger = logger
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)

        self.binary_classification_threshold = 0.5

        # Initialize separate metrics for train and validation phases
        self.val_metrics = self._init_metrics()

        if self.logger is not None:
            self.logger.log_params(
                {
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "device": str(self.device),
                    "notes":"Training, validation, and testing",
                }
            )

        # --- Preemption Recovery Logic ---
        self.start_epoch = 1
        self.best_val_loss = float('inf')
        
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        latest_ckpt_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_ckpt_path.exists():
            print(f"\n[*] Preemption detected! Resuming from {latest_ckpt_path}")
            last_epoch = self.load_checkpoint(latest_ckpt_path)
            self.start_epoch = last_epoch + 1
            print(f"[*] Resuming at Epoch {self.start_epoch} (Best Val Loss: {self.best_val_loss:.4f})")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _forward_pass(self, batch_data: dict) -> Tensor:
        """Automatically routes data based on the dataset structure."""
        kwargs = {}
        if "loc_emb" in batch_data: kwargs["loc_emb"] = batch_data["loc_emb"].to(self.device)
        if "satellite_age" in batch_data: kwargs["delta_t"] = batch_data["satellite_age"].to(self.device) 
        if "olmo_emb" in batch_data: kwargs["olmo_emb"] = batch_data["olmo_emb"].to(self.device)
        if "alpha_emb" in batch_data: kwargs["alpha_emb"] = batch_data["alpha_emb"].to(self.device)

        if "input_grids" in batch_data:
            return self.model(batch_data["input_grids"].to(self.device), **kwargs)
            
        elif "input_rgb" in batch_data and "input_env" in batch_data:
            return self.model(batch_data["input_rgb"].to(self.device), 
                              batch_data["input_env"].to(self.device), **kwargs)
            
        elif "state" in batch_data and "dynamic" in batch_data and "constant" in batch_data:
            return self.model(batch_data["state"].to(self.device), 
                              batch_data["dynamic"].to(self.device), 
                              batch_data["constant"].to(self.device))
        else:
            raise ValueError("Unrecognized batch structure from DataLoader.")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for the training split.
            val_loader: DataLoader for the validation split.
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)


        # for epoch in range(1, self.config.num_epochs + 1):
        for epoch in range(self.start_epoch, self.config.num_epochs + 1):

            train_loss, train_metrics_results = self.train_epoch(train_loader, epoch)
            val_loss, val_metrics_results = self.validate(val_loader, "Val", epoch)

            # Target evaluation strictly for console logging feedback
            target_metric_key = "Val_IoU_2_New_Fire" if self.config.use_cumuarea else "Val_IoU"
            current_val_iou = val_metrics_results.get(target_metric_key, 0.0)

            print(f"Epoch [{epoch}/{self.config.num_epochs}]  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_iou={current_val_iou:.4f}")

            if self.logger is not None:
                self.logger.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss},
                    step=epoch,
                    epoch=epoch,
                )
                self.logger.log_metrics(train_metrics_results, step=epoch, epoch=epoch)
                self.logger.log_metrics(val_metrics_results, step=epoch, epoch=epoch)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_metrics({"learning_rate": current_lr}, step=epoch, epoch=epoch)

            # Check if this is the best model so far based on the loss
            if val_loss < self.best_val_loss:
                
                print(f"*** Validation Loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving best model! ***")
                self.best_val_loss = val_loss
                
                best_ckpt_path = checkpoint_dir / "best_checkpoint.pt"
                self.save_checkpoint(best_ckpt_path, epoch)

            # Latest checkpoint for preemption
            latest_ckpt_path = self.checkpoint_dir / "latest_checkpoint.pt"
            self.save_checkpoint(latest_ckpt_path, epoch)

    def train_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, dict[str, float]]:
        """Run one training epoch.

        Args:
            loader: DataLoader for the training split.
            epoch: Current epoch number (used for logging).

        Returns:
            Average training loss over all batches.
        """

        self.model.train()
        total_loss = 0.0

        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train Epoch {epoch}")

        for batch_idx, batch_data in pbar:

            labels = batch_data["label"].long().to(self.device)

            self.optimizer.zero_grad()

            # Pass the images
            predictions = self._forward_pass(batch_data)

            loss: Tensor = self.loss_fn(predictions, labels)
                
            loss.backward()
            
            self.optimizer.step()

            total_loss += loss.item()

            self.scheduler.step()

            if (batch_idx + 1) % self.config.log_interval == 0:
                avg = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch} | batch {batch_idx + 1}/{len(loader)} | loss={avg:.4f}")

        train_metric_results = {}
        
        return total_loss / len(loader), train_metric_results

    def validate(self, loader: DataLoader, prefix: str = "Val", epoch: int | None = None) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(loader, total=len(loader), desc="Validating", leave=False)
        logged_image_this_epoch = False

        with torch.no_grad():

            for batch_idx, batch in enumerate(pbar):

                # Grab the label
                labels = batch["label"].long().to(self.device)

                predictions = self._forward_pass(batch)

                loss: Tensor = self.loss_fn(predictions, labels)
                total_loss += loss.item()

                # Update validation metrics
                self._update_metrics(self.val_metrics, predictions, labels)

                # IMAGE LOGGING
                if not logged_image_this_epoch and self.logger is not None and epoch is not None:
                    min_pixels = 30
                    selected_idx = -1
                    is_3_class = self.config.use_cumuarea
                    vmax_val = 2 if is_3_class else 1
                    
                    for i in range(labels.size(0)):
                        current_label = labels[i]
                        
                        if is_3_class:
                            c1_count = (current_label== 1).sum().item()
                            c2_count = (current_label == 2).sum().item()
                            is_interesting = (c1_count >= min_pixels) and (c2_count >= min_pixels)
                        else:
                            c1_count = (current_label == 1).sum().item()
                            is_interesting = (c1_count >= min_pixels)
                        
                        if is_interesting:
                            selected_idx = i
                            break
                    
                    if selected_idx == -1 and batch_idx == len(loader) - 1:
                        selected_idx = 0

                    if selected_idx != -1:
                        # Grab predictions for the selected image to plot
                        if predictions.shape[1] == 1:
                            pred_probs = torch.sigmoid(predictions[selected_idx]).squeeze(0)
                            pred_label = (pred_probs > self.binary_classification_threshold).long().cpu().numpy()
                        else:
                            pred_label = torch.argmax(predictions[selected_idx], dim=0).cpu().numpy()

                        true_label = labels[selected_idx].cpu().numpy()
                        
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        axes[0].imshow(true_label, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[0].set_title("Ground Truth")
                        axes[0].axis('off')
                        
                        axes[1].imshow(pred_label, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[1].set_title(f"Prediction (Epoch {epoch})")
                        axes[1].axis('off')
                        
                        random_id = uuid.uuid4().hex[:6]
                        temp_img_path = f"temp_val_epoch_{epoch}_{random_id}.png"
                        plt.savefig(temp_img_path, bbox_inches='tight')
                        plt.close(fig)
                        
                        self.logger.log_image(temp_img_path, name=f"Validation/Epoch_{epoch}", epoch=epoch)
                        os.remove(temp_img_path)
                        
                        logged_image_this_epoch = True
                
                pbar.set_postfix({"loss": f"{(total_loss / (batch_idx + 1)):.4f}"})

        # Compute final val metrics and reset states for the next epoch
        val_raw_results = self._compute_and_reset_metrics(self.val_metrics, prefix=f"{prefix}_")

        val_metric_results = {**val_raw_results}
        
        return total_loss / len(loader), val_metric_results

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "best_val_loss": getattr(self, "best_val_loss", float('inf')),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]
            
        return checkpoint["epoch"]

    # ------------------------------------------------------------------
    # Metric Helpers
    # ------------------------------------------------------------------

    def _init_metrics(self) -> dict[str, nn.Module]:
        """Initialize and return a dictionary of required metric objects."""
        metrics = {}
        if self.config.use_cumuarea:
            metrics['iou'] = MeanIoU(num_classes=3, per_class=True, input_format='index', include_background=True).to(self.device)
            metrics['dice'] = DiceScore(num_classes=3, average="none", input_format="index", include_background=True).to(self.device)
        else:
            metrics['iou'] = BinaryJaccardIndex(threshold=self.binary_classification_threshold).to(self.device)
            metrics['dice'] = BinaryF1Score(threshold=self.binary_classification_threshold).to(self.device)
            metrics['precision'] = BinaryPrecision().to(self.device)
            metrics['recall'] = BinaryRecall().to(self.device)
            metrics['ap'] = BinaryAveragePrecision(thresholds=200).to(self.device)
            metrics['aucpr'] = BinaryPrecisionRecallCurve(thresholds=200).to(self.device)
        return metrics

    def _update_metrics(self, metrics_dict: dict[str, nn.Module], predictions: Tensor, labels: Tensor) -> None:
        """Process model predictions and update the provided metrics dictionary."""
        pred_probs = None
        
        if predictions.shape[1] == 1:
            pred_probs = torch.sigmoid(predictions).squeeze(1)
            pred_classes = (pred_probs > self.binary_classification_threshold).long()
        else:
            pred_classes = torch.argmax(predictions, dim=1)

        metrics_dict['iou'].update(pred_classes, labels)
        metrics_dict['dice'].update(pred_classes, labels)

        if not self.config.use_cumuarea and pred_probs is not None:
            metrics_dict['precision'].update(pred_classes, labels)
            metrics_dict['recall'].update(pred_classes, labels)
            if 'ap' in metrics_dict:
                metrics_dict['ap'].update(pred_probs, labels)
            if 'aucpr' in metrics_dict:
                metrics_dict['aucpr'].update(pred_probs, labels)

    def _compute_and_reset_metrics(self, metrics_dict: dict[str, nn.Module], prefix: str = "") -> dict[str, float]:
        """Compute the final scores from the tracker, map them to a dict, and reset the states."""
        class_ious = metrics_dict['iou'].compute() 
        class_dices = metrics_dict['dice'].compute()

        final_metrics = {}

        if self.config.use_cumuarea:
            valid_iou_label = class_ious != -1
            macro_iou = class_ious[valid_iou_label].mean().item() if valid_iou_label.any() else 0.0

            valid_dice_label = class_dices != -1
            macro_dice = class_dices[valid_dice_label].mean().item() if valid_dice_label.any() else 0.0
            
            final_metrics.update({
                f"{prefix}Macro_IoU": macro_iou,
                f"{prefix}Macro_Dice": macro_dice,
                f"{prefix}IoU_0_Background": class_ious[0].item(),
                f"{prefix}IoU_1_Old_Fire": class_ious[1].item(),
                f"{prefix}IoU_2_New_Fire": class_ious[2].item(),
                f"{prefix}Dice_0_Background": class_dices[0].item(),
                f"{prefix}Dice_1_Old_Fire": class_dices[1].item(),
                f"{prefix}Dice_2_New_Fire": class_dices[2].item(),
            })
        else:
            final_metrics.update({
                f"{prefix}IoU": class_ious.item(),
                f"{prefix}F1": class_dices.item(),
                f"{prefix}Precision": metrics_dict['precision'].compute().item(),
                f"{prefix}Recall": metrics_dict['recall'].compute().item(),
            })

            if 'ap' in metrics_dict:
                final_metrics[f"{prefix}Average_Precision"] = metrics_dict['ap'].compute().item()
            
            if 'aucpr' in metrics_dict:
                # Get the curve arrays
                precision, recall, _ = metrics_dict['aucpr'].compute()
                # Move to CPU and compute AUC PR using scikit-learn
                precision_np = precision.cpu().numpy()
                recall_np = recall.cpu().numpy()
                # Replace any NaNs (from 0/0 division) with 1.0 to anchor the Y-axis
                clean_precision = np.nan_to_num(precision_np, nan=1.0)
                # Calculate the true trapezoidal area using scikit-learn
                final_metrics[f"{prefix}AUC_PR"] = auc(recall_np, clean_precision)

        # Reset all metrics for the next pass
        for metric in metrics_dict.values():
            metric.reset()

        return final_metrics

    # ------------------------------------------------------------------
    # Environment Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)