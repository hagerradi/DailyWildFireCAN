"""
Evaluator: encapsulates the final evaluation logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from torchmetrics.segmentation import MeanIoU, DiceScore
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryF1Score,
    BinaryPrecision, 
    BinaryRecall, 
    BinaryAveragePrecision,
    BinaryPrecisionRecallCurve
)
from sklearn.metrics import auc
from tqdm import tqdm
import numpy as np

from src.config import TrainingConfig

if TYPE_CHECKING:
    from src.logger import CometLogger


class Evaluator:
    """Manages the final evaluation using checkpoints.

    Args:
        model: PyTorch model to evaluate.
        loss_fn: Loss function.
        config: Training-specific configuration.
        logger: Optional :class:`~src.logger.CometLogger` instance.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
        logger: CometLogger | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.logger = logger
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)

        self.binary_classification_threshold = 0.5

        # Initialize metrics
        self.test_metrics = self._init_metrics()

        if self.logger is not None:
            self.logger.log_params(
                {
                    "batch_size": config.batch_size,
                    "num_epochs": config.num_epochs,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "device": str(self.device),
                    "notes":"Final Evaluation",
                }
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, loader: DataLoader) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(loader, total=len(loader), desc="Evaluating on test data", leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                
                images, labels = batch[0], batch[1]
                
                images: Tensor = images.to(self.device)
                labels: Tensor = labels.long().to(self.device)

                if len(batch) == 3:
                    delta_t = batch[2]
                    delta_t = delta_t.to(self.device)
                    predictions = self.model(images, delta_t)
                else:
                    predictions = self.model(images)

                loss: Tensor = self.loss_fn(predictions, labels)
                total_loss += loss.item()

                # Update test metrics
                self._update_metrics(self.test_metrics, predictions, labels)
                
                pbar.set_postfix({"loss": f"{(total_loss / (batch_idx + 1)):.4f}"})

        # Compute final metrics
        test_metric_results = self._compute_and_reset_metrics(self.test_metrics, prefix="Test_")
        
        return total_loss / len(loader), test_metric_results

    def load_checkpoint(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
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