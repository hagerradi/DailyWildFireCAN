"""Pixel-level binary segmentation metrics reported in the paper.

All metrics are accumulated over every pixel of a split (micro-averaged):
IoU, F1, precision and recall at a 0.5 probability threshold, average precision
(AP) and the area under the precision-recall curve (AUC-PR, the main metric),
both computed on 200 thresholds.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import auc
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
)

THRESHOLD = 0.5
NUM_PR_THRESHOLDS = 200
METRIC_NAMES = ("AUC_PR", "Average_Precision", "IoU", "F1", "Precision", "Recall")


class BinaryMetrics:
    """Accumulates binary metrics over batches of logits.

    Example::

        metrics = BinaryMetrics(device)
        for batch in loader:
            metrics.update(model_logits, labels)
        results = metrics.compute()   # {"AUC_PR": ..., "IoU": ..., ...}; also resets the state
    """

    def __init__(self, device: torch.device | str = "cpu", threshold: float = THRESHOLD) -> None:
        self.threshold = threshold
        self._metrics = {
            "IoU": BinaryJaccardIndex(threshold=threshold),
            "F1": BinaryF1Score(threshold=threshold),
            "Precision": BinaryPrecision(threshold=threshold),
            "Recall": BinaryRecall(threshold=threshold),
            "Average_Precision": BinaryAveragePrecision(thresholds=NUM_PR_THRESHOLDS),
            "PR_curve": BinaryPrecisionRecallCurve(thresholds=NUM_PR_THRESHOLDS),
        }
        for metric in self._metrics.values():
            metric.to(device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Update with ``(B, 1, H, W)`` logits and ``(B, H, W)`` binary labels."""
        probs = torch.sigmoid(logits).reshape(labels.shape)
        labels = labels.long()
        for metric in self._metrics.values():
            metric.update(probs, labels)

    def compute(self, prefix: str = "") -> dict[str, float]:
        """Return the metrics (keys prefixed with ``prefix``) and reset the state."""
        results = {name: self._metrics[name].compute().item()
                   for name in ("IoU", "F1", "Precision", "Recall", "Average_Precision")}
        precision, recall, _ = self._metrics["PR_curve"].compute()
        # 0/0 precision at the highest thresholds is set to 1 to anchor the curve.
        precision = np.nan_to_num(precision.cpu().numpy(), nan=1.0)
        results["AUC_PR"] = float(auc(recall.cpu().numpy(), precision))
        self.reset()
        return {f"{prefix}{name}": results[name] for name in METRIC_NAMES}

    def reset(self) -> None:
        for metric in self._metrics.values():
            metric.reset()
