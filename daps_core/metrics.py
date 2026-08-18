from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum(dtype=np.float64)
    union = np.logical_or(pred_b, gt_b).sum(dtype=np.float64)
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float((inter + eps) / (union + eps))


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum(dtype=np.float64)
    denom = pred_b.sum(dtype=np.float64) + gt_b.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def multiclass_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    include_background: bool = True,
    eps: float = 1e-7,
) -> Dict[str, object]:
    """Compute mean IoU and per-class IoU for semantic segmentation.

    Pixels with gt == ignore_index are ignored. Classes absent in both pred and gt are
    excluded from the mean, following the usual VOC-style practical evaluation for one image.
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    valid = gt != ignore_index
    pred_v = pred[valid]
    gt_v = gt[valid]

    class_range = range(num_classes) if include_background else range(1, num_classes)
    per_class: List[Optional[float]] = []
    for c in class_range:
        p = pred_v == c
        g = gt_v == c
        union = np.logical_or(p, g).sum(dtype=np.float64)
        if union == 0:
            per_class.append(None)
            continue
        inter = np.logical_and(p, g).sum(dtype=np.float64)
        per_class.append(float((inter + eps) / (union + eps)))

    present = [x for x in per_class if x is not None]
    miou = float(np.mean(present)) if present else 0.0
    pixel_acc = float(np.mean(pred_v == gt_v)) if pred_v.size else 0.0
    return {"miou": miou, "pixel_acc": pixel_acc, "per_class": per_class}
