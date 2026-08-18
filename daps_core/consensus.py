from __future__ import annotations

from typing import Tuple

import numpy as np


def binary_entropy(prob: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Normalized binary Shannon entropy in [0, 1]."""
    p = np.clip(prob.astype(np.float32), eps, 1.0 - eps)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return (h / np.log(2.0)).astype(np.float32)


def multiclass_entropy(prob: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Normalized multiclass entropy in [0, 1] for prob [C,H,W]."""
    p = np.clip(prob.astype(np.float32), eps, 1.0)
    p = p / np.maximum(p.sum(axis=0, keepdims=True), eps)
    h = -np.sum(p * np.log(p), axis=0)
    max_h = np.log(max(2, p.shape[0]))
    return (h / max_h).astype(np.float32)


def daps_v6_average(
    prob1: np.ndarray, prob2: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Output-level analogue of DAPS v6: simple average without epistemic gate."""
    prob = 0.5 * prob1.astype(np.float32) + 0.5 * prob2.astype(np.float32)
    if prob.ndim == 2:
        mask = prob >= threshold
    elif prob.ndim == 3:
        mask = np.argmax(prob, axis=0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported probability shape {prob.shape}")
    return prob.astype(np.float32), mask


def daps_v7_entropy_weighted(
    prob1: np.ndarray, prob2: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Output-level analogue of DAPS v7: confidence/entropy weighted consensus.

    Binary case: prob is [H,W]. Multiclass case: prob is [C,H,W].
    High entropy means low confidence; the more confident model receives more weight
    voxel/pixel-wise. This is the MVP version of the entropy-gated dialectical veto.
    """
    prob1 = prob1.astype(np.float32)
    prob2 = prob2.astype(np.float32)

    if prob1.ndim == 2:
        h1 = binary_entropy(prob1)
        h2 = binary_entropy(prob2)
        c1 = 1.0 - h1
        c2 = 1.0 - h2
        denom = c1 + c2 + 1e-6
        w1 = c1 / denom
        w2 = c2 / denom
        prob = w1 * prob1 + w2 * prob2
        mask = prob >= threshold
        return prob.astype(np.float32), mask

    if prob1.ndim == 3:
        h1 = multiclass_entropy(prob1)
        h2 = multiclass_entropy(prob2)
        c1 = 1.0 - h1
        c2 = 1.0 - h2
        denom = c1 + c2 + 1e-6
        w1 = c1 / denom
        w2 = c2 / denom
        prob = w1[None, :, :] * prob1 + w2[None, :, :] * prob2
        prob = prob / np.maximum(prob.sum(axis=0, keepdims=True), 1e-6)
        mask = np.argmax(prob, axis=0).astype(np.uint8)
        return prob.astype(np.float32), mask

    raise ValueError(f"Unsupported probability shape {prob1.shape}")


def union_consensus(
    prob1: np.ndarray, prob2: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    if prob1.ndim != 2:
        raise ValueError(
            "union is meaningful only for binary masks/probabilities. Use daps_v6_average or daps_v7_entropy_weighted for VOC."
        )
    mask = (prob1 >= threshold) | (prob2 >= threshold)
    return mask.astype(np.float32), mask


def intersection_consensus(
    prob1: np.ndarray, prob2: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    if prob1.ndim != 2:
        raise ValueError(
            "intersection is meaningful only for binary masks/probabilities. Use daps_v6_average or daps_v7_entropy_weighted for VOC."
        )
    mask = (prob1 >= threshold) & (prob2 >= threshold)
    return mask.astype(np.float32), mask


def run_daps(
    mode: str, prob1: np.ndarray, prob2: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    if prob1.shape != prob2.shape:
        raise ValueError(f"Probability shapes differ: {prob1.shape} vs {prob2.shape}")
    if mode == "daps_v7_entropy_weighted":
        return daps_v7_entropy_weighted(prob1, prob2, threshold)
    if mode == "daps_v6_average":
        return daps_v6_average(prob1, prob2, threshold)
    if mode == "union":
        return union_consensus(prob1, prob2, threshold)
    if mode == "intersection":
        return intersection_consensus(prob1, prob2, threshold)
    raise ValueError(f"Unknown DAPS mode '{mode}'")
