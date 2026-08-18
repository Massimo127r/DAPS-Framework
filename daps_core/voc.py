from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

NUM_VOC_CLASSES = len(VOC_CLASSES)
IGNORE_INDEX = 255

# Official PASCAL VOC palette for the first 21 labels.
VOC_PALETTE = np.asarray(
    [
        (0, 0, 0),
        (128, 0, 0),
        (0, 128, 0),
        (128, 128, 0),
        (0, 0, 128),
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 128),
        (64, 0, 0),
        (192, 0, 0),
        (64, 128, 0),
        (192, 128, 0),
        (64, 0, 128),
        (192, 0, 128),
        (64, 128, 128),
        (192, 128, 128),
        (0, 64, 0),
        (128, 64, 0),
        (0, 192, 0),
        (128, 192, 0),
        (0, 64, 128),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class VOCCasePaths:
    case_id: str
    image_path: Path
    label_path: Optional[Path]


def _read_split_ids(voc_root: Path, split: str) -> List[str]:
    split_file = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not split_file.exists():
        return []
    return [
        line.strip() for line in split_file.read_text().splitlines() if line.strip()
    ]


def find_voc_cases(voc_root: str | Path, split: str = "val") -> List[VOCCasePaths]:
    """Find PASCAL VOC2012 segmentation cases.

    Expected layout:
        VOCdevkit/VOC2012/JPEGImages/*.jpg
        VOCdevkit/VOC2012/SegmentationClass/*.png
        VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt
    """
    root = Path(voc_root).expanduser().resolve()
    image_dir = root / "JPEGImages"
    label_dir = root / "SegmentationClass"
    if not image_dir.exists():
        return []

    ids = _read_split_ids(root, split)
    if not ids:
        ids = sorted(p.stem for p in label_dir.glob("*.png"))

    cases: List[VOCCasePaths] = []
    for case_id in ids:
        image_path = image_dir / f"{case_id}.jpg"
        label_path = label_dir / f"{case_id}.png"
        if image_path.exists():
            cases.append(
                VOCCasePaths(
                    case_id=case_id,
                    image_path=image_path,
                    label_path=label_path if label_path.exists() else None,
                )
            )
    return cases


def voc_case_map(voc_root: str | Path, split: str = "val") -> Dict[str, VOCCasePaths]:
    return {c.case_id: c for c in find_voc_cases(voc_root, split)}


def load_voc_case(
    voc_root: str | Path, case_id: str, split: str = "val"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    cases = voc_case_map(voc_root, split)
    if case_id not in cases:
        raise FileNotFoundError(f"VOC case '{case_id}' not found in {voc_root}")
    cp = cases[case_id]
    image = np.asarray(Image.open(cp.image_path).convert("RGB"), dtype=np.uint8)
    label = None
    if cp.label_path is not None:
        label = np.asarray(Image.open(cp.label_path), dtype=np.uint8)
    return image, label


def one_hot_from_mask(
    mask: np.ndarray,
    num_classes: int = NUM_VOC_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    mask = np.asarray(mask)
    h, w = mask.shape
    prob = np.zeros((num_classes, h, w), dtype=np.float32)
    valid = mask != ignore_index
    safe = np.where(valid, mask, 0)
    safe = np.clip(safe, 0, num_classes - 1)
    prob[safe, np.arange(h)[:, None], np.arange(w)[None, :]] = valid.astype(np.float32)
    return prob


def mask_from_prob(prob: np.ndarray) -> np.ndarray:
    return np.argmax(prob, axis=0).astype(np.uint8)


def colorize_voc_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    valid = mask != IGNORE_INDEX
    clipped = np.clip(mask, 0, NUM_VOC_CLASSES - 1)
    out[valid] = VOC_PALETTE[clipped[valid]]
    out[~valid] = np.asarray([255, 255, 255], dtype=np.uint8)
    return out


def overlay_voc_mask(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.50
) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32)
    color = colorize_voc_mask(mask).astype(np.float32)
    valid = np.asarray(mask) != IGNORE_INDEX
    foreground = valid & (np.asarray(mask) != 0)
    out = base.copy()
    out[foreground] = (1.0 - alpha) * out[foreground] + alpha * color[foreground]
    return np.clip(out, 0, 255).astype(np.uint8)


def available_voc_prediction_models(base_dir: str | Path) -> List[str]:
    """Return folders under ./voc_predictions as model names 'vocpred:<folder>'."""
    base = Path(base_dir).expanduser().resolve()
    pred_root = base / "voc_predictions"
    if not pred_root.exists():
        return []
    models: List[str] = []
    for child in sorted(pred_root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.png")) or any(child.glob("*.npy")):
            models.append(f"vocpred:{child.name}")
    return models


def find_voc_prediction_file(
    base_dir: str | Path, pred_model_name: str, case_id: str
) -> Optional[Path]:
    if not pred_model_name.startswith("vocpred:"):
        return None
    folder = pred_model_name.split(":", 1)[1]
    root = Path(base_dir).expanduser().resolve() / "voc_predictions" / folder
    candidates = [
        root / f"{case_id}.png",
        root / f"{case_id}_pred.png",
        root / f"{case_id}.npy",
        root / f"{case_id}_prob.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = sorted(
        list(root.glob(f"*{case_id}*.png")) + list(root.glob(f"*{case_id}*.npy"))
    )
    return matches[0] if matches else None
