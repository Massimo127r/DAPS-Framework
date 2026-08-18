from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class NaturalCasePaths:
    case_id: str
    image_path: Path
    reference_path: Optional[Path] = None


@dataclass(frozen=True)
class ProcessedCasePaths:
    case_id: str
    pt_path: Path


def _safe_case_id(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return str(rel.with_suffix(""))


def find_image_cases(root_dir: str | Path) -> List[NaturalCasePaths]:
    """Find RGB images in raw_images/ or test_images.

    The search is recursive so subfolders are allowed. Hidden files and macOS
    metadata files are skipped.
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    cases: List[NaturalCasePaths] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        cases.append(
            NaturalCasePaths(case_id=_safe_case_id(root, path), image_path=path)
        )
    return cases


def image_case_map(root_dir: str | Path) -> Dict[str, NaturalCasePaths]:
    return {c.case_id: c for c in find_image_cases(root_dir)}


def load_image_case(root_dir: str | Path, case_id: str) -> np.ndarray:
    cases = image_case_map(root_dir)
    if case_id not in cases:
        raise FileNotFoundError(f"Image case '{case_id}' not found in {root_dir}")
    return np.asarray(
        Image.open(cases[case_id].image_path).convert("RGB"), dtype=np.uint8
    )


def find_processed_cases(root_dir: str | Path) -> List[ProcessedCasePaths]:
    """Find .pt packages produced by the original build_offline_dataset.py.

    Expected package keys:
        image: Tensor [3,H,W]
        target_semantic: Tensor [H,W]
        target_instance: dict, optional and not required by this GUI
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    cases: List[ProcessedCasePaths] = []
    for path in sorted(root.rglob("*.pt")):
        if not path.is_file() or path.name.startswith("."):
            continue
        cases.append(
            ProcessedCasePaths(case_id=_safe_case_id(root, path), pt_path=path)
        )
    return cases


def processed_case_map(root_dir: str | Path) -> Dict[str, ProcessedCasePaths]:
    return {c.case_id: c for c in find_processed_cases(root_dir)}


def _tensor_to_numpy(x):
    # Avoid importing torch unless a processed_dataset case is actually loaded.
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _image_tensor_to_uint8_rgb(image_arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_arr)
    if arr.ndim != 3:
        raise ValueError(f"Processed image must be 3D, got shape {arr.shape}")

    # [C,H,W] -> [H,W,C]
    if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)
    if arr.min() >= 0.0 and arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def load_processed_case(
    root_dir: str | Path, case_id: str
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    cases = processed_case_map(root_dir)
    if case_id not in cases:
        raise FileNotFoundError(f"Processed case '{case_id}' not found in {root_dir}")

    try:
        import torch
    except Exception as exc:
        raise ImportError(
            "Per leggere processed_dataset/*.pt devi installare torch: pip install -r requirements_torch.txt"
        ) from exc

    cp = cases[case_id]
    try:
        data = torch.load(cp.pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(cp.pt_path, map_location="cpu")

    if not isinstance(data, dict) or "image" not in data:
        raise ValueError(
            f"Processed package {cp.pt_path} is not a DAPS dictionary with key 'image'."
        )

    image = _image_tensor_to_uint8_rgb(_tensor_to_numpy(data["image"]))
    reference = None
    if "target_semantic" in data and data["target_semantic"] is not None:
        reference = _tensor_to_numpy(data["target_semantic"]).astype(np.uint8)
        if reference.ndim == 3:
            # Accept [1,H,W] or [H,W,1]
            reference = np.squeeze(reference)
        if reference.shape != image.shape[:2]:
            raise ValueError(
                f"target_semantic shape {reference.shape} differs from image shape {image.shape[:2]} in {cp.pt_path}"
            )
    return image, reference, str(cp.pt_path)
