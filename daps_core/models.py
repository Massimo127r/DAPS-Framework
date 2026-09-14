from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from .voc import (
    NUM_VOC_CLASSES,
    find_voc_prediction_file,
    mask_from_prob,
    one_hot_from_mask,
)


@dataclass
class ModelOutput:
    name: str
    prob: np.ndarray
    mask: np.ndarray
    source: str


VOC_DEMO_MODELS = [
    "voc_demo_red_green",
    "voc_demo_blue_bright",
]

TORCHVISION_VOC_MODELS = [
    "torchvision:deeplabv3_resnet50",
    "torchvision:fcn_resnet50",
    "torchvision:lraspp_mobilenet_v3_large",
]


def torchvision_is_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except Exception:
        return False


def available_torchvision_voc_models() -> List[str]:
    return TORCHVISION_VOC_MODELS if torchvision_is_available() else []


def _voc_demo_red_green(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Very weak RGB heuristic, only for UI testing without real models."""
    img = image.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    h, w = r.shape
    prob = np.zeros((NUM_VOC_CLASSES, h, w), dtype=np.float32)
    prob[0] = 0.70
    prob[15] = np.clip(0.60 * r + 0.25 * g - 0.15 * b, 0.0, 1.0)  # person-ish
    prob[8] = np.clip(0.45 * r + 0.20 * b - 0.10 * g, 0.0, 1.0)  # cat-ish
    prob[12] = np.clip(0.25 * r + 0.35 * g + 0.15 * b, 0.0, 1.0)  # dog-ish
    prob = prob / np.maximum(prob.sum(axis=0, keepdims=True), 1e-6)
    return prob, mask_from_prob(prob)


def _voc_demo_blue_bright(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Very weak RGB heuristic, only for UI testing without real models."""
    img = image.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    brightness = (r + g + b) / 3.0
    h, w = r.shape
    prob = np.zeros((NUM_VOC_CLASSES, h, w), dtype=np.float32)
    prob[0] = 0.75
    prob[6] = np.clip(0.35 * r + 0.25 * g + 0.10 * brightness, 0.0, 1.0)  # bus-ish
    prob[7] = np.clip(0.35 * b + 0.20 * brightness, 0.0, 1.0)  # car-ish
    prob[4] = np.clip(0.45 * b - 0.10 * r + 0.20 * brightness, 0.0, 1.0)  # boat-ish
    prob = prob / np.maximum(prob.sum(axis=0, keepdims=True), 1e-6)
    return prob, mask_from_prob(prob)


def _load_voc_prediction(
    path: Path, expected_hw: tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr)
        if arr.ndim == 3:
            # Expected [C,H,W] probability/logit tensor. If it does not look like
            # probabilities, softmax it.
            prob = arr.astype(np.float32)
            if prob.shape[1:] != expected_hw:
                raise ValueError(
                    f"VOC prediction {path} has shape {prob.shape}, expected Cx{expected_hw[0]}x{expected_hw[1]}"
                )
            if (
                float(prob.min()) < 0.0
                or float(prob.max()) > 1.0
                or not np.allclose(prob.sum(axis=0), 1.0, atol=1e-2)
            ):
                prob = prob - prob.max(axis=0, keepdims=True)
                exp = np.exp(prob)
                prob = exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-6)
            return prob.astype(np.float32), mask_from_prob(prob)
        if arr.ndim == 2:
            mask = arr.astype(np.uint8)
            if mask.shape != expected_hw:
                raise ValueError(
                    f"VOC prediction {path} has shape {mask.shape}, expected {expected_hw}"
                )
            return one_hot_from_mask(mask), mask
        raise ValueError(f"Unsupported .npy prediction shape {arr.shape} for {path}")

    mask = np.asarray(Image.open(path), dtype=np.uint8)
    if mask.shape != expected_hw:
        raise ValueError(
            f"VOC prediction {path} has shape {mask.shape}, expected {expected_hw}"
        )
    return one_hot_from_mask(mask), mask


@lru_cache(maxsize=3)
def _load_torchvision_model(model_key: str):
    import torch
    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        FCN_ResNet50_Weights,
        LRASPP_MobileNet_V3_Large_Weights,
        deeplabv3_resnet50,
        fcn_resnet50,
        lraspp_mobilenet_v3_large,
    )

    if model_key == "deeplabv3_resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
    elif model_key == "fcn_resnet50":
        weights = FCN_ResNet50_Weights.DEFAULT
        model = fcn_resnet50(weights=weights)
    elif model_key == "lraspp_mobilenet_v3_large":
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = lraspp_mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unknown torchvision model key '{model_key}'")

    model.eval()
    return model, weights.transforms(), torch.device("cpu")


def _predict_torchvision_voc(
    model_name: str, image: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Run a torchvision semantic segmenter and return probabilities at image size.

    The original daps_translator_06 pipeline did not use the torchvision
    weight-specific resize transform. It resized every natural image to 512x512,
    converted it with T.ToTensor(), and then interpolated logits to the working
    size. The previous GUI used weights.transforms(); that can change the spatial
    size internally, so raw_images/test_images/processed_dataset could fail when
    overlays/metrics expected the original HxW.

    This adapter preserves the 512x512 resize, applies the selected weights'
    channel normalization, and always resizes logits
    back to the original image height/width before returning them.
    """
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T

    model_key = model_name.split(":", 1)[1]
    model, weight_preprocess, device = _load_torchvision_model(model_key)

    orig_h, orig_w = image.shape[:2]
    pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")

    # Match daps_translator_06/test_inferenza_v6.py and build_offline_dataset.py:
    # Image.open(...).convert("RGB").resize((512, 512)) + T.ToTensor().
    pil_512 = pil.resize((512, 512))
    tensor = T.ToTensor()(pil_512)
    tensor = T.Normalize(mean=weight_preprocess.mean, std=weight_preprocess.std)(tensor)
    batch = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(batch)["out"]
        if logits.shape[-2:] != (orig_h, orig_w):
            logits = F.interpolate(
                logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
        prob_t = torch.softmax(logits[0], dim=0)

    prob = prob_t.cpu().numpy().astype(np.float32)
    mask = np.argmax(prob, axis=0).astype(np.uint8)
    return (
        prob,
        mask,
        f"torchvision pretrained {model_key}; RGB normalization from pretrained weights; resize 512 -> original {orig_h}x{orig_w}; first run may download weights",
    )


def predict_voc(
    model_name: str, image: np.ndarray, case_id: str, base_dir: str | Path
) -> ModelOutput:
    """Run one PASCAL VOC semantic-segmentation adapter.

    Built-ins:
        voc_demo_red_green, voc_demo_blue_bright only test the UI.

    Precomputed predictions:
        vocpred:<folder>, loaded from ./voc_predictions/<folder>/<case_id>.png or .npy

    Optional real models:
        torchvision:<name>, if torch/torchvision are installed.
    """
    if model_name == "voc_demo_red_green":
        prob, mask = _voc_demo_red_green(image)
        return ModelOutput(
            model_name, prob, mask, "Demo RGB heuristic, not a real segmenter"
        )

    if model_name == "voc_demo_blue_bright":
        prob, mask = _voc_demo_blue_bright(image)
        return ModelOutput(
            model_name, prob, mask, "Demo RGB heuristic, not a real segmenter"
        )

    if model_name.startswith("vocpred:"):
        pred_path = find_voc_prediction_file(base_dir, model_name, case_id)
        if pred_path is None:
            raise FileNotFoundError(
                f"Prediction for VOC case {case_id} not found for {model_name}. "
                f"Expected ./voc_predictions/{model_name.split(':', 1)[1]}/{case_id}.png or .npy"
            )
        h, w = image.shape[:2]
        prob, mask = _load_voc_prediction(pred_path, (h, w))
        return ModelOutput(model_name, prob, mask, str(pred_path))

    if model_name.startswith("torchvision:"):
        if not torchvision_is_available():
            raise ImportError(
                "torch/torchvision non sono installati. Installa con: pip install -r requirements_torch.txt"
            )
        prob, mask, source = _predict_torchvision_voc(model_name, image)
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Torchvision output shape {mask.shape} differs from image shape {image.shape[:2]}"
            )
        return ModelOutput(model_name, prob, mask, source)

    raise ValueError(f"Unknown VOC model '{model_name}'")
