from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import gradio as gr
import numpy as np

from daps_core.consensus import run_daps
from daps_core.metrics import multiclass_iou
from daps_core.models import (
    VOC_DEMO_MODELS,
    available_torchvision_voc_models,
    predict_voc,
    torchvision_is_available,
)
from daps_core.natural import (
    find_image_cases,
    find_processed_cases,
    load_image_case,
    load_processed_case,
)
from daps_core.voc import (
    IGNORE_INDEX,
    NUM_VOC_CLASSES,
    VOC_CLASSES,
    find_voc_cases,
    load_voc_case,
    overlay_voc_mask,
)

APP_DIR = Path(__file__).resolve().parent
DATASETS_DIR = APP_DIR / "datasets"

DEFAULT_VOC_ROOT = str(DATASETS_DIR / "voc")
DEFAULT_RAW_IMAGES_ROOT = str(DATASETS_DIR / "raw_images")
DEFAULT_TEST_IMAGES_ROOT = str(DATASETS_DIR / "test_images")
DEFAULT_PROCESSED_ROOT = str(DATASETS_DIR / "processed_dataset")
DATASET_VOC = "PASCAL_VOC2012"
DATASET_RAW = "DAPS raw_images"
DATASET_TEST = "DAPS test_images"
DATASET_PROCESSED = "DAPS processed_dataset"

DATASET_TYPES = [
    DATASET_VOC,
    DATASET_RAW,
    DATASET_TEST,
    DATASET_PROCESSED,
]
DAPS_MODES = [
    "daps_v7_entropy_weighted",
    "daps_v6_average",
]

VOC_SPLITS = ["val", "train", "trainval"]


def _is_natural_dataset(dataset_type: str) -> bool:
    return dataset_type in {DATASET_RAW, DATASET_TEST, DATASET_PROCESSED}


def _root_for_dataset(dataset_type: str) -> str:
    if dataset_type == DATASET_VOC:
        return DEFAULT_VOC_ROOT
    if dataset_type == DATASET_RAW:
        return DEFAULT_RAW_IMAGES_ROOT
    if dataset_type == DATASET_TEST:
        return DEFAULT_TEST_IMAGES_ROOT
    if dataset_type == DATASET_PROCESSED:
        return DEFAULT_PROCESSED_ROOT
    raise ValueError(f"Dataset non supportato: {dataset_type}")


def _voc_model_choices() -> List[str]:
    # voc_predictions/ is expected inside the same folder as app.py.
    models = VOC_DEMO_MODELS + available_torchvision_voc_models()
    # Add precomputed VOC predictions if they exist.
    try:
        from daps_core.voc import available_voc_prediction_models

        models = (
            VOC_DEMO_MODELS
            + available_voc_prediction_models(APP_DIR)
            + available_torchvision_voc_models()
        )
    except Exception:
        pass
    return models


def update_dataset_defaults(dataset_type: str) -> Tuple[Any, Any]:
    is_voc = dataset_type == DATASET_VOC
    return (
        gr.update(value=_root_for_dataset(dataset_type)),
        gr.update(visible=is_voc),
    )


def refresh(
    dataset_type: str, dataset_root: str, voc_split: str
) -> Tuple[Any, Any, Any, str]:
    if dataset_type == DATASET_VOC:
        cases = find_voc_cases(dataset_root, split=voc_split)
        case_ids = [c.case_id for c in cases]
        models = _voc_model_choices()
        msg = f"Trovati {len(case_ids)} casi PASCAL VOC2012 split={voc_split} in {dataset_root}."
        if not case_ids:
            msg += (
                " Controlla JPEGImages/, SegmentationClass/ e ImageSets/Segmentation/."
            )
        if not any(m.startswith("vocpred:") for m in models):
            msg += " Nessuna cartella voc_predictions/<nome_modello>/ trovata."
        if torchvision_is_available():
            msg += " Torchvision disponibile: puoi usare DeepLabV3/FCN/LRASPP pre-addestrati."
        else:
            msg += " Torchvision non installato: installa requirements_torch.txt per usare modelli reali pre-addestrati."
    elif dataset_type in {DATASET_RAW, DATASET_TEST}:
        cases = find_image_cases(dataset_root)
        case_ids = [c.case_id for c in cases]
        models = _voc_model_choices()
        msg = f"Trovate {len(case_ids)} immagini RGB in {dataset_root}."
        if not case_ids:
            msg += " Controlla che la cartella contenga .jpg/.jpeg/.png."
        msg += (
            " Non c'e' ground truth: la GUI mostrera' agreement IoU tra modelli e DAPS."
        )
        if torchvision_is_available():
            msg += " Torchvision disponibile per modelli VOC pre-addestrati."
        else:
            msg += (
                " Installa requirements_torch.txt per usare DeepLabV3/FCN/LRASPP reali."
            )
    elif dataset_type == DATASET_PROCESSED:
        cases = find_processed_cases(dataset_root)
        case_ids = [c.case_id for c in cases]
        models = _voc_model_choices()
        msg = f"Trovati {len(case_ids)} pacchetti .pt in {dataset_root}."
        if not case_ids:
            msg += " Controlla che processed_dataset contenga i .pt generati da build_offline_dataset.py."
        msg += " target_semantic viene usato come pseudo-ground-truth, non come annotazione umana."
        if not torchvision_is_available():
            msg += " Per leggere i .pt e/o usare modelli reali installa requirements_torch.txt."
    else:
        raise ValueError(f"Dataset non supportato: {dataset_type}")

    default_case = case_ids[0] if case_ids else None
    default_model_1 = models[0] if models else None
    default_model_2 = models[1] if len(models) > 1 else default_model_1
    return (
        gr.update(choices=case_ids, value=default_case),
        gr.update(choices=models, value=default_model_1),
        gr.update(choices=models, value=default_model_2),
        msg,
    )


def _voc_metrics_table(
    gt: np.ndarray, m1: np.ndarray, m2: np.ndarray, md: np.ndarray, ref_name: str = "GT"
) -> List[List[Any]]:
    rows = []
    for name, mask in [
        ("Model 1", m1),
        ("Model 2", m2),
        ("DAPS", md),
        ("Agreement M1-M2", m1),
    ]:
        if name == "Agreement M1-M2":
            stats_all = multiclass_iou(
                m1,
                m2,
                NUM_VOC_CLASSES,
                ignore_index=IGNORE_INDEX,
                include_background=True,
            )
            stats_fg = multiclass_iou(
                m1,
                m2,
                NUM_VOC_CLASSES,
                ignore_index=IGNORE_INDEX,
                include_background=False,
            )
            label = name
        else:
            stats_all = multiclass_iou(
                mask,
                gt,
                NUM_VOC_CLASSES,
                ignore_index=IGNORE_INDEX,
                include_background=True,
            )
            stats_fg = multiclass_iou(
                mask,
                gt,
                NUM_VOC_CLASSES,
                ignore_index=IGNORE_INDEX,
                include_background=False,
            )
            label = f"{name} vs {ref_name}"
        rows.append(
            [
                label,
                f"mIoU: {stats_all['miou']:.4f}",
                f"mIoU no-bg: {stats_fg['miou']:.4f}",
                f"Pixel acc: {stats_all['pixel_acc']:.4f}",
            ]
        )
    return rows


def _agreement_metrics_table(
    m1: np.ndarray, m2: np.ndarray, md: np.ndarray
) -> List[List[Any]]:
    pairs = [
        ("Agreement M1-M2", m1, m2),
        ("Agreement DAPS-M1", md, m1),
        ("Agreement DAPS-M2", md, m2),
    ]
    rows = []
    for name, a, b in pairs:
        stats_all = multiclass_iou(
            a, b, NUM_VOC_CLASSES, ignore_index=IGNORE_INDEX, include_background=True
        )
        stats_fg = multiclass_iou(
            a, b, NUM_VOC_CLASSES, ignore_index=IGNORE_INDEX, include_background=False
        )
        rows.append(
            [
                name,
                f"mIoU agreement: {stats_all['miou']:.4f}",
                f"mIoU no-bg: {stats_fg['miou']:.4f}",
                f"Pixel agreement: {stats_all['pixel_acc']:.4f}",
            ]
        )
    return rows


def _class_list(mask: Optional[np.ndarray]) -> str:
    if mask is None:
        return "non disponibile"
    classes = sorted(int(x) for x in np.unique(mask) if x not in (0, IGNORE_INDEX))
    return (
        ", ".join(VOC_CLASSES[c] for c in classes if c < len(VOC_CLASSES))
        or "solo background/void"
    )


def _run_voc_like(
    dataset_label: str,
    image: np.ndarray,
    reference: Optional[np.ndarray],
    case_id: str,
    model_1_name: str,
    model_2_name: str,
    daps_mode: str,
    extra_source: str = "",
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[Any]], str
]:
    if daps_mode in {"union", "intersection"}:
        raise gr.Error(
            "Per immagini RGB/multiclasse usa daps_v7_entropy_weighted oppure daps_v6_average. Union/intersection sono solo per segmentazione binaria."
        )

    out1 = predict_voc(model_1_name, image, case_id, APP_DIR)
    out2 = predict_voc(model_2_name, image, case_id, APP_DIR)
    _, daps_mask = run_daps(daps_mode, out1.prob, out2.prob, threshold=0.5)

    original_img = image
    if reference is None:
        # Use a neutral blank image for the Reference panel when no GT/pseudo-GT exists.
        ref_img = np.full_like(image, 245, dtype=np.uint8)
        table = _agreement_metrics_table(out1.mask, out2.mask, daps_mask)
        ref_note = "Reference: non disponibile; metriche = agreement tra segmentazioni, non IoU rispetto a ground truth."
    else:
        ref_img = overlay_voc_mask(image, reference, alpha=0.55)
        table = _voc_metrics_table(
            reference, out1.mask, out2.mask, daps_mask, ref_name="reference"
        )
        ref_note = "Reference: target_semantic/pseudo-GT disponibile; non e' annotazione umana se viene da processed_dataset."

    m1_img = overlay_voc_mask(image, out1.mask, alpha=0.55)
    m2_img = overlay_voc_mask(image, out2.mask, alpha=0.55)
    daps_img = overlay_voc_mask(image, daps_mask, alpha=0.55)

    changed_vs_m1 = float(np.mean(daps_mask != out1.mask) * 100.0)
    changed_vs_m2 = float(np.mean(daps_mask != out2.mask) * 100.0)

    note = (
        f"Dataset: {dataset_label}\n"
        f"Case: {case_id}\n"
        f"{extra_source}\n"
        f"Classi reference: {_class_list(reference)}\n"
        f"Classi Model 1: {_class_list(out1.mask)}\n"
        f"Classi Model 2: {_class_list(out2.mask)}\n"
        f"Classi DAPS: {_class_list(daps_mask)}\n"
        f"Model 1 source: {out1.source}\n"
        f"Model 2 source: {out2.source}\n"
        f"DAPS mode: {daps_mode}\n"
        f"DAPS difference rate vs Model 1: {changed_vs_m1:.2f}%\n"
        f"DAPS difference rate vs Model 2: {changed_vs_m2:.2f}%\n"
        f"{ref_note}"
    )

    return original_img, ref_img, m1_img, m2_img, daps_img, table, note


def _run_voc(
    dataset_root: str,
    voc_split: str,
    case_id: str,
    model_1_name: str,
    model_2_name: str,
    daps_mode: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[Any]], str
]:
    image, gt = load_voc_case(dataset_root, case_id, split=voc_split)
    if gt is None:
        raise gr.Error(
            "Ground truth VOC non trovata. Usa casi presenti in SegmentationClass/."
        )

    return _run_voc_like(
        dataset_label=f"PASCAL VOC2012 split={voc_split}",
        image=image,
        reference=gt,
        case_id=case_id,
        model_1_name=model_1_name,
        model_2_name=model_2_name,
        daps_mode=daps_mode,
        extra_source="Nota: i pixel VOC con label 255 sono ignore/void e non entrano nelle metriche.",
    )


def _run_natural(
    dataset_type: str,
    dataset_root: str,
    case_id: str,
    model_1_name: str,
    model_2_name: str,
    daps_mode: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[Any]], str
]:
    if dataset_type in {DATASET_RAW, DATASET_TEST}:
        image = load_image_case(dataset_root, case_id)
        return _run_voc_like(
            dataset_label=dataset_type,
            image=image,
            reference=None,
            case_id=case_id,
            model_1_name=model_1_name,
            model_2_name=model_2_name,
            daps_mode=daps_mode,
            extra_source=f"Image root: {dataset_root}",
        )

    if dataset_type == DATASET_PROCESSED:
        image, reference, source_path = load_processed_case(dataset_root, case_id)
        return _run_voc_like(
            dataset_label=dataset_type,
            image=image,
            reference=reference,
            case_id=case_id,
            model_1_name=model_1_name,
            model_2_name=model_2_name,
            daps_mode=daps_mode,
            extra_source=f"Processed package: {source_path}",
        )

    raise gr.Error(f"Dataset naturale non supportato: {dataset_type}")


def run_demo(
    dataset_type: str,
    dataset_root: str,
    voc_split: str,
    case_id: str,
    model_1_name: str,
    model_2_name: str,
    daps_mode: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[Any]], str
]:
    if not case_id:
        raise gr.Error("Seleziona un caso valido.")
    if not model_1_name or not model_2_name:
        raise gr.Error("Seleziona due modelli.")

    if dataset_type == DATASET_VOC:
        return _run_voc(
            dataset_root, voc_split, case_id, model_1_name, model_2_name, daps_mode
        )
    if _is_natural_dataset(dataset_type):
        return _run_natural(
            dataset_type, dataset_root, case_id, model_1_name, model_2_name, daps_mode
        )

    raise gr.Error(f"Dataset non supportato: {dataset_type}")


with gr.Blocks(title="Consenso DAPS - Semantic Segmentation MVP") as demo:
    gr.Markdown("# DAPS4Massimo - DAPS Segmentation MVP")
    gr.Markdown(
        "Seleziona  PASCAL VOC2012 oppure le cartelle originali DAPS "
        "raw_images / test_images / processed_dataset. La GUI visualizza Model 1, Model 2 e DAPS."
    )

    with gr.Row():
        dataset_type = gr.Dropdown(
            label="Dataset / sorgente", choices=DATASET_TYPES, value=DATASET_VOC
        )
        dataset_root = gr.Textbox(label="Dataset root", value=DEFAULT_VOC_ROOT, scale=3)
        refresh_btn = gr.Button("Refresh cases/models", scale=1)

    with gr.Row():
        voc_split = gr.Dropdown(
            label="VOC split", choices=VOC_SPLITS, value="val", visible=False
        )

    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        case_dropdown = gr.Dropdown(label="Caso / immagine", choices=[], value=None)

    with gr.Row():
        model_1 = gr.Dropdown(
            label="Modello 1", choices=VOC_DEMO_MODELS, value=VOC_DEMO_MODELS[0]
        )
        model_2 = gr.Dropdown(
            label="Modello 2", choices=VOC_DEMO_MODELS, value=VOC_DEMO_MODELS[1]
        )
        daps_mode = gr.Dropdown(
            label="DAPS mode", choices=DAPS_MODES, value="daps_v7_entropy_weighted"
        )

    run_btn = gr.Button("Run segmentation", variant="primary")

    with gr.Row():
        original = gr.Image(label="Original image")
        gt_img = gr.Image(label="Ground Truth / Reference")
        m1_img = gr.Image(label="Model 1")
        m2_img = gr.Image(label="Model 2")
        daps_img = gr.Image(label="DAPS")

    metrics = gr.Dataframe(
        headers=["Segmentation", "Metric 1", "Metric 2", "Metric 3"],
        label="Metriche",
        interactive=False,
    )
    notes = gr.Textbox(label="Dettagli", lines=13, interactive=False)

    dataset_type.change(
        update_dataset_defaults,
        inputs=[dataset_type],
        outputs=[dataset_root, voc_split],
    ).then(
        refresh,
        inputs=[dataset_type, dataset_root, voc_split],
        outputs=[case_dropdown, model_1, model_2, status],
    )

    refresh_btn.click(
        refresh,
        inputs=[dataset_type, dataset_root, voc_split],
        outputs=[case_dropdown, model_1, model_2, status],
    )

    voc_split.change(
        refresh,
        inputs=[dataset_type, dataset_root, voc_split],
        outputs=[case_dropdown, model_1, model_2, status],
    )

    run_btn.click(
        run_demo,
        inputs=[
            dataset_type,
            dataset_root,
            voc_split,
            case_dropdown,
            model_1,
            model_2,
            daps_mode,
        ],
        outputs=[original, gt_img, m1_img, m2_img, daps_img, metrics, notes],
    )

    demo.load(
        refresh,
        inputs=[dataset_type, dataset_root, voc_split],
        outputs=[case_dropdown, model_1, model_2, status],
    )


if __name__ == "__main__":
    demo.launch()
