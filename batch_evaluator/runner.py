from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from daps_core.consensus import run_daps
from daps_core.metrics import multiclass_iou
from daps_core.models import predict_voc
from daps_core.natural import (
    find_image_cases,
    find_processed_cases,
    load_image_case,
    load_processed_case,
)
from daps_core.voc import (
    IGNORE_INDEX,
    NUM_VOC_CLASSES,
    find_voc_cases,
    load_voc_case,
)

from .excel_export import write_excel_report


DATASET_VOC = "PASCAL_VOC2012"
DATASET_RAW = "DAPS raw_images"
DATASET_TEST = "DAPS test_images"
DATASET_PROCESSED = "DAPS processed_dataset"


def get_case_ids(
    dataset_type: str,
    dataset_root: str,
    voc_split: str,
) -> list[str]:
    """Trova tutti i case ID disponibili nel dataset."""

    if dataset_type == DATASET_VOC:
        cases = find_voc_cases(dataset_root, split=voc_split)

    elif dataset_type == DATASET_PROCESSED:
        cases = find_processed_cases(dataset_root)

    elif dataset_type in {DATASET_RAW, DATASET_TEST}:
        cases = find_image_cases(dataset_root)

    else:
        raise ValueError(f"Dataset non supportato: {dataset_type}")

    return [case.case_id for case in cases]


def load_case(
    dataset_type: str,
    dataset_root: str,
    voc_split: str,
    case_id: str,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Restituisce immagine RGB e reference.

    La reference è None per raw_images e test_images.
    """

    if dataset_type == DATASET_VOC:
        return load_voc_case(
            dataset_root,
            case_id,
            split=voc_split,
        )

    if dataset_type == DATASET_PROCESSED:
        image, reference, _source_path = load_processed_case(
            dataset_root,
            case_id,
        )
        return image, reference

    if dataset_type in {DATASET_RAW, DATASET_TEST}:
        image = load_image_case(dataset_root, case_id)
        return image, None

    raise ValueError(f"Dataset non supportato: {dataset_type}")


def calculate_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """
    Calcola mIoU, mIoU senza background e pixel accuracy.

    Può essere usata sia contro la ground truth sia fra due predizioni.
    """

    all_classes = multiclass_iou(
        prediction,
        reference,
        num_classes=NUM_VOC_CLASSES,
        ignore_index=IGNORE_INDEX,
        include_background=True,
    )

    foreground_only = multiclass_iou(
        prediction,
        reference,
        num_classes=NUM_VOC_CLASSES,
        ignore_index=IGNORE_INDEX,
        include_background=False,
    )

    return {
        "miou": float(all_classes["miou"]),
        "miou_no_bg": float(foreground_only["miou"]),
        "pixel_accuracy": float(all_classes["pixel_acc"]),
    }


def evaluate_case(
    dataset_type: str,
    dataset_root: str,
    voc_split: str,
    case_id: str,
    model_1_name: str,
    model_2_name: str,
    daps_mode: str,
    app_dir: str | Path,
) -> dict[str, Any]:
    """Esegue una valutazione completa su un singolo caso."""

    image, reference = load_case(
        dataset_type=dataset_type,
        dataset_root=dataset_root,
        voc_split=voc_split,
        case_id=case_id,
    )

    output_1 = predict_voc(
        model_1_name,
        image,
        case_id,
        app_dir,
    )

    output_2 = predict_voc(
        model_2_name,
        image,
        case_id,
        app_dir,
    )

    _, daps_mask = run_daps(
        daps_mode,
        output_1.prob,
        output_2.prob,
        threshold=0.5,
    )

    result: dict[str, Any] = {
        "case_id": case_id,
        "status": "completed",
        "error": None,
        "has_reference": reference is not None,

        # Confronti prediction-vs-prediction.
        "agreement_m1_m2": calculate_metrics(
            output_1.mask,
            output_2.mask,
        ),
        "agreement_daps_m1": calculate_metrics(
            daps_mask,
            output_1.mask,
        ),
        "agreement_daps_m2": calculate_metrics(
            daps_mask,
            output_2.mask,
        ),
    }

    # Confronti prediction-vs-ground-truth.
    if reference is not None:
        result["model_1"] = calculate_metrics(
            output_1.mask,
            reference,
        )

        result["model_2"] = calculate_metrics(
            output_2.mask,
            reference,
        )

        result["daps"] = calculate_metrics(
            daps_mask,
            reference,
        )

    else:
        result["model_1"] = None
        result["model_2"] = None
        result["daps"] = None

    return result


def run_batch(
    dataset_type: str,
    dataset_root: str,
    voc_split: str,
    model_1_name: str,
    model_2_name: str,
    daps_mode: str,
    app_dir: str | Path,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict[str, Any]]:
    """
    Esegue il batch su tutto il dataset.

    on_progress riceve:
    - casi completati;
    - casi totali;
    - case ID corrente.
    """

    case_ids = get_case_ids(
        dataset_type,
        dataset_root,
        voc_split,
    )

    if not case_ids:
        raise ValueError("Nessun caso trovato nel dataset selezionato.")

    results: list[dict[str, Any]] = []
    total_cases = len(case_ids)

    for completed, case_id in enumerate(case_ids, start=1):
        try:
            result = evaluate_case(
                dataset_type=dataset_type,
                dataset_root=dataset_root,
                voc_split=voc_split,
                case_id=case_id,
                model_1_name=model_1_name,
                model_2_name=model_2_name,
                daps_mode=daps_mode,
                app_dir=app_dir,
            )

        except Exception as error:
            # Un errore su un caso non interrompe tutto il batch.
            result = {
                "case_id": case_id,
                "status": "error",
                "error": str(error),
                "has_reference": None,
                "model_1": None,
                "model_2": None,
                "daps": None,
                "agreement_m1_m2": None,
                "agreement_daps_m1": None,
                "agreement_daps_m2": None,
            }

        results.append(result)

        if on_progress is not None:
            on_progress(completed, total_cases, case_id)

    return results


def run_batch_and_export(
    dataset_type: str,
    dataset_root: str,
    voc_split: str,
    model_1_name: str,
    model_2_name: str,
    daps_mode: str,
    app_dir: str | Path,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> Path:
    """Esegue il batch e restituisce il percorso del report Excel creato."""

    results = run_batch(
        dataset_type=dataset_type,
        dataset_root=dataset_root,
        voc_split=voc_split,
        model_1_name=model_1_name,
        model_2_name=model_2_name,
        daps_mode=daps_mode,
        app_dir=app_dir,
        on_progress=on_progress,
    )

    completed_cases = sum(
        result["status"] == "completed"
        for result in results
    )

    error_cases = sum(
        result["status"] == "error"
        for result in results
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_path = output_dir / f"daps_batch_{timestamp}.xlsx"

    run_info = {
        "Dataset type": dataset_type,
        "Dataset root": dataset_root,
        "VOC split": voc_split,
        "Model 1": model_1_name,
        "Model 2": model_2_name,
        "DAPS mode": daps_mode,
        "Total cases": len(results),
        "Completed cases": completed_cases,
        "Cases with error": error_cases,
    }

    return write_excel_report(
        output_path=output_path,
        results=results,
        run_info=run_info,
    )