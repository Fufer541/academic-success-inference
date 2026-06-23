from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_ID = "academic-success-terminal-binary-ensemble-v1"

MODEL_NAME = "Academic Success Binary Terminal Ensemble"

EXPECTED_LABEL_ORDER = ["Dropout", "Graduate"]
EXPECTED_PROBLEM_TYPE = "binary_terminal_outcome_with_pseudo_labeling"

ARTIFACT_NAME = "academic_success_enrollment_catboost_lightgbm_bundle.pkl"


class ModelServiceError(RuntimeError):
    """Raised when the model bundle can't serve an inference request."""


def _project_root() -> Path:
    return Path(__file__).parents[2]


def _candidate_model_paths() -> list[Path]:

    configured_path = os.getenv("MODEL_BUNDLE_PATH")
    candidates: list[Path] = []
    if configured_path:
        candidates.append(Path(configured_path).expanduser())

    root = _project_root()
    candidates.extend(
        [
            root / "research" / "pickles" / ARTIFACT_NAME,
            root.parent / "research" / "pickles" / ARTIFACT_NAME,
            Path("/models") / ARTIFACT_NAME,
        ]
    )
    return candidates


def get_model_path() -> Path:
    for path in _candidate_model_paths():
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in _candidate_model_paths())
    raise FileNotFoundError(f"Model bundle not found. Searched: {searched}")


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    path = get_model_path()
    bundle = joblib.load(path)

    required_keys = {
        "features",
        "categorical_features",
        "numeric_features",
        "label_order",
        "id_to_label",
        "models",
        "ensemble",
    }

    missing_keys = sorted(required_keys - set(bundle))
    if missing_keys:
        raise ModelServiceError(
            f"Model bundle is missing required keys: {', '.join(missing_keys)}"
        )

    label_order = list(bundle.get("label_order", []))
    if label_order != EXPECTED_LABEL_ORDER:
        raise ModelServiceError(
            "Model bundle must expose binary labels "
            f"{EXPECTED_LABEL_ORDER}; got {label_order}"
        )

    problem_type = bundle.get("problem_type")
    if problem_type != EXPECTED_PROBLEM_TYPE:
        raise ModelServiceError(
            "Model bundle problem_type must be "
            f"{EXPECTED_PROBLEM_TYPE!r}; got {problem_type!r}"
        )

    return bundle


def bundle_status() -> dict[str, Any]:
    """Return health-check metadata without letting model-load errors escape."""

    try:
        bundle = load_bundle()
    except Exception as exc:  # pragma: no cover - surfaced through /health
        return {
            "loaded": False,
            "error": str(exc),
            "path": str(_candidate_model_paths()[0]),
        }

    return {
        "loaded": True,
        "path": str(get_model_path()),
        "phase": bundle.get("phase"),
        "feature_count": len(bundle["features"]),
        "labels": list(bundle["label_order"]),
        "problem_type": bundle.get("problem_type"),
        "pseudo_labeling": _pseudo_labeling_metadata(bundle),
    }


def _to_native_number(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _pseudo_labeling_metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    pseudo = bundle.get("pseudo_labeling") or {}
    return {
        "enabled": bool(pseudo.get("enabled", False)),
        "threshold": _to_native_number(pseudo.get("threshold")),
        "unresolvedLabel": pseudo.get(
            "unresolved_label", bundle.get("unresolved_label")
        ),
        "unresolvedRows": _to_native_number(pseudo.get("unresolved_rows")),
        "selectedRows": _to_native_number(pseudo.get("selected_rows")),
        "selectedDropout": _to_native_number(pseudo.get("selected_dropout")),
        "selectedGraduate": _to_native_number(pseudo.get("selected_graduate")),
        "confidenceMin": _to_native_number(pseudo.get("confidence_min")),
        "confidenceMean": _to_native_number(pseudo.get("confidence_mean")),
    }


def model_metadata() -> dict[str, Any]:
    bundle = load_bundle()
    categorical_features = set(bundle["categorical_features"])
    numeric_features = set(bundle["numeric_features"])

    return {
        "id": MODEL_ID,
        "name": MODEL_NAME,
        "description": "Binary CatBoost and LightGBM ensemble that estimates terminal student outcome after pseudo-labeling confident enrolled records.",
        "phase": bundle.get("phase", "enrollment"),
        "problemType": bundle.get("problem_type"),
        "unresolvedLabel": bundle.get("unresolved_label"),
        "pseudoLabeling": _pseudo_labeling_metadata(bundle),
        # The frontend renders its form dynamically from this list, so feature
        # ordering stays identical to training and no field names are duplicated
        # in browser code.
        "features": [
            {
                "name": feature,
                "kind": (
                    "categorical"
                    if feature in categorical_features
                    else "numeric" if feature in numeric_features else "unknown"
                ),
                "required": True,
            }
            for feature in bundle["features"]
        ],
        "labels": list(bundle["label_order"]),
        "metrics": bundle.get("metrics", {}),
    }


def normalize_feature_name(name: str, bundle: dict[str, Any]) -> str:
    cleaned = name.replace("\ufeff", "").strip()
    return bundle.get("column_renames", {}).get(cleaned, cleaned)


def coerce_categorical_value(value: Any) -> Any:
    if isinstance(value, bool):
        # Python treats bool as a subclass of int, but handling it explicitly
        # makes binary feature conversion clear and keeps category values as 0/1.
        return int(value)
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        try:
            number = float(stripped)
        except ValueError:
            return stripped
        return int(number) if number.is_integer() else number
    return value


def prepare_frames(
    features: dict[str, Any], bundle: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = {
        normalize_feature_name(name, bundle): value for name, value in features.items()
    }
    required_features = list(bundle["features"])

    missing_features = [
        feature for feature in required_features if feature not in normalized
    ]
    if missing_features:
        raise ModelServiceError(
            "Missing required feature"
            + ("s" if len(missing_features) != 1 else "")
            + f": {', '.join(missing_features)}"
        )

    row = {feature: normalized[feature] for feature in required_features}
    empty_features = [
        feature
        for feature, value in row.items()
        if value is None or (isinstance(value, str) and not value.strip())
    ]
    if empty_features:
        raise ModelServiceError(
            "Empty value for required feature"
            + ("s" if len(empty_features) != 1 else "")
            + f": {', '.join(empty_features)}"
        )

    row_df = pd.DataFrame([row], columns=required_features)

    for column in bundle["numeric_features"]:
        row_df[column] = pd.to_numeric(row_df[column], errors="coerce")
        if row_df[column].isna().any():
            raise ModelServiceError(f"Feature '{column}' must be numeric")

    catboost_frame = row_df.copy()
    lightgbm_frame = row_df.copy()
    for column in bundle["categorical_features"]:
        categorical_values = row_df[column].map(coerce_categorical_value)
        catboost_frame[column] = categorical_values.astype("string").fillna(
            "__missing__"
        )
        lightgbm_frame[column] = categorical_values.astype("category")

    return catboost_frame, lightgbm_frame


def align_proba(model: Any, frame: pd.DataFrame, class_count: int) -> np.ndarray:
    proba = np.asarray(model.predict_proba(frame), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    aligned = np.zeros((proba.shape[0], class_count), dtype=float)
    for source_index, class_id in enumerate(classes):
        aligned[:, int(class_id)] = proba[:, source_index]
    return aligned


def weighted_average_proba(
    probas: list[np.ndarray], weights: list[float]
) -> np.ndarray:

    normalized_weights = np.asarray(weights, dtype=float)
    total_weight = normalized_weights.sum()
    if total_weight <= 0:
        raise ModelServiceError("Model ensemble weights are invalid")
    normalized_weights = normalized_weights / total_weight
    stacked = np.stack(probas, axis=0)
    return np.tensordot(normalized_weights, stacked, axes=(0, 0))


def predict(features: dict[str, Any]) -> dict[str, Any]:

    bundle = load_bundle()
    catboost_frame, lightgbm_frame = prepare_frames(features, bundle)

    class_count = len(bundle["label_order"])

    catboost_proba = align_proba(
        bundle["models"]["catboost"], catboost_frame, class_count
    )
    lightgbm_proba = align_proba(
        bundle["models"]["lightgbm"], lightgbm_frame, class_count
    )

    proba = weighted_average_proba(
        [catboost_proba, lightgbm_proba],
        [
            float(bundle["ensemble"]["catboost_weight"]),
            float(bundle["ensemble"]["lightgbm_weight"]),
        ],
    )[0]

    predicted_id = int(np.argmax(proba))
    labels = list(bundle["label_order"])

    probabilities = {
        label: float(probability)
        for label, probability in zip(labels, proba, strict=True)
    }
    return {
        "model": MODEL_ID,
        "prediction": bundle["id_to_label"][predicted_id],
        "confidence": float(proba[predicted_id]),
        "probabilities": probabilities,
        "modelVersion": f"{MODEL_ID}-{bundle.get('phase', 'unknown')}",
    }