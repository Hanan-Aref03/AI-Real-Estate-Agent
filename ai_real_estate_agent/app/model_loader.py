"""Model loading and prediction helpers for the FastAPI backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
TRAIN_STATS_PATH = MODEL_DIR / "train_stats.json"


class ModelArtifactsError(RuntimeError):
    """Raised when required model artifacts are unavailable or invalid."""


class ModelRegistry:
    """In-memory registry for model artifacts loaded at application startup."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.feature_names: list[str] = []
        self.train_stats: dict[str, Any] = {}

    def load(self) -> None:
        """Load the trained model, feature names, and training statistics."""

        if not TRAIN_STATS_PATH.exists():
            raise ModelArtifactsError("Missing required model artifact: train_stats.json")

        with TRAIN_STATS_PATH.open("r", encoding="utf-8") as file_handle:
            self.train_stats = json.load(file_handle)

        loaded_feature_names: Any = None
        if FEATURE_NAMES_PATH.exists():
            loaded_feature_names = joblib.load(FEATURE_NAMES_PATH)
        else:
            loaded_feature_names = self.train_stats.get("features")
            logger.warning(
                "feature_names.pkl is missing; falling back to train_stats.json features for extraction"
            )

        if not isinstance(loaded_feature_names, (list, tuple)) or not loaded_feature_names:
            raise ModelArtifactsError(
                "No usable feature names found in feature_names.pkl or train_stats.json"
            )
        self.feature_names = [str(name) for name in loaded_feature_names]

        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = None
            logger.warning("best_model.pkl is missing; prediction endpoint will remain unavailable")

        logger.info(
            "Loaded available artifacts from %s with %d required features",
            MODEL_DIR,
            len(self.feature_names),
        )

    @property
    def is_ready(self) -> bool:
        return self.model is not None and bool(self.feature_names)


_registry = ModelRegistry()


def load_artifacts() -> ModelRegistry:
    """Load model artifacts once and return the shared registry."""

    if not _registry.is_ready:
        _registry.load()
    return _registry


def get_required_features() -> list[str]:
    """Return the canonical ordered feature list."""

    registry = load_artifacts()
    return list(registry.feature_names)


def get_training_stats() -> dict[str, Any]:
    """Return training statistics used by stage 2 interpretations."""

    registry = load_artifacts()
    return dict(registry.train_stats)


def predict_price(features_dict: dict[str, float | int]) -> float:
    """Predict a sale price from a complete feature dictionary."""

    registry = load_artifacts()
    if registry.model is None:
        raise ModelArtifactsError("Missing required model artifact: best_model.pkl")
    missing = [name for name in registry.feature_names if name not in features_dict]
    if missing:
        raise ValueError(f"Missing feature values for prediction: {', '.join(missing)}")

    ordered_features = {
        feature_name: features_dict[feature_name]
        for feature_name in registry.feature_names
    }
    feature_frame = pd.DataFrame([ordered_features], columns=registry.feature_names)

    prediction = registry.model.predict(feature_frame)
    if len(prediction) != 1:
        raise RuntimeError("Model returned an unexpected number of predictions")

    return float(prediction[0])
