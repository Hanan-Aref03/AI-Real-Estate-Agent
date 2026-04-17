"""Local training script for the AI Real Estate Agent model artifacts.

This script trains a scikit-learn pipeline using a compact Ames feature set that
matches the FastAPI backend contract. It writes the following artifacts:

- models/best_model.pkl
- models/feature_names.pkl
- models/train_stats.json
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "training" / "data"
MODELS_DIR = BASE_DIR / "models"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET_COLUMN = "SalePrice"
RANDOM_STATE = 42

# Backend feature names mapped to the raw Kaggle Ames columns.
FEATURE_MAP: dict[str, str] = {
    "lot_area": "LotArea",
    "year_built": "YearBuilt",
    "year_remod_add": "YearRemodAdd",
    "mas_vnr_area": "MasVnrArea",
    "bsmt_unf_sf": "BsmtUnfSF",
    "total_bsmt_sf": "TotalBsmtSF",
    "first_flr_sf": "1stFlrSF",
    "garage_area": "GarageArea",
    "living_area": "GrLivArea",
}


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Kaggle train and test CSV files from the local data folder."""

    missing = [path.name for path in (TRAIN_PATH, TEST_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing training dataset files in training/data: " + ", ".join(missing)
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and rename the core feature set expected by the API."""

    missing_columns = [raw_name for raw_name in FEATURE_MAP.values() if raw_name not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Input dataset is missing required Ames columns: " + ", ".join(missing_columns)
        )

    selected = frame[list(FEATURE_MAP.values())].copy()
    selected.columns = list(FEATURE_MAP.keys())
    return selected


def train_pipeline(train_df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    """Train the regression pipeline and return model plus evaluation metrics."""

    X = prepare_features(train_df)
    y = train_df[TARGET_COLUMN].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_valid)
    metrics = {
        "validation_rmse": float(np.sqrt(mean_squared_error(y_valid, predictions))),
        "validation_mae": float(mean_absolute_error(y_valid, predictions)),
        "validation_r2": float(r2_score(y_valid, predictions)),
    }
    return pipeline, metrics


def build_training_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pipeline: Pipeline,
    metrics: dict[str, float],
) -> dict[str, object]:
    """Create the stats payload consumed by stage 2 interpretation."""

    feature_names = list(FEATURE_MAP.keys())
    regressor = pipeline.named_steps["regressor"]
    feature_importance = {
        feature_name: float(importance)
        for feature_name, importance in sorted(
            zip(feature_names, regressor.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    prepared_features = prepare_features(train_df)
    feature_statistics = {
        feature_name: {
            "median": float(prepared_features[feature_name].median()),
            "min": float(prepared_features[feature_name].min()),
            "max": float(prepared_features[feature_name].max()),
        }
        for feature_name in feature_names
    }

    return {
        "features": feature_names,
        "training_row_count": int(len(train_df)),
        "test_row_count": int(len(test_df)),
        "median_sale_price": float(train_df[TARGET_COLUMN].median()),
        "min_sale_price": float(train_df[TARGET_COLUMN].min()),
        "max_sale_price": float(train_df[TARGET_COLUMN].max()),
        "feature_importance": feature_importance,
        "feature_statistics": feature_statistics,
        "metrics": metrics,
    }


def save_artifacts(pipeline: Pipeline, stats: dict[str, object]) -> None:
    """Persist trained artifacts for the API."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODELS_DIR / "best_model.pkl")
    joblib.dump(list(FEATURE_MAP.keys()), MODELS_DIR / "feature_names.pkl")

    with (MODELS_DIR / "train_stats.json").open("w", encoding="utf-8") as file_handle:
        json.dump(stats, file_handle, indent=2)


def main() -> None:
    """Train the model and write the API artifacts."""

    train_df, test_df = load_datasets()
    pipeline, metrics = train_pipeline(train_df)
    stats = build_training_stats(train_df, test_df, pipeline, metrics)
    save_artifacts(pipeline, stats)

    print("Training complete.")
    print(f"Saved model to: {MODELS_DIR / 'best_model.pkl'}")
    print(f"Saved feature names to: {MODELS_DIR / 'feature_names.pkl'}")
    print(f"Saved training stats to: {MODELS_DIR / 'train_stats.json'}")
    print("Validation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  - {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
