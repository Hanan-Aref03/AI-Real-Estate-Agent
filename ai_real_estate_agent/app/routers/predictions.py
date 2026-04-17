"""Price prediction endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.llm_client import LLMOutputError, extract_features, interpret_prediction
from app.model_loader import (
    ModelArtifactsError,
    get_required_features,
    get_training_stats,
    predict_price,
)
from app.schemas import ErrorResponse, ExtractedFeatures, PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["predictions"])


def _merge_features(
    extracted: ExtractedFeatures,
    user_filled_features: dict[str, float | int | None] | None,
    required_features: list[str],
) -> ExtractedFeatures:
    merged = dict(extracted.features)
    if user_filled_features:
        for key, value in user_filled_features.items():
            if key in merged and value is not None:
                merged[key] = value

    missing_fields = [name for name in required_features if merged.get(name) is None]
    return extracted.model_copy(
        update={
            "features": merged,
            "missing_fields": missing_fields,
            "extraction_complete": not missing_fields,
        }
    )


def _build_stats_summary(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "median_sale_price": stats.get("median_sale_price"),
        "min_sale_price": stats.get("min_sale_price"),
        "max_sale_price": stats.get("max_sale_price"),
        "feature_importance": stats.get("feature_importance", {}),
        "feature_statistics": stats.get("feature_statistics", {}),
    }


@router.post("", response_model=PredictionResponse)
async def predict_route(request: PredictionRequest):
    """Predict a property price from a natural-language query plus optional filled features."""

    extraction_payload: ExtractedFeatures | None = None
    try:
        required_features = get_required_features()
        stats = get_training_stats()
        extraction_payload = extract_features(request.query, required_features)
        merged_extraction = _merge_features(
            extraction_payload, request.user_filled_features, required_features
        )

        if merged_extraction.missing_fields:
            error_payload = ErrorResponse(
                detail="Missing required features for prediction",
                missing_fields=merged_extraction.missing_fields,
                extraction=merged_extraction,
                stats_summary=_build_stats_summary(stats),
                user_message="Add the missing details, or use the suggested values from the UI.",
            )
            return JSONResponse(status_code=400, content=error_payload.model_dump())

        complete_features = {
            feature_name: merged_extraction.features[feature_name]
            for feature_name in required_features
            if merged_extraction.features[feature_name] is not None
        }
        predicted_price_value = predict_price(complete_features)
        interpretation = interpret_prediction(
            request.query,
            complete_features,
            predicted_price_value,
            stats,
        )

        warnings: list[str] = []
        if merged_extraction.source.startswith("gemini:"):
            warnings.append("Gemini handled the language steps for this response.")
        elif merged_extraction.source == "mock":
            warnings.append("Gemini was unavailable, so a local fallback was used.")

        return PredictionResponse(
            predicted_price=predicted_price_value,
            currency="USD",
            features_used=complete_features,
            missing_fields=[],
            extraction=merged_extraction,
            interpretation=interpretation,
            stats_summary=_build_stats_summary(stats),
            warnings=warnings,
            user_benefit_summary="This estimate helps benchmark the property against the training market.",
        )
    except ModelArtifactsError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except LLMOutputError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
