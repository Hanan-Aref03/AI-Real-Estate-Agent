"""FastAPI application entry point for the AI Real Estate Agent backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.llm_client import (
    LLMOutputError,
    extract_features,
    interpret_prediction,
    query_real_estate_assistant,
)
from app.model_loader import (
    ModelArtifactsError,
    get_required_features,
    get_training_stats,
    load_artifacts,
    predict_price,
)
from app.schemas import (
    ErrorResponse,
    ExtractedFeatures,
    PredictionRequest,
    PredictionResponse,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _merge_features(
    extracted: ExtractedFeatures,
    user_filled_features: dict[str, float | int | None] | None,
    required_features: list[str],
) -> ExtractedFeatures:
    """Merge LLM extraction with user-provided overrides without inventing defaults."""

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
    """Keep the response stats payload small and useful for clients."""

    return {
        "median_sale_price": stats.get("median_sale_price"),
        "min_sale_price": stats.get("min_sale_price"),
        "max_sale_price": stats.get("max_sale_price"),
        "feature_importance": stats.get("feature_importance", {}),
        "feature_statistics": stats.get("feature_statistics", {}),
    }


def _build_user_benefit_summary(price: float, stats: dict[str, Any]) -> str:
    """Explain why the product is useful in plain language."""

    median_price = float(stats.get("median_sale_price", 0) or 0)
    if median_price and price >= median_price:
        positioning = "above the training-set midpoint"
    elif median_price:
        positioning = "below the training-set midpoint"
    else:
        positioning = "within the observed market range"

    return (
        f"This estimate helps a buyer, seller, or agent quickly benchmark a home against the training-set market, "
        f"see whether the property sits {positioning}, and understand which details most influence the final value."
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load model artifacts at startup so runtime requests stay fast."""

    try:
        load_artifacts()
        logger.info("Model artifacts loaded successfully during startup")
    except ModelArtifactsError:
        logger.exception("Startup completed without model readiness")
    yield


app = FastAPI(
    title="AI Real Estate Agent API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple readiness endpoint."""

    try:
        registry = load_artifacts()
        required_feature_count = len(registry.feature_names)
    except ModelArtifactsError:
        return {"status": "degraded"}
    state = "ready" if registry.model is not None else "degraded"
    return {"status": f"{state}:{required_feature_count}_features"}


@app.post("/extract", response_model=ExtractedFeatures)
async def extract_endpoint(request: PredictionRequest) -> ExtractedFeatures:
    """Helper endpoint to expose stage 1 extraction without predicting."""

    try:
        required_features = get_required_features()
        extraction = extract_features(request.query, required_features)
        return _merge_features(extraction, request.user_filled_features, required_features)
    except ModelArtifactsError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except LLMOutputError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected extraction failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {type(exc).__name__}",
        ) from exc


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """General real estate AI assistant endpoint backed by Gemini."""

    try:
        result = query_real_estate_assistant(request.question, request.context)
        return QueryResponse(
            answer=result.text,
            source=result.source,
            warnings=result.warnings,
        )
    except Exception as exc:
        logger.exception("Unexpected assistant query failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assistant query failed: {type(exc).__name__}",
        ) from exc


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """Run the two-stage prompt chain plus ML inference."""

    extraction_payload: ExtractedFeatures | None = None
    try:
        required_features = get_required_features()
        stats = get_training_stats()
        extraction_payload = extract_features(request.query, required_features)
        merged_extraction = _merge_features(
            extraction_payload, request.user_filled_features, required_features
        )

        if merged_extraction.missing_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required features for prediction",
                headers={"X-Missing-Fields": ",".join(merged_extraction.missing_fields)},
            )

        complete_features = {
            feature_name: merged_extraction.features[feature_name]
            for feature_name in required_features
            if merged_extraction.features[feature_name] is not None
        }
        predicted_price = predict_price(complete_features)
        interpretation = interpret_prediction(
            request.query,
            complete_features,
            predicted_price,
            stats,
        )

        warnings: list[str] = []
        if merged_extraction.source == "mock":
            warnings.append(
                "Mock LLM fallback was used because no remote LLM provider was available."
            )
        if merged_extraction.source.startswith("gemini:"):
            warnings.append("Gemini handled the language steps for this response.")

        return PredictionResponse(
            predicted_price=predicted_price,
            currency="USD",
            features_used=complete_features,
            missing_fields=[],
            extraction=merged_extraction,
            interpretation=interpretation,
            stats_summary=_build_stats_summary(stats),
            warnings=warnings,
            user_benefit_summary=_build_user_benefit_summary(predicted_price, stats),
        )
    except HTTPException as exc:
        if exc.status_code == status.HTTP_400_BAD_REQUEST:
            error_payload = ErrorResponse(
                detail="Missing required features for prediction",
                missing_fields=extraction_payload.missing_fields if extraction_payload else [],
                extraction=extraction_payload,
                stats_summary=_build_stats_summary(stats) if "stats" in locals() else {},
                user_message=(
                    "Complete the missing property details to unlock the valuation. "
                    "You can type the exact values you know or apply typical values as a starting point."
                ),
            )
            return JSONResponse(status_code=400, content=error_payload.model_dump())
        raise exc
    except ModelArtifactsError as exc:
        error_payload = ErrorResponse(
            detail=str(exc),
            missing_fields=[],
            extraction=None,
            stats_summary={},
            user_message="The pricing model is unavailable right now. Check that the trained model artifacts exist.",
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_payload.model_dump(),
        )
    except LLMOutputError as exc:
        error_payload = ErrorResponse(
            detail=f"Malformed LLM output: {exc}",
            missing_fields=[],
            extraction=extraction_payload,
            stats_summary=_build_stats_summary(stats) if "stats" in locals() else {},
            user_message="The language step returned an invalid response. You can retry or fill features manually.",
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=error_payload.model_dump(),
        )
    except Exception as exc:
        logger.exception("Unexpected prediction failure")
        fallback_message = "Prediction failed due to an internal server error"
        error_payload = ErrorResponse(
            detail=f"{fallback_message}: {type(exc).__name__}",
            missing_fields=[],
            extraction=extraction_payload,
            stats_summary=_build_stats_summary(stats) if "stats" in locals() else {},
            user_message="The request did not complete. Please retry, or use the extraction audit to inspect the inputs.",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_payload.model_dump(),
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
