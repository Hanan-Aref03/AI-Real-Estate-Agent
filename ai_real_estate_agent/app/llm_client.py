"""LLM client utilities for feature extraction and prediction interpretation."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from app.schemas import ExtractedFeatures

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ModuleNotFoundError:
    OpenAI = Any  # type: ignore[assignment]
    OPENAI_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_STAGE1_MODEL = os.getenv("OPENAI_STAGE1_MODEL", "gpt-4.1-mini")
DEFAULT_STAGE2_MODEL = os.getenv("OPENAI_STAGE2_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class LLMOutputError(Exception):
    """Raised when LLM output cannot be parsed into the required JSON structure."""


@dataclass(frozen=True)
class LLMSettings:
    """Runtime LLM configuration."""

    api_key: str | None
    stage1_model: str = DEFAULT_STAGE1_MODEL
    stage2_model: str = DEFAULT_STAGE2_MODEL


def _get_settings() -> LLMSettings:
    return LLMSettings(api_key=OPENAI_API_KEY)


def _get_client() -> OpenAI | None:
    settings = _get_settings()
    if not settings.api_key:
        return None
    if not OPENAI_SDK_AVAILABLE:
        logger.warning("openai package is not installed; using mock LLM fallback")
        return None
    return OpenAI(api_key=settings.api_key)


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Parse JSON from free-form text with a fenced-code and brace fallback."""

    cleaned = text.strip()
    if not cleaned:
        raise LLMOutputError("LLM returned an empty response")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"Invalid JSON inside code fence: {exc}") from exc

    brace_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"Invalid JSON in regex fallback: {exc}") from exc

    raise LLMOutputError("No JSON object found in LLM output")


def _coerce_feature_value(value: Any) -> float | int | None:
    """Convert parsed values into numbers or None without inventing defaults."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"unknown", "null", "none", "missing", "n/a"}:
            return None
        try:
            numeric = float(stripped.replace(",", ""))
        except ValueError as exc:
            raise LLMOutputError(f"Feature value is not numeric: {value!r}") from exc
        return int(numeric) if numeric.is_integer() else numeric
    raise LLMOutputError(f"Unsupported feature value type: {type(value).__name__}")


def _normalize_extraction_payload(
    payload: dict[str, Any],
    required_features: list[str],
    *,
    source: str,
    prompt_variant: str,
    comparison_notes: str | None,
    raw_llm_output: str | None,
) -> ExtractedFeatures:
    """Validate and normalize stage 1 output from either real or mock LLM paths."""

    feature_block = payload.get("features", payload)
    if not isinstance(feature_block, dict):
        raise LLMOutputError("Stage 1 payload must include a JSON object of features")

    normalized_features = {name: None for name in required_features}
    for feature_name in required_features:
        normalized_features[feature_name] = _coerce_feature_value(feature_block.get(feature_name))

    missing_fields = [name for name, value in normalized_features.items() if value is None]
    extraction_complete = not missing_fields

    return ExtractedFeatures(
        features=normalized_features,
        missing_fields=missing_fields,
        extraction_complete=extraction_complete,
        source=source,
        prompt_variant=prompt_variant,
        comparison_notes=comparison_notes,
        raw_llm_output=raw_llm_output,
    )


def _build_stage1_prompt(query: str, required_features: list[str], variant: str) -> str:
    """Construct a strict extraction prompt. Variants differ in reasoning style."""

    joined_features = ", ".join(required_features)
    shared_rules = (
        "Extract only explicitly stated or directly inferable numeric property details from the user query. "
        "Never guess or backfill missing values. "
        "Return valid JSON only with this shape: "
        '{"features": {"feature_name": number_or_null, "...": number_or_null}}. '
        f"Required features: {joined_features}."
    )

    if variant == "A":
        return (
            "You are a careful real-estate information extraction engine.\n"
            f"{shared_rules}\n"
            "Use null when a feature is missing. Keep feature names exactly as requested.\n"
            f"User query: {query}"
        )

    return (
        "You convert property descriptions into ML-ready Ames housing features.\n"
        f"{shared_rules}\n"
        "Before answering, internally verify each value is numeric and supported by the query text. "
        "If uncertain, use null.\n"
        f"Property description: {query}"
    )


def _mock_extract_features(query: str, required_features: list[str]) -> ExtractedFeatures:
    """
    Deterministic fallback for local/dev use without an OpenAI key.

    This intentionally stays conservative and only extracts simple numeric mentions so
    the API never pretends it knows missing values.
    """

    text = query.lower()
    patterns: dict[str, list[str]] = {
        "lot_area": [r"lot (?:area|size)\D+(\d+(?:,\d+)*)", r"(\d+(?:,\d+)*)\s*(?:sq\s*ft|square feet).*(?:lot)"],
        "year_built": [r"(?:built|constructed) (?:in )?(\d{4})"],
        "year_remod_add": [r"(?:remodel(?:ed)?|renovat(?:ed|ion)) (?:in )?(\d{4})"],
        "mas_vnr_area": [r"masonry veneer (?:area )?\D+(\d+(?:,\d+)*)"],
        "bsmt_unf_sf": [r"(?:unfinished basement|unf basement)\D+(\d+(?:,\d+)*)"],
        "total_bsmt_sf": [r"(?:total basement|basement)\D+(\d+(?:,\d+)*)\s*(?:sq\s*ft|square feet)?"],
        "first_flr_sf": [r"(?:first floor|1st floor)\D+(\d+(?:,\d+)*)"],
        "garage_area": [r"garage (?:area )?\D+(\d+(?:,\d+)*)"],
        "living_area": [r"(?:living area|above ground living area|gr liv area)\D+(\d+(?:,\d+)*)"],
    }

    features: dict[str, float | int | None] = {}
    for name in required_features:
        extracted_value: float | int | None = None
        for pattern in patterns.get(name, []):
            match = re.search(pattern, text)
            if match:
                numeric = float(match.group(1).replace(",", ""))
                extracted_value = int(numeric) if numeric.is_integer() else numeric
                break
        features[name] = extracted_value

    missing_fields = [name for name, value in features.items() if value is None]
    return ExtractedFeatures(
        features=features,
        missing_fields=missing_fields,
        extraction_complete=not missing_fields,
        source="mock",
        prompt_variant="A",
        comparison_notes="Mock extraction used because OPENAI_API_KEY is not configured.",
        raw_llm_output=None,
    )


def extract_features(query: str, required_features: list[str]) -> ExtractedFeatures:
    """Stage 1: extract structured features from natural language."""

    client = _get_client()
    if client is None:
        logger.warning("OPENAI_API_KEY not configured; using mock extraction fallback")
        return _mock_extract_features(query, required_features)

    settings = _get_settings()
    raw_outputs: dict[str, str] = {}
    parsed_variants: dict[str, ExtractedFeatures] = {}

    for variant in ("A", "B"):
        prompt = _build_stage1_prompt(query, required_features, variant)
        response = client.responses.create(
            model=settings.stage1_model,
            input=prompt,
            temperature=0,
        )
        raw_text = response.output_text.strip()
        raw_outputs[variant] = raw_text

        payload = _extract_json_payload(raw_text)
        parsed_variants[variant] = _normalize_extraction_payload(
            payload,
            required_features,
            source=f"openai:{settings.stage1_model}",
            prompt_variant=variant,
            comparison_notes=None,
            raw_llm_output=raw_text,
        )

    primary = parsed_variants["A"]
    secondary = parsed_variants["B"]
    differing_fields = [
        name
        for name in required_features
        if primary.features.get(name) != secondary.features.get(name)
    ]

    comparison_notes = (
        "Stage 1 variants matched exactly."
        if not differing_fields
        else f"Variant comparison differences on: {', '.join(differing_fields)}"
    )
    logger.info("Stage 1 prompt comparison complete: %s", comparison_notes)

    return primary.model_copy(update={"comparison_notes": comparison_notes})


def _mock_interpret_prediction(
    query: str,
    features: dict[str, float | int],
    price: float,
    stats: dict[str, Any],
) -> str:
    """Local interpretation fallback that still uses training statistics."""

    median_price = float(stats.get("median_sale_price", 0) or 0)
    min_price = float(stats.get("min_sale_price", 0) or 0)
    max_price = float(stats.get("max_sale_price", 0) or 0)
    feature_importance = stats.get("feature_importance", {})

    comparison = "near the training-set median"
    if median_price:
        if price > median_price * 1.15:
            comparison = "above the training-set median"
        elif price < median_price * 0.85:
            comparison = "below the training-set median"

    important_features = ", ".join(list(feature_importance.keys())[:3]) or "the available property features"
    return (
        f'For the request "{query}", the estimated price is ${price:,.0f}, which is {comparison}. '
        f"The training data spans roughly ${min_price:,.0f} to ${max_price:,.0f}, and the model places the most "
        f"weight on {important_features}. The interpretation is based on the extracted values: {features}."
    )


def interpret_prediction(
    query: str,
    features: dict[str, float | int],
    price: float,
    stats: dict[str, Any],
) -> str:
    """Stage 2: generate a contextual interpretation of the predicted price."""

    client = _get_client()
    if client is None:
        logger.warning("OPENAI_API_KEY not configured; using mock interpretation fallback")
        return _mock_interpret_prediction(query, features, price, stats)

    settings = _get_settings()
    prompt = (
        "You are explaining a house price prediction to an end user.\n"
        "Write 1 short paragraph in plain English. Be specific, avoid hype, and reference the training set context.\n"
        "Mention whether the predicted price is below, near, or above the training median when possible.\n"
        "Do not mention hidden chain-of-thought or internal uncertainty calculations.\n\n"
        f"Original user query: {query}\n"
        f"Structured features: {json.dumps(features, sort_keys=True)}\n"
        f"Predicted price: ${price:,.2f}\n"
        f"Training statistics: {json.dumps(stats, sort_keys=True)}"
    )

    try:
        response = client.responses.create(
            model=settings.stage2_model,
            input=prompt,
            temperature=0.2,
        )
        output = response.output_text.strip()
        if not output:
            raise LLMOutputError("Stage 2 returned an empty explanation")
        return output
    except Exception as exc:
        logger.exception("Stage 2 interpretation failed; falling back to mock response")
        fallback = _mock_interpret_prediction(query, features, price, stats)
        return f"{fallback} Fallback reason: {type(exc).__name__}."
