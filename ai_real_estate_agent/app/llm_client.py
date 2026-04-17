"""LLM client utilities for Gemini-powered real estate assistant features."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from app.schemas import ExtractedFeatures

load_dotenv()

try:
    import google.generativeai as genai

    GEMINI_SDK_AVAILABLE = True
except ModuleNotFoundError:
    genai = None  # type: ignore[assignment]
    GEMINI_SDK_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_SDK_AVAILABLE = True
except ModuleNotFoundError:
    OpenAI = Any  # type: ignore[assignment]
    OPENAI_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMOutputError(Exception):
    """Raised when LLM output cannot be parsed into the required JSON structure."""


@dataclass(frozen=True)
class LLMSettings:
    """Runtime LLM configuration."""

    provider: str
    api_key: str | None
    stage1_model: str
    stage2_model: str


@dataclass(frozen=True)
class AssistantResult:
    """Normalized LLM answer with provenance."""

    text: str
    source: str
    warnings: list[str]


class RealEstateLLMClient:
    """Small compatibility wrapper used by existing query routes."""

    def __init__(self) -> None:
        self.settings = _get_settings()

    @property
    def source(self) -> str:
        settings = self.settings
        if settings.provider == "gemini":
            return f"gemini:{settings.stage2_model}"
        if settings.provider == "openai":
            return f"openai:{settings.stage2_model}"
        return "mock"

    def query(self, question: str, context: str | None = None) -> str:
        return query_real_estate_assistant(question, context).text


def _get_settings() -> LLMSettings:
    """Read environment variables at call time so .env changes are picked up."""

    provider_preference = os.getenv("LLM_PROVIDER", "").strip().lower()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if provider_preference == "mock":
        return LLMSettings("mock", None, "mock", "mock")

    if provider_preference == "gemini" and gemini_api_key:
        default_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        return LLMSettings(
            provider="gemini",
            api_key=gemini_api_key,
            stage1_model=os.getenv("GEMINI_STAGE1_MODEL", default_model),
            stage2_model=os.getenv("GEMINI_STAGE2_MODEL", default_model),
        )

    if provider_preference == "openai" and openai_api_key:
        return LLMSettings(
            provider="openai",
            api_key=openai_api_key,
            stage1_model=os.getenv("OPENAI_STAGE1_MODEL", "gpt-4.1-mini"),
            stage2_model=os.getenv("OPENAI_STAGE2_MODEL", "gpt-4.1-mini"),
        )

    if gemini_api_key:
        default_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        return LLMSettings(
            provider="gemini",
            api_key=gemini_api_key,
            stage1_model=os.getenv("GEMINI_STAGE1_MODEL", default_model),
            stage2_model=os.getenv("GEMINI_STAGE2_MODEL", default_model),
        )

    if openai_api_key:
        return LLMSettings(
            provider="openai",
            api_key=openai_api_key,
            stage1_model=os.getenv("OPENAI_STAGE1_MODEL", "gpt-4.1-mini"),
            stage2_model=os.getenv("OPENAI_STAGE2_MODEL", "gpt-4.1-mini"),
        )

    return LLMSettings("mock", None, "mock", "mock")


def _get_openai_client(settings: LLMSettings) -> OpenAI | None:
    if settings.provider != "openai" or not settings.api_key:
        return None
    if not OPENAI_SDK_AVAILABLE:
        logger.warning("OpenAI SDK not installed; using fallback behavior")
        return None
    return OpenAI(api_key=settings.api_key)


def _get_gemini_model(model_name: str, api_key: str | None) -> Any | None:
    if not api_key:
        return None
    if not GEMINI_SDK_AVAILABLE or genai is None:
        logger.warning("Gemini SDK not installed; using fallback behavior")
        return None
    genai.configure(api_key=api_key, transport="rest")
    return genai.GenerativeModel(model_name)


def _generate_text(prompt: str, *, stage: str) -> tuple[str | None, str]:
    """Generate text from the configured provider and report its source."""

    settings = _get_settings()
    model_name = settings.stage1_model if stage == "stage1" else settings.stage2_model

    try:
        if settings.provider == "gemini":
            model = _get_gemini_model(model_name, settings.api_key)
            if model is not None:
                response = model.generate_content(prompt, request_options={"timeout": 20})
                text = getattr(response, "text", "") or ""
                return text.strip(), f"gemini:{model_name}"

        if settings.provider == "openai":
            client = _get_openai_client(settings)
            if client is not None:
                response = client.responses.create(
                    model=model_name,
                    input=prompt,
                    temperature=0 if stage == "stage1" else 0.2,
                    timeout=20,
                )
                return response.output_text.strip(), f"openai:{model_name}"
    except Exception:
        logger.exception("Remote LLM call failed during %s; using fallback behavior", stage)
        return None, "mock"

    return None, "mock"


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Parse JSON from free-form text with fenced and brace fallbacks."""

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
    """Construct a Gemini-friendly extraction prompt for real estate valuation."""

    joined_features = ", ".join(required_features)
    shared_rules = (
        "You are an AI real estate assistant helping a home pricing system. "
        "Extract only numeric home attributes that are explicitly stated or directly inferable. "
        "Never guess. Never invent defaults. "
        'Return valid JSON only using this exact format: {"features": {"feature_name": number_or_null}}. '
        f"Required features are: {joined_features}."
    )

    if variant == "A":
        return (
            "Task: convert this property description into structured pricing features.\n"
            f"{shared_rules}\n"
            "If a feature is absent, return null.\n"
            f"Property description: {query}"
        )

    return (
        "Task: read this home description and prepare structured input for a real estate price model.\n"
        f"{shared_rules}\n"
        "Use null for any uncertain field and keep names exactly unchanged.\n"
        f"Listing notes: {query}"
    )


def _build_assistant_prompt(question: str, context: str | None) -> str:
    """Construct a Gemini-friendly assistant prompt for real estate conversations."""

    context_block = f"\nRelevant context:\n{context}\n" if context else ""
    return (
        "You are a professional real estate AI assistant. "
        "Help buyers, sellers, and agents understand pricing, listings, missing property details, "
        "and valuation drivers in clear, practical language.\n"
        "Guidelines:\n"
        "- Be concise and useful.\n"
        "- Stay focused on real estate.\n"
        "- If the user gives an incomplete property description, explain what details would improve a valuation.\n"
        "- Do not claim certainty when important details are missing.\n"
        f"{context_block}"
        f"\nUser question:\n{question}"
    )


def _mock_extract_features(query: str, required_features: list[str]) -> ExtractedFeatures:
    """Deterministic fallback when no remote provider is available."""

    text = query.lower()
    patterns: dict[str, list[str]] = {
        "lot_area": [
            r"lot (?:area|size)\D+(\d+(?:,\d+)*)",
            r"(\d+(?:,\d+)*)\s*(?:sq\s*ft|square feet).*(?:lot)",
        ],
        "year_built": [r"(?:built|constructed) (?:in )?(\d{4})"],
        "year_remod_add": [
            r"(?:remodel(?:ed)?|renovat(?:ed|ion)|updated) (?:in )?(\d{4})",
            r"remodeled in (\d{4})",
        ],
        "mas_vnr_area": [r"masonry veneer (?:area )?\D+(\d+(?:,\d+)*)"],
        "bsmt_unf_sf": [
            r"(?:unfinished basement|unf basement)\D+(\d+(?:,\d+)*)",
            r"(\d+(?:,\d+)*)\s*(?:sq\s*ft|square feet)\s*(?:unfinished basement)",
        ],
        "total_bsmt_sf": [
            r"(?:total basement|basement)\D+(\d+(?:,\d+)*)\s*(?:sq\s*ft|square feet)?",
            r"(\d+(?:,\d+)*)\s*(?:sq\s*ft|square feet)\s*(?:total basement)",
        ],
        "first_flr_sf": [
            r"(?:first floor|1st floor)\D+(\d+(?:,\d+)*)",
            r"(\d+(?:,\d+)*)\s*(?:sq\s*ft|square feet)\s*(?:on the first floor|first floor)",
        ],
        "garage_area": [r"garage (?:area )?\D+(\d+(?:,\d+)*)"],
        "living_area": [
            r"(?:living area|above ground living area|gr liv area)\D+(\d+(?:,\d+)*)",
            r"(\d+(?:,\d+)*)\s*(?:sq\s*ft|square feet)\s*(?:of living area|living area)",
        ],
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
    comparison_notes = "Mock extraction used because Gemini was unavailable."

    return ExtractedFeatures(
        features=features,
        missing_fields=missing_fields,
        extraction_complete=not missing_fields,
        source="mock",
        prompt_variant="A",
        comparison_notes=comparison_notes,
        raw_llm_output=None,
    )


def _mock_query_real_estate_assistant(question: str, context: str | None) -> AssistantResult:
    """Fallback assistant response when Gemini is unavailable."""

    answer = (
        "I can help with property pricing, listing analysis, and missing valuation details. "
        "For a stronger estimate, include the lot area, year built, living area, first-floor area, "
        "garage area, basement size, remodel year, and masonry veneer area."
    )
    if context:
        answer += f" Context received: {context}"
    return AssistantResult(text=answer, source="mock", warnings=["Gemini was unavailable, so a local fallback response was used."])


def get_llm_client() -> RealEstateLLMClient:
    """Compatibility factory used by legacy query routes."""

    return RealEstateLLMClient()


def query_real_estate_assistant(question: str, context: str | None = None) -> AssistantResult:
    """Answer a general real estate question using Gemini, with fallback behavior."""

    prompt = _build_assistant_prompt(question, context)
    output, source = _generate_text(prompt, stage="stage2")
    if output:
        warnings = []
        if source.startswith("gemini:"):
            warnings.append("Gemini answered this request.")
        return AssistantResult(text=output, source=source, warnings=warnings)
    return _mock_query_real_estate_assistant(question, context)


def extract_features(query: str, required_features: list[str]) -> ExtractedFeatures:
    """Stage 1: extract structured features from natural language."""

    parsed_variants: dict[str, ExtractedFeatures] = {}

    for variant in ("A", "B"):
        prompt = _build_stage1_prompt(query, required_features, variant)
        raw_text, source = _generate_text(prompt, stage="stage1")
        if raw_text is None:
            logger.warning("Gemini unavailable; using mock extraction fallback")
            return _mock_extract_features(query, required_features)

        payload = _extract_json_payload(raw_text)
        parsed_variants[variant] = _normalize_extraction_payload(
            payload,
            required_features,
            source=source,
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
    """Fallback interpretation that still uses training statistics."""

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
        f'This home is estimated at ${price:,.0f}, which is {comparison}. '
        f"The training data spans about ${min_price:,.0f} to ${max_price:,.0f}, and the model gives the most "
        f"weight to {important_features}. The estimate is based on: {features}."
    )


def interpret_prediction(
    query: str,
    features: dict[str, float | int],
    price: float,
    stats: dict[str, Any],
) -> str:
    """Stage 2: generate a contextual interpretation of the predicted price."""

    prompt = (
        "You are a real estate AI assistant explaining a home price estimate.\n"
        "Write one short paragraph for a buyer, seller, or agent.\n"
        "Be practical, specific, and easy to understand.\n"
        "Mention whether the estimated price is below, near, or above the training-set median when possible.\n"
        "If details are limited, avoid overclaiming certainty.\n\n"
        f"User property description: {query}\n"
        f"Structured features: {json.dumps(features, sort_keys=True)}\n"
        f"Predicted price: ${price:,.2f}\n"
        f"Training statistics: {json.dumps(stats, sort_keys=True)}"
    )

    try:
        output, _ = _generate_text(prompt, stage="stage2")
        if not output:
            raise LLMOutputError("Stage 2 returned an empty explanation")
        return output
    except Exception:
        logger.exception("Stage 2 interpretation failed; falling back to local response")
        return _mock_interpret_prediction(query, features, price, stats)
