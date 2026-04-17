"""Pydantic schemas for the AI Real Estate Agent backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExtractedFeatures(BaseModel):
    """Normalized representation of stage 1 feature extraction output."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, float | int | None] = Field(
        default_factory=dict,
        description="Extracted Ames housing features. Missing values must stay null/None.",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Required features that are still missing after extraction and merge.",
    )
    extraction_complete: bool = Field(
        ...,
        description="True only when every required feature is available.",
    )
    source: str = Field(
        ...,
        description="The extraction mode used, for example openai:gpt-4.1-mini or mock.",
    )
    prompt_variant: str = Field(
        ...,
        description="Stage 1 prompt variant used for the primary extraction result.",
    )
    comparison_notes: str | None = Field(
        default=None,
        description="Optional comparison summary between Stage 1 prompt variants.",
    )
    raw_llm_output: str | None = Field(
        default=None,
        description="Raw LLM output retained for diagnostics.",
    )

    @field_validator("missing_fields")
    @classmethod
    def _sort_missing_fields(cls, value: list[str]) -> list[str]:
        return sorted(dict.fromkeys(value))


class PredictionRequest(BaseModel):
    """Incoming request payload for feature extraction and prediction."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        description="Natural language description of the property or user request.",
    )
    user_filled_features: dict[str, float | int | None] | None = Field(
        default=None,
        description="Optional user-supplied feature overrides used to complete missing values.",
    )

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class PredictionResponse(BaseModel):
    """Validated API response for prediction requests."""

    model_config = ConfigDict(extra="forbid")

    predicted_price: float = Field(..., ge=0)
    currency: str = Field(default="USD")
    features_used: dict[str, float | int] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    extraction: ExtractedFeatures
    interpretation: str = Field(..., min_length=1)
    stats_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    user_benefit_summary: str | None = None


class QueryRequest(BaseModel):
    """Request payload for the real estate AI assistant."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1)
    context: str | None = None

    @field_validator("question")
    @classmethod
    def _question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


class QueryResponse(BaseModel):
    """Response payload for the real estate AI assistant."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1)
    source: str
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Structured error payload for predictable client handling."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    missing_fields: list[str] = Field(default_factory=list)
    extraction: ExtractedFeatures | None = None
    stats_summary: dict[str, Any] = Field(default_factory=dict)
    user_message: str | None = None
