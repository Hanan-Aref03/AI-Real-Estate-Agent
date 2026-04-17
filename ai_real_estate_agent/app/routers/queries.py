"""LLM query endpoints."""

from fastapi import APIRouter, HTTPException

from app.llm_client import query_real_estate_assistant
from app.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["queries"])


@router.post("", response_model=QueryResponse)
async def llm_query(request: QueryRequest):
    """
    Query the LLM with optional context.

    Args:
        request: QueryRequest with question and optional context

    Returns:
        QueryResponse with answer
    """
    try:
        result = query_real_estate_assistant(request.question, request.context)
        return QueryResponse(
            answer=result.text,
            source=result.source,
            warnings=result.warnings,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
