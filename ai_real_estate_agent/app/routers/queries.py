"""LLM query endpoints."""
from fastapi import APIRouter, HTTPException
from app.schemas import QueryRequest, QueryResponse
from app.llm_client import get_llm_client

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
        llm = get_llm_client()
        answer = llm.query(request.question, request.context)
        
        return QueryResponse(
            answer=answer,
            source="openai"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
