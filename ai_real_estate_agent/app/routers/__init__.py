"""API route modules."""
from app.routers.predictions import router as predictions_router
from app.routers.queries import router as queries_router

__all__ = ["predictions_router", "queries_router"]
