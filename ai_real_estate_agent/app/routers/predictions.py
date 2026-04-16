"""Price prediction endpoints."""
from fastapi import APIRouter, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import get_model_loader
from app.llm_client import get_llm_client

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predict property price based on features.

    Args:
        request: PredictionRequest with property features

    Returns:
        PredictionResponse with predicted price
    """
    try:
        model_loader = get_model_loader()
        
        if model_loader.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Convert request to dict
        features = request.property_features.dict()
        
        # Make prediction
        predicted_price = model_loader.predict(features)
        
        if predicted_price is None:
            raise HTTPException(status_code=400, detail="Prediction failed")

        explanation = None
        if request.include_explanation:
            llm = get_llm_client()
            context = f"Property features: {features}\nPredicted price: ${predicted_price:,.2f}"
            explanation = llm.query(
                "Provide a brief explanation of this property valuation.",
                context
            )

        return PredictionResponse(
            predicted_price=predicted_price,
            explanation=explanation
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
