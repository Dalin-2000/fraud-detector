from fastapi import APIRouter, HTTPException

from app.schemas.transaction import TransactionInput, PredictionResult
from app.services.fraud_service import FraudService

router = APIRouter()
_service = FraudService()


@router.post("/predict", response_model=PredictionResult)
def predict(transaction: TransactionInput):
    """
    Predict whether a financial transaction is fraudulent.

    Returns a fraud probability score and a boolean verdict.
    """
    try:
        result = _service.predict(transaction)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
