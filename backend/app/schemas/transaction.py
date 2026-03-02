from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Input schema matching the 18-feature training pipeline."""

    # Core numerical features
    amount_ngn: float = Field(..., gt=0, description="Transaction amount in Nigerian Naira")
    time_since_last_transaction: float = Field(..., ge=0, description="Minutes since last transaction by this user")
    spending_deviation_score: float = Field(..., description="How different this amount is from the user's typical spending")
    velocity_score: int = Field(..., ge=0, description="Quick successive transaction score")
    geo_anomaly_score: float = Field(..., ge=0, description="Geographic unusualness score")

    # Categorical features
    transaction_type: str = Field(..., description="Type of transaction e.g. Transfer, Payment, Withdrawal")
    merchant_category: str = Field(..., description="Merchant category e.g. Retail, Food, Entertainment")
    location: str = Field(..., description="City where transaction occurred")
    device_used: str = Field(..., description="Device/channel e.g. Mobile, Web, POS")
    payment_channel: str = Field(..., description="Payment route e.g. Card, Bank Transfer, USSD")
    sender_persona: str = Field(..., description="Sender behavioural profile e.g. Regular, High-Value")

    # Boolean security flags
    bvn_linked: bool = Field(..., description="Whether BVN is linked to the account")
    new_device_transaction: bool = Field(..., description="True if this device is new for this user")

    # Optional timestamp — defaults to now if not provided
    timestamp: Optional[datetime] = Field(None, description="Transaction datetime (defaults to now)")

    class Config:
        json_schema_extra = {
            "example": {
                "amount_ngn": 150000.0,
                "time_since_last_transaction": 120.5,
                "spending_deviation_score": 2.3,
                "velocity_score": 3,
                "geo_anomaly_score": 0.8,
                "transaction_type": "Transfer",
                "merchant_category": "Retail",
                "location": "Lagos",
                "device_used": "Mobile",
                "payment_channel": "Bank Transfer",
                "sender_persona": "Regular",
                "bvn_linked": True,
                "new_device_transaction": False,
                "timestamp": "2025-03-01T14:30:00"
            }
        }


class PredictionResult(BaseModel):
    """Response schema for a fraud prediction."""

    is_fraud: bool = Field(..., description="True if the transaction is predicted as fraudulent")
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability score that the transaction is fraud"
    )
    message: str = Field(..., description="Human-readable verdict")
