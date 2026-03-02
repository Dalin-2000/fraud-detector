import os
import joblib
from pathlib import Path

from app.schemas.transaction import TransactionInput, PredictionResult
from app.utils.preprocessing import preprocess

MODEL_PATH = Path(__file__).resolve().parents[2] / "ml_models" / "fraud_model.pkl"


class FraudService:
    """Loads the trained model bundle and exposes a predict method."""

    def __init__(self):
        model_path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
        self.model = None
        self.threshold = 0.5  # default fallback
        if model_path.exists():
            bundle = joblib.load(model_path)
            self.model     = bundle.get("model")
            self.threshold = bundle.get("threshold", 0.5)

    def predict(self, transaction: TransactionInput) -> PredictionResult:
        """Run inference on a single transaction."""
        if self.model is None:
            probability = 0.9 if transaction.amount_ngn >= 100000 else 0.1
            fallback = True
        else:
            features = preprocess(transaction)

            if hasattr(self.model, "predict_proba"):
                probability = float(self.model.predict_proba(features)[0][1])
            else:
                probability = float(self.model.predict(features)[0])
            fallback = False

        is_fraud = probability >= self.threshold
        message = (
            "FRAUD DETECTED: This transaction is likely fraudulent."
            if is_fraud
            else "LEGITIMATE: This transaction appears to be genuine."
        )

        if fallback:
            message = f"{message} (Fallback mode: no trained model loaded.)"

        return PredictionResult(
            is_fraud=is_fraud,
            fraud_probability=round(probability, 4),
            message=message,
        )
