from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.routers.predict._service")
def test_predict_fraud(mock_service):
    mock_service.predict.return_value = MagicMock(
        is_fraud=True,
        fraud_probability=0.92,
        message="FRAUD DETECTED: This transaction is likely fraudulent.",
    )
    payload = {
        "amount": 150000.0,
        "transaction_type": "TRANSFER",
        "old_balance_orig": 200000.0,
        "new_balance_orig": 50000.0,
        "old_balance_dest": 0.0,
        "new_balance_dest": 150000.0,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] == 0.92


@patch("app.routers.predict._service")
def test_predict_legitimate(mock_service):
    mock_service.predict.return_value = MagicMock(
        is_fraud=False,
        fraud_probability=0.03,
        message="LEGITIMATE: This transaction appears to be genuine.",
    )
    payload = {
        "amount": 500.0,
        "transaction_type": "PAYMENT",
        "old_balance_orig": 10000.0,
        "new_balance_orig": 9500.0,
        "old_balance_dest": 0.0,
        "new_balance_dest": 500.0,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is False
