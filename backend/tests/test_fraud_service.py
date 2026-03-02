import pytest
from unittest.mock import patch, mock_open, MagicMock
import pickle


@patch("builtins.open", new_callable=mock_open, read_data=pickle.dumps(MagicMock()))
@patch("app.services.fraud_service.Path.exists", return_value=True)
def test_fraud_service_loads_model(mock_exists, mock_file):
    from app.services.fraud_service import FraudService
    service = FraudService()
    assert service.model is not None


@patch("app.services.fraud_service.Path.exists", return_value=False)
def test_fraud_service_raises_when_model_missing(mock_exists):
    from app.services.fraud_service import FraudService
    with pytest.raises(FileNotFoundError):
        FraudService()
