"""
Tests for app/api.py

All MLflow and model loading is mocked — tests run without a trained model
or a running MLflow server.  The FastAPI TestClient is used for all requests.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures — mock model, artifacts, reference_stats
# ---------------------------------------------------------------------------

VALID_TRANSACTION = {
    "TransactionDT": 86400.0,
    "TransactionAmt": 150.5,
    "ProductCD": "W",
    "card1": 9500.0,
    "card4": "visa",
    "card6": "debit",
    "addr1": 315.0,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "C1": 1.0,
    "D1": 14.0,
    "M1": "T",
}

REFERENCE_STATS = {
    "numerical": {
        "TransactionAmt": {
            "mean": 150.0,
            "std": 50.0,
            "min": 10.0,
            "max": 500.0,
            "p25": 80.0,
            "p50": 130.0,
            "p75": 200.0,
            "values_sample": list(np.random.normal(150, 50, 500).tolist()),
        }
    },
    "categorical": {
        "card4": {"frequencies": {"visa": 0.6, "mastercard": 0.3, "amex": 0.1}}
    },
}


def _make_mock_model(prob: float = 0.1):
    """Return a mock LightGBM-like model that always returns `prob`."""
    mock = MagicMock()
    mock.predict_proba.return_value = np.array([[1 - prob, prob]])
    mock.feature_name_ = []  # no column alignment
    return mock


def _make_artifacts():
    """Minimal artifacts dict — transform() will use these."""
    return {
        "medians": {},
        "enc": None,
        "uid1_agg": None,
        "freq_maps": {},
        "dt_max": 1.0,
    }


@pytest.fixture()
def client_no_model(mocker):
    """TestClient with model=None (simulates server not ready)."""
    import app.api as api_module

    mocker.patch.object(api_module, "_load_model", new=AsyncMock())
    mocker.patch.object(api_module, "model", None)
    mocker.patch.object(api_module, "artifacts", None)
    mocker.patch.object(api_module, "reference_stats", None)
    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def client(mocker):
    """TestClient with a fully mocked model and artifacts."""
    import app.api as api_module

    mock_model = _make_mock_model(prob=0.1)
    mock_artifacts = _make_artifacts()

    mocker.patch.object(api_module, "_load_model", new=AsyncMock())
    mocker.patch.object(api_module, "model", mock_model)
    mocker.patch.object(api_module, "artifacts", mock_artifacts)
    mocker.patch.object(api_module, "reference_stats", REFERENCE_STATS)
    mocker.patch(
        "app.api.transform",
        return_value=pd.DataFrame(
            [
                {
                    "TransactionDT": 86400.0,
                    "TransactionAmt": 150.5,
                }
            ]
        ),
    )

    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def client_fraud(mocker):
    """TestClient where model always predicts fraud (prob=0.9)."""
    import app.api as api_module

    mock_model = _make_mock_model(prob=0.9)
    mock_artifacts = _make_artifacts()

    mocker.patch.object(api_module, "_load_model", new=AsyncMock())
    mocker.patch.object(api_module, "model", mock_model)
    mocker.patch.object(api_module, "artifacts", mock_artifacts)
    mocker.patch.object(api_module, "reference_stats", REFERENCE_STATS)
    mocker.patch(
        "app.api.transform",
        return_value=pd.DataFrame(
            [
                {
                    "TransactionDT": 86400.0,
                    "TransactionAmt": 150.5,
                }
            ]
        ),
    )

    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


HEADERS = {"X-API-Key": "fraud"}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_keys(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_no_auth_required(self, client):
        # /health must be accessible without an API key
        resp = client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /predict — authentication
# ---------------------------------------------------------------------------


class TestPredictAuth:
    def test_missing_api_key_returns_403(self, client):
        resp = client.post("/predict", json=VALID_TRANSACTION)
        assert resp.status_code == 403

    def test_wrong_api_key_returns_403(self, client):
        resp = client.post(
            "/predict", json=VALID_TRANSACTION, headers={"X-API-Key": "wrong-key"}
        )
        assert resp.status_code == 403

    def test_valid_api_key_accepted(self, client):
        resp = client.post("/predict", json=VALID_TRANSACTION, headers=HEADERS)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /predict — input validation
# ---------------------------------------------------------------------------


class TestPredictValidation:
    def test_missing_required_field_returns_422(self, client):
        payload = {k: v for k, v in VALID_TRANSACTION.items() if k != "TransactionAmt"}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        assert resp.status_code == 422

    def test_negative_amount_returns_422(self, client):
        payload = {**VALID_TRANSACTION, "TransactionAmt": -10.0}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        assert resp.status_code == 422

    def test_zero_amount_returns_422(self, client):
        payload = {**VALID_TRANSACTION, "TransactionAmt": 0.0}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        assert resp.status_code == 422

    def test_negative_transaction_dt_returns_422(self, client):
        payload = {**VALID_TRANSACTION, "TransactionDT": -1.0}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        assert resp.status_code == 422

    def test_optional_fields_can_be_omitted(self, client):
        payload = {"TransactionDT": 86400.0, "TransactionAmt": 100.0}
        resp = client.post("/predict", json=payload, headers=HEADERS)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /predict — response schema
# ---------------------------------------------------------------------------


class TestPredictResponse:
    def test_response_keys(self, client):
        resp = client.post("/predict", json=VALID_TRANSACTION, headers=HEADERS)
        data = resp.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data

    def test_fraud_probability_in_range(self, client):
        resp = client.post("/predict", json=VALID_TRANSACTION, headers=HEADERS)
        prob = resp.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_level_valid_values(self, client):
        resp = client.post("/predict", json=VALID_TRANSACTION, headers=HEADERS)
        assert resp.json()["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

    def test_low_prob_not_fraud(self, client):
        # fixture uses prob=0.1 → is_fraud should be False
        resp = client.post("/predict", json=VALID_TRANSACTION, headers=HEADERS)
        assert resp.json()["is_fraud"] is False

    def test_high_prob_is_fraud(self, client_fraud):
        # fixture uses prob=0.9 → is_fraud should be True
        resp = client_fraud.post("/predict", json=VALID_TRANSACTION, headers=HEADERS)
        assert resp.json()["is_fraud"] is True

    def test_risk_level_low(self, client):
        resp = client.post("/predict", json=VALID_TRANSACTION, headers=HEADERS)
        assert resp.json()["risk_level"] == "LOW"

    def test_risk_level_high(self, client_fraud):
        resp = client_fraud.post("/predict", json=VALID_TRANSACTION, headers=HEADERS)
        assert resp.json()["risk_level"] == "HIGH"


# ---------------------------------------------------------------------------
# /predict — model not loaded
# ---------------------------------------------------------------------------


class TestPredictModelNotLoaded:
    def test_returns_503_when_model_none(self, client_no_model):
        resp = client_no_model.post("/predict", json=VALID_TRANSACTION, headers=HEADERS)
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------


class TestPredictBatch:
    def test_returns_200(self, client):
        payload = {"transactions": [VALID_TRANSACTION, VALID_TRANSACTION]}
        resp = client.post("/predict/batch", json=payload, headers=HEADERS)
        assert resp.status_code == 200

    def test_response_keys(self, client):
        payload = {"transactions": [VALID_TRANSACTION]}
        resp = client.post("/predict/batch", json=payload, headers=HEADERS)
        data = resp.json()
        assert "predictions" in data
        assert "total" in data
        assert "fraud_count" in data

    def test_total_matches_input(self, client):
        payload = {"transactions": [VALID_TRANSACTION] * 3}
        resp = client.post("/predict/batch", json=payload, headers=HEADERS)
        assert resp.json()["total"] == 3

    def test_predictions_length_matches_input(self, client):
        payload = {"transactions": [VALID_TRANSACTION] * 4}
        resp = client.post("/predict/batch", json=payload, headers=HEADERS)
        assert len(resp.json()["predictions"]) == 4

    def test_fraud_count_correct(self, client_fraud):
        payload = {"transactions": [VALID_TRANSACTION] * 3}
        resp = client_fraud.post("/predict/batch", json=payload, headers=HEADERS)
        assert resp.json()["fraud_count"] == 3

    def test_missing_api_key_returns_403(self, client):
        payload = {"transactions": [VALID_TRANSACTION]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 403

    def test_returns_503_when_model_none(self, client_no_model):
        payload = {"transactions": [VALID_TRANSACTION]}
        resp = client_no_model.post("/predict/batch", json=payload, headers=HEADERS)
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /monitor/drift
# ---------------------------------------------------------------------------


class TestMonitorDrift:
    def test_returns_200(self, client):
        payload = {"transactions": [VALID_TRANSACTION] * 10}
        resp = client.post("/monitor/drift", json=payload, headers=HEADERS)
        assert resp.status_code == 200

    def test_response_keys(self, client):
        payload = {"transactions": [VALID_TRANSACTION] * 10}
        resp = client.post("/monitor/drift", json=payload, headers=HEADERS)
        data = resp.json()
        for key in (
            "drift_detected",
            "n_samples",
            "drifted_features",
            "numerical",
            "categorical",
        ):
            assert key in data

    def test_n_samples_correct(self, client):
        payload = {"transactions": [VALID_TRANSACTION] * 5}
        resp = client.post("/monitor/drift", json=payload, headers=HEADERS)
        assert resp.json()["n_samples"] == 5

    def test_missing_api_key_returns_403(self, client):
        payload = {"transactions": [VALID_TRANSACTION]}
        resp = client.post("/monitor/drift", json=payload)
        assert resp.status_code == 403

    def test_returns_503_when_reference_stats_none(self, client_no_model):
        payload = {"transactions": [VALID_TRANSACTION]}
        resp = client_no_model.post("/monitor/drift", json=payload, headers=HEADERS)
        assert resp.status_code == 503
