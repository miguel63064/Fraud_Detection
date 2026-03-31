"""
Fraud Detection API
-------------------
Serves the LightGBM model trained in main.py via MLflow.
Uses transform() from src/feature_engineer.py for inference preprocessing.
"""

import os
import joblib
import mlflow
import pandas as pd
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Annotated, Optional
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import logging

from src.feature_engineer import transform
from src.monitoring import detect_drift

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

# Configure MLflow tracking URI from environment (supports file:// and http://)
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
API_KEY = os.getenv("FRAUD_API_KEY")
if not API_KEY:
    raise RuntimeError("FRAUD_API_KEY environment variable is not set.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)) -> str:
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return key

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
artifacts = None        # training artifacts: medians, encoder, freq_maps, ...
reference_stats = None  # training distribution stats for drift detection

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and training artifacts from MLflow on application startup."""
    global model, artifacts, reference_stats
    await _load_model()
    yield


async def _load_model() -> None:
    global model, artifacts, reference_stats

    run_id = os.getenv("MLFLOW_RUN_ID")

    if not run_id:
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud_detection")
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)

        if exp is None:
            raise RuntimeError(
                f"MLflow experiment '{experiment_name}' not found. "
                "Mount the mlruns directory with -v and set MLFLOW_TRACKING_URI=file:///app/mlruns."
            )

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            raise RuntimeError(
                f"No finished runs found in experiment '{experiment_name}'. "
                "Train the model first (python main.py) and mount the mlruns directory."
            )

        run_id = runs[0].info.run_id

    logger.info(f"Loading model from run: {run_id}")

    tracking_uri = mlflow.get_tracking_uri()
    if tracking_uri.startswith("file:"):
        # artifact_uri in meta.yaml stores the absolute host path, which may
        # not exist inside the container. Build the path from the configured URI.
        base_path = tracking_uri[len("file:"):].lstrip("/")
        if not base_path.startswith("/"):
            base_path = "/" + base_path
        client = mlflow.tracking.MlflowClient()
        exp_id = client.get_run(run_id).info.experiment_id
        artifact_base = os.path.join(base_path, exp_id, run_id, "artifacts")
        model = mlflow.sklearn.load_model(os.path.join(artifact_base, "model"))
        artifacts = joblib.load(
            os.path.join(artifact_base, "train_artifacts", "artifacts.pkl")
        )
        ref_path = os.path.join(artifact_base, "train_artifacts", "reference_stats.json")
    else:
        # Remote tracking (http://, s3://, etc.)
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        client = mlflow.tracking.MlflowClient()
        local_dir = client.download_artifacts(run_id, "train_artifacts")
        artifacts = joblib.load(os.path.join(local_dir, "artifacts.pkl"))
        ref_path = os.path.join(local_dir, "reference_stats.json")

    if os.path.exists(ref_path):
        import json
        with open(ref_path) as f:
            reference_stats = json.load(f)
        logger.info("Reference statistics loaded — drift detection is active.")
    else:
        logger.warning(
            "reference_stats.json not found — retrain the model to enable drift detection."
        )

    logger.info("Model and artifacts loaded successfully.")

# ---------------------------------------------------------------------------
# App instance (declared after lifespan so the reference is valid)
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for financial transactions — LightGBM + MLflow.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    TransactionDT: Annotated[float, Field(gt=0)]
    TransactionAmt: Annotated[float, Field(gt=0)]
    ProductCD: Optional[str] = None
    card1: Optional[float] = None
    card2: Optional[float] = None
    card4: Optional[str] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    C1: Optional[float] = None
    C2: Optional[float] = None
    D1: Optional[float] = None
    D2: Optional[float] = None
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    id_12: Optional[str] = None
    id_15: Optional[str] = None
    id_16: Optional[str] = None
    id_30: Optional[str] = None
    id_31: Optional[str] = None
    id_33: Optional[str] = None
    id_34: Optional[str] = None
    id_35: Optional[str] = None
    id_36: Optional[str] = None
    id_37: Optional[str] = None
    id_38: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionDT": 86400,
                "TransactionAmt": 150.5,
                "ProductCD": "W",
                "card1": 9500,
                "card4": "visa",
                "card6": "debit",
                "addr1": 315.0,
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "gmail.com",
                "C1": 1.0,
                "D1": 14.0,
                "M1": "T",
            }
        }


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str


class BatchInput(BaseModel):
    transactions: list[TransactionInput]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    fraud_count: int


class DriftReport(BaseModel):
    drift_detected: bool
    n_samples: int
    drifted_features: list[str]
    numerical: dict
    categorical: dict


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_one(transaction: TransactionInput) -> PredictionResponse:
    df = pd.DataFrame([transaction.model_dump()])

    df = transform(df, artifacts)

    # Align columns to match those expected by the model
    if hasattr(model, 'feature_name_'):
        for col in model.feature_name_:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_name_]

    prob = float(model.predict_proba(df)[0][1])
    risk = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.6 else "HIGH"

    return PredictionResponse(
        is_fraud=prob >= 0.5,
        fraud_probability=round(prob, 4),
        risk_level=risk,
    )

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
async def health() -> dict:
    """Health check. Returns whether the model is loaded and ready."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@limiter.limit("60/minute")
async def predict(
    request: Request,
    transaction: TransactionInput,
    _: str = Depends(verify_api_key),
):
    """Score a single transaction and return fraud probability."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return predict_one(transaction)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
@limiter.limit("10/minute")
async def predict_batch(
    request: Request,
    payload: BatchInput,
    _: str = Depends(verify_api_key),
):
    """Score a batch of transactions in a single request."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    predictions = [predict_one(t) for t in payload.transactions]
    return BatchResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=sum(1 for p in predictions if p.is_fraud),
    )


@app.post("/monitor/drift", response_model=DriftReport, tags=["Monitoring"])
@limiter.limit("20/minute")
async def monitor_drift(
    request: Request,
    payload: BatchInput,
    _: str = Depends(verify_api_key),
):
    """
    Compare a batch of incoming transactions against the training distribution.

    Returns a drift report with per-feature KS / chi-squared test results.
    Requires ``reference_stats.json`` in the MLflow run artifacts — generated
    automatically when you train with ``python main.py``.
    """
    if reference_stats is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Drift detection unavailable — retrain the model to generate "
                "reference_stats.json."
            ),
        )
    batch_df = pd.DataFrame([t.model_dump() for t in payload.transactions])
    report = detect_drift(batch_df, reference_stats)
    return DriftReport(**report)