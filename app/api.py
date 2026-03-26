"""
Fraud Detection API
-------------------
Serve o modelo LightGBM treinado no main.py via MLflow.
Usa diretamente o transform() do src/feature_engineer.py.
"""

import os
import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import logging

from src.feature_engineer import transform                           # ← importa direto

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Detecção de fraude em transações — modelo LightGBM + MLflow.",
    version="1.0.0",
)

# Configura tracking URI via variável de ambiente (suporta file:// e http://)
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")

# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------
API_KEY = os.getenv("FRAUD_API_KEY", "fraud")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="API Key inválida")
    return key

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------
model = None
artifacts = None   # artefactos do treino (medianas, encoder, freq_maps, ...)

# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def load_model():
    global model, artifacts

    run_id = os.getenv("MLFLOW_RUN_ID")

    if not run_id:
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud_detection")
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)

        if exp is None:
            raise RuntimeError(
                f"Experimento MLflow '{experiment_name}' não encontrado. "
                "Monte o diretório mlruns com -v e defina MLFLOW_TRACKING_URI=file:///app/mlruns."
            )

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if not runs:
            raise RuntimeError(
                f"Nenhum run FINISHED encontrado no experimento '{experiment_name}'. "
                "Treine o modelo antes (python main.py) e monte o mlruns no container."
            )

        run_id = runs[0].info.run_id

    logger.info(f"A carregar modelo do run: {run_id}")

    tracking_uri = mlflow.get_tracking_uri()
    if tracking_uri.startswith("file:"):
        # O artifact_uri no meta.yaml guarda o path absoluto do host.
        # Construímos o path a partir do tracking URI configurado no container.
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
    else:
        # Tracking remoto (http://, s3://, etc.)
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        client = mlflow.tracking.MlflowClient()
        local_dir = client.download_artifacts(run_id, "train_artifacts")
        artifacts = joblib.load(os.path.join(local_dir, "artifacts.pkl"))

    logger.info("Modelo e artefactos carregados com sucesso!")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    TransactionDT: float
    TransactionAmt: float
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

# ---------------------------------------------------------------------------
# Previsão
# ---------------------------------------------------------------------------
def predict_one(transaction: TransactionInput) -> PredictionResponse:
    df = pd.DataFrame([transaction.model_dump()])

    df = transform(df, artifacts)           # ← usa a função do teu src/

    # Alinha colunas com as que o modelo espera
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
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    transaction: TransactionInput,
    _: str = Depends(verify_api_key),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    try:
        return predict_one(transaction)
    except Exception as e:
        logger.error(f"Erro: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(
    payload: BatchInput,
    _: str = Depends(verify_api_key),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    predictions = [predict_one(t) for t in payload.transactions]
    return BatchResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=sum(1 for p in predictions if p.is_fraud),
    )