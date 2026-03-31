# Fraud Detection — LightGBM + FastAPI + MLflow

Real-time fraud detection for financial transactions.  
Trains a LightGBM classifier, tracks experiments with MLflow, exposes a REST API via FastAPI, and monitors data drift in production.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Train the Model](#train-the-model)
4. [Run the API](#run-the-api)
5. [API Endpoints](#api-endpoints)
6. [Drift Detection](#drift-detection)
7. [Run Tests](#run-tests)
8. [Run with Docker](#run-with-docker)
9. [Run with Docker Compose](#run-with-docker-compose)
10. [MLflow UI](#mlflow-ui)
11. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
fraud_project/
├── app/
│   └── api.py                  # FastAPI application
├── src/
│   ├── load_data.py            # Data loading and splitting
│   ├── feature_engineer.py     # Feature engineering (fit_transform + transform)
│   ├── models.py               # LightGBM / XGBoost model definitions
│   ├── evaluation.py           # Metrics + MLflow logging
│   ├── monitoring.py           # Drift detection (KS + chi-squared tests)
│   └── predict.py              # Kaggle submission generator
├── scripts/
│   ├── simulate_drift.py       # Drift simulation against the live API
│   ├── run-tests.sh            # Test runner helper
│   ├── pre-commit-test.sh      # Pre-commit hook
│   └── setup-hooks.sh          # Install git hooks
├── tests/                      # Pytest test suite
├── data/raw/                   # Raw CSVs (Kaggle IEEE-CIS dataset)
├── mlruns/                     # MLflow local tracking store
├── tmp_artifacts/              # Intermediate artifacts (artifacts.pkl, reference_stats.json)
├── main.py                     # Training pipeline entry point
├── requirements.txt
├── requirements-dev.txt
└── Dockerfile
```

---

## Setup

### Prerequisites

- Python 3.11 (via conda or system)
- Raw data CSVs in `data/raw/` (IEEE-CIS Fraud Detection — Kaggle)

### 1. Create environment

```bash
conda create -n fraud-project python=3.11 -y
conda activate fraud-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt       # runtime deps
pip install -r requirements-dev.txt   # test deps (pytest, etc.)
```

### 3. Install git hooks (optional)

```bash
bash scripts/setup-hooks.sh
```

---

## Train the Model

Runs the full pipeline: load data → feature engineering → train LightGBM → log to MLflow → save artifacts.

```bash
python main.py
```

This produces:
- An MLflow run under experiment `fraud_detection`
- `tmp_artifacts/artifacts.pkl` — feature engineering artifacts (encoders, medians, freq maps)
- `tmp_artifacts/reference_stats.json` — training distribution statistics for drift detection
- `submissions/submission.csv` — Kaggle submission file

> **Note:** Training on the full dataset takes several minutes. Progress is printed to stdout.

---

## Run the API

```bash
python -m uvicorn app.api:app --reload
```

The API will be available at `http://localhost:8000`.

On startup, it automatically loads the latest finished MLflow run (model + artifacts).

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | local `mlruns/` | MLflow tracking URI (`file:///app/mlruns` or remote) |
| `MLFLOW_RUN_ID` | _(auto: latest finished run)_ | Pin a specific run ID |
| `MLFLOW_EXPERIMENT_NAME` | `fraud_detection` | Experiment name to search |
| `FRAUD_API_KEY` | `fraud` | API key required in `X-API-Key` header |

```bash
# Example with explicit run ID
MLFLOW_RUN_ID=abc123 python -m uvicorn app.api:app --reload
```

---

## API Endpoints

Interactive docs: **http://localhost:8000/docs**

### `GET /health`

Health check — no authentication required.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

---

### `POST /predict`

Score a single transaction.

```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: fraud" \
  -H "Content-Type: application/json" \
  -d '{
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
    "M1": "T"
  }'
```

```json
{
  "is_fraud": false,
  "fraud_probability": 0.0312,
  "risk_level": "LOW"
}
```

`risk_level` values: `LOW` (prob < 0.3), `MEDIUM` (0.3–0.6), `HIGH` (> 0.6).

---

### `POST /predict/batch`

Score multiple transactions in one request.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "X-API-Key: fraud" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"TransactionDT": 86400, "TransactionAmt": 150.5, "ProductCD": "W"},
      {"TransactionDT": 90000, "TransactionAmt": 9999.0, "ProductCD": "C"}
    ]
  }'
```

```json
{
  "predictions": [...],
  "total": 2,
  "fraud_count": 1
}
```

---

### `POST /monitor/drift`

Compare a batch of incoming transactions against the training distribution.  
Returns per-feature KS test (numerical) and chi-squared test (categorical) results.

```bash
curl -X POST http://localhost:8000/monitor/drift \
  -H "X-API-Key: fraud" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"TransactionDT": 86400, "TransactionAmt": 50000.0, "card4": "unknown"}
    ]
  }'
```

```json
{
  "drift_detected": true,
  "n_samples": 1,
  "drifted_features": ["TransactionAmt"],
  "numerical": {
    "TransactionAmt": {"ks_statistic": 0.95, "p_value": 0.0, "drift": true, "new_mean": 50000.0, "ref_mean": 135.03}
  },
  "categorical": {}
}
```

> **Requires** `reference_stats.json` in the MLflow run artifacts. Run `python main.py` first.  
> Returns `503` if the file is missing.

---

## Drift Detection

### How it works

| Data type | Test | Alert condition |
|---|---|---|
| Numerical | Kolmogorov-Smirnov (2-sample) | p-value < 0.05 |
| Categorical | Chi-squared | p-value < 0.05 |

`compute_reference_stats()` is called during training and captures the raw distribution of every column. At inference time, `detect_drift()` compares any incoming batch against those reference distributions.

### Simulate drift

With the API running, execute the simulation script:

```bash
python scripts/simulate_drift.py
```

This sends three batches to `/monitor/drift`:

| Scenario | Description | Expected result |
|---|---|---|
| `no_drift` | Sampled from the real training distribution | No alerts |
| `moderate_drift` | Transaction amounts ×5 higher | Some numerical drift |
| `severe_drift` | Amounts ×100 + unseen categories | Many alerts |

```bash
# Custom URL or API key
python scripts/simulate_drift.py --url http://localhost:8000 --api-key fraud

# Custom reference stats location
python scripts/simulate_drift.py --ref-stats tmp_artifacts/reference_stats.json
```

---

## Run Tests

### Quick smoke tests (~30 seconds)

```bash
bash scripts/run-tests.sh quick
```

### All tests (~3–5 minutes)

```bash
bash scripts/run-tests.sh all
```

### By module

```bash
bash scripts/run-tests.sh data      # data loading & splitting
bash scripts/run-tests.sh feature   # feature engineering
bash scripts/run-tests.sh model     # model training & predictions
```

### With coverage report

```bash
bash scripts/run-tests.sh coverage
xdg-open htmlcov/index.html         # Linux
```

### Directly with pytest

```bash
pytest tests/ -v                    # all tests, verbose
pytest tests/ -x                    # stop on first failure
pytest tests/ -m smoke              # only smoke tests
pytest tests/ -k "test_split"       # filter by name
pytest tests/ --pdb                 # drop into debugger on failure
```

### Pre-commit validation

```bash
bash scripts/pre-commit-test.sh
```

---

## Run with Docker

### Build

```bash
docker build -t fraud_app .
```

### Run

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  fraud_app
```

The container loads the latest finished run from the mounted `mlruns` directory.

### With a specific run

```bash
docker run -p 8000:8000 \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  -e MLFLOW_RUN_ID=<your_run_id> \
  fraud_app
```

---

## Run with Docker Compose

Orchestra a API + MLflow tracking server together with a single command.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/) installed
- Model already trained (`python main.py`) so `mlruns/` exists

### Start

```bash
docker-compose up --build
```

This starts two services:

| Service | URL | Description |
|---|---|---|
| `api` | http://localhost:8000 | Fraud Detection REST API |
| `mlflow` | http://localhost:5000/# | MLflow tracking UI |

### Stop

```bash
docker-compose down
```

### Custom API key

```bash
FRAUD_API_KEY=my-secret docker-compose up
```

Or create a `.env` file in the project root:

```
FRAUD_API_KEY=my-secret
```

### Rebuild after code changes

```bash
docker-compose up --build
```

### View logs

```bash
docker-compose logs -f api      # API logs only
docker-compose logs -f mlflow   # MLflow logs only
```

> **Note:** The `mlruns/` folder is mounted as a volume into both containers, so the model trained locally with `python main.py` is immediately available inside Docker without rebuilding.

---

## MLflow UI

```bash
mlflow ui
```

Open **http://localhost:5000/#** to browse experiments, compare runs, and inspect logged metrics and artifacts.

Metrics logged per run:

| Metric | Description |
|---|---|
| `cv_auc` / `test_auc` | AUC-ROC on validation / test set |
| `cv_pr_auc` / `test_pr_auc` | PR-AUC (better for imbalanced data) |
| `cv_precision` / `test_precision` | Precision at threshold 0.3 |
| `cv_recall` / `test_recall` | Recall at threshold 0.3 |
| `cv_f1` / `test_f1` | F1-score at threshold 0.3 |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pkg_resources'`

```bash
pip install "setuptools>=69.0.0,<81"
```

### API returns `503 Model not loaded`

The MLflow experiment or run was not found. Make sure you have trained the model first:

```bash
python main.py
```

Then restart the API and check that `mlruns/` is present.

### API returns `503 Drift detection unavailable`

`reference_stats.json` is missing from the MLflow run artifacts. Retrain the model:

```bash
python main.py
```

### Docker: `OSError: libgomp.so.1 not found`

The Dockerfile already includes `apt-get install -y libgomp1`. Rebuild the image:

```bash
docker build --no-cache -t fraud_app .
```

### Tests fail with import errors

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

