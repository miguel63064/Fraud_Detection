"""
Training pipeline for the fraud detection model.

Loads data, engineers features, trains a LightGBM classifier, logs
everything to MLflow, and writes a Kaggle submission CSV.
"""
import json
import os

import joblib
import mlflow

from src.evaluation import evaluation, final_evaluation, plot_importance
from src.feature_engineer import fit_transform
from src.load_data import load_data, prepare_final_test, split_train_data
from src.models import (
    best_optimize_lgb,
    best_optimize_xgb,
    lgb_model,
    xgb_model,
    optimize_lgb,
    best_optimize_xgb,
    train_model,
    xgb_model,
)
from src.monitoring import compute_reference_stats
from src.predict import submission


mlflow.set_experiment("fraud_detection")

with mlflow.start_run(run_name="lgb_model"):
    train, test = load_data()
    print("Data loaded.")

    # Compute reference stats before fit_transform so raw distributions are captured
    reference_stats = compute_reference_stats(train)

    train, test, artifacts = fit_transform(train, test)
    print("Features engineered.")

    x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_train_data(
        train
    )
    x_final_test, test_ids = prepare_final_test(test)

    #model = best_optimize_xgb(x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight, 50)
    model = lgb_model(scale_pos_weight)
    train_model(model, x_train, y_train, x_cv, y_cv)

    mlflow.log_params({f"lgb_{k}": v for k, v in model.get_params().items()})
    mlflow.sklearn.log_model(model, "model")

    os.makedirs("tmp_artifacts", exist_ok=True)
    joblib.dump(artifacts, "tmp_artifacts/artifacts.pkl")
    with open("tmp_artifacts/reference_stats.json", "w") as f:
        json.dump(reference_stats, f)
    mlflow.log_artifacts("tmp_artifacts", artifact_path="train_artifacts")

    evaluation(model, x_cv, y_cv, x_test, y_test)
    final_preds = final_evaluation(model, x_final_test)

    submission(test_ids, final_preds)
