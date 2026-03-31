from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
import mlflow
import pandas as pd


def evaluation(model, x_cv, y_cv, x_test, y_test):
    cv_proba = model.predict_proba(x_cv)[:, 1]
    test_proba = model.predict_proba(x_test)[:, 1]

    # Convert probabilities to binary labels (threshold 0.3)
    cv_preds = (cv_proba >= 0.3).astype(int)
    test_preds = (test_proba >= 0.3).astype(int)

    metrics = {
        # AUC-ROC
        "cv_auc": roc_auc_score(y_cv, cv_proba),
        "test_auc": roc_auc_score(y_test, test_proba),
        # PR-AUC (better than ROC-AUC for imbalanced datasets like fraud)
        "cv_pr_auc": average_precision_score(y_cv, cv_proba),
        "test_pr_auc": average_precision_score(y_test, test_proba),
        # Precision
        "cv_precision": precision_score(y_cv, cv_preds, zero_division=0),
        "test_precision": precision_score(y_test, test_preds, zero_division=0),
        # Recall
        "cv_recall": recall_score(y_cv, cv_preds, zero_division=0),
        "test_recall": recall_score(y_test, test_preds, zero_division=0),
        # F1
        "cv_f1": f1_score(y_cv, cv_preds, zero_division=0),
        "test_f1": f1_score(y_test, test_preds, zero_division=0),
    }

    # Print
    print("\n── Validation ──────────────────────────")
    print(f"  AUC-ROC  : {metrics['cv_auc']:.4f}")
    print(f"  PR AUC   : {metrics['cv_pr_auc']:.4f}")
    print(f"  Precision: {metrics['cv_precision']:.4f}")
    print(f"  Recall   : {metrics['cv_recall']:.4f}")
    print(f"  F1       : {metrics['cv_f1']:.4f}")

    print("\n── Test ────────────────────────────────")
    print(f"  AUC-ROC  : {metrics['test_auc']:.4f}")
    print(f"  PR AUC   : {metrics['test_pr_auc']:.4f}")
    print(f"  Precision: {metrics['test_precision']:.4f}")
    print(f"  Recall   : {metrics['test_recall']:.4f}")
    print(f"  F1       : {metrics['test_f1']:.4f}")

    plot_importance(model)

    # Log all metrics to MLflow
    mlflow.log_metrics(metrics)


def final_evaluation(model, x_final_test):
    return model.predict_proba(x_final_test)[:, 1]


def plot_importance(model):
    importance = pd.DataFrame(
        {"feature": model.feature_name_, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print("\nTop 10 features:")
    print(importance.head(10).to_string())
    print(f"\nFeatures with zero importance: {(importance['importance']==0).sum()}")
    print(importance[importance["importance"] == 0]["feature"].tolist())

    return importance
