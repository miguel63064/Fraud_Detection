from sklearn.metrics import roc_auc_score
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

def evaluation(model, x_cv, y_cv, x_test, y_test):
    cv_preds   = model.predict_proba(x_cv)[:, 1]
    test_preds = model.predict_proba(x_test)[:, 1]

    cv_auc = roc_auc_score(y_cv, cv_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    print(f"CV AUC: {cv_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    plot_importance(model)
    mlflow.log_metric("cv_auc", cv_auc)
    mlflow.log_metric("test_auc", test_auc)
    
def final_evaluation(model, x_final_test):
    return model.predict_proba(x_final_test)[:, 1]

def plot_importance(model):
    importance = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features:")
    print(importance.head(10).to_string())
    print(f"\nFeatures com importance zero: {(importance['importance']==0).sum()}")
    print(f"Features com importance zero: {importance[importance['importance']==0]['feature'].tolist()}")
    
    return importance