import lightgbm as lgb
import xgboost as xgb
import mlflow
import optuna
from sklearn.metrics import roc_auc_score


def lgb_model(scale_pos_weight):
    params = dict(
        n_estimators=3124,
        learning_rate=0.015,
        num_leaves=288,
        max_depth=12,
        min_child_samples=30,
        subsample=0.75,
        colsample_bytree=0.73,
        reg_lambda=1,
        reg_alpha=1.2,
        early_stopping_rounds=100
    )
    mlflow.log_params({f"lgb_{k}": v for k, v in params.items()})
    return lgb.LGBMClassifier(**params)


def xgb_model(scale_pos_weight):
    params = dict(
        n_estimators=500,         # número máximo de árvores
        learning_rate=0.03,       # taxa de aprendizado
        max_depth=6,              # profundidade da árvore
        min_child_weight=5,       # Exige mais amostras por folha (evita memorizar fraudes raras)
        gamma=1,
        subsample=0.8,            # amostragem de linhas
        colsample_bytree=0.8,     # amostragem de features
        scale_pos_weight=scale_pos_weight,  # balanceamento
        eval_metric='auc',        # métrica de avaliação
        random_state=42,
        early_stopping_rounds=50
    )
    mlflow.log_params({f"xgb_{k}": v for k, v in params.items()})
    return xgb.XGBClassifier(**params)

def train_model(model, x_train, y_train, x_cv, y_cv):
    model.fit(
        x_train, y_train,
        eval_set=[(x_cv, y_cv)],
    )
    mlflow.sklearn.log_model(model, "model")
    


def optimize_lgb(x_train, y_train, x_cv, y_cv, x_test, y_test, n_trials=50):
    
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 3000, 5000),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.05),
            num_leaves=trial.suggest_int('num_leaves', 30, 300),
            max_depth=trial.suggest_int('max_depth', 3, 12),
            min_child_samples=trial.suggest_int('min_child_samples', 20, 200),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_lambda=trial.suggest_float('reg_lambda', 0.1, 2.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 2.0),
            early_stopping_rounds=100,
            verbose=-1
        )
        
        model = lgb.LGBMClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_cv, y_cv)])
        
        preds = model.predict_proba(x_cv)[:, 1]
        preds_test = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_cv, preds)
        test_auc = roc_auc_score(y_test, preds_test)
        trial.set_user_attr("test_auc", test_auc)
        return auc
    
    def callback(study, trial):
        print(f'Trial {trial.number:3d} | AUC: {trial.value:.4f} | Test AUC: {trial.user_attrs.get("test_auc", 0):.4f} | Best: {study.best_value:.4f}')
        
        # Regista cada trial como nested run no MLflow
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("test_auc", trial.user_attrs.get("test_auc", 0))
            mlflow.log_metric("cv_auc", trial.value)

    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    print(f'\nMelhor AUC: {study.best_value:.4f}')
    print(f'Melhores parâmetros: {study.best_params}')
    
    return study.best_params

def best_optimize_lgb(x_train, y_train, x_cv, y_cv, x_test, y_test, n_trials):
    best_params = optimize_lgb(x_train, y_train, x_cv, y_cv, x_test, y_test, n_trials=n_trials)
    best_params['early_stopping_rounds'] = 100
    model = lgb.LGBMClassifier(**best_params)
    
    return model