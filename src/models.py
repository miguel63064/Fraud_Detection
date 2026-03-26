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
        early_stopping_rounds=100,
    )
    return lgb.LGBMClassifier(**params)


def xgb_model(scale_pos_weight):
    params = dict(
        n_estimators=500,        # max trees
        learning_rate=0.03,      # shrinkage factor per step
        max_depth=6,             # tree depth
        min_child_weight=5,      # min samples per leaf (reduces rare-fraud overfitting)
        gamma=1,
        subsample=0.8,           # row subsampling
        colsample_bytree=0.8,    # feature subsampling
        scale_pos_weight=scale_pos_weight,  # class imbalance weight
        eval_metric="auc",       # early-stopping metric
        random_state=42,
        early_stopping_rounds=50,
    )
    return xgb.XGBClassifier(**params)


def train_model(model, x_train, y_train, x_cv, y_cv):
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_cv, y_cv)],
    )


def optimize_lgb(x_train, y_train, x_cv, y_cv, x_test, y_test, n_trials=50):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 1000, 5000),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            num_leaves=trial.suggest_int("num_leaves", 32, 256),
            max_depth=-1,
            min_child_samples=trial.suggest_int("min_child_samples", 50, 300),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 0.9),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.1, 10, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
            early_stopping_rounds=100,
            subsample_freq=1,
            verbose=-1,
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
        print(
            f'Trial {trial.number:3d} | AUC: {trial.value:.4f} | Test AUC: {trial.user_attrs.get("test_auc", 0):.4f} | Best: {study.best_value:.4f}'
        )

        # Log each trial as a nested MLflow run
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric("test_auc", trial.user_attrs.get("test_auc", 0))
            mlflow.log_metric("cv_auc", trial.value)

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    return study.best_params


def best_optimize_lgb(x_train, y_train, x_cv, y_cv, x_test, y_test, n_trials):
    best_params = optimize_lgb(
        x_train, y_train, x_cv, y_cv, x_test, y_test, n_trials=n_trials
    )
    best_params["early_stopping_rounds"] = 100
    model = lgb.LGBMClassifier(**best_params)

    return model
