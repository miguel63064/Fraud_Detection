import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from src.models import lgb_model, train_model
from src.evaluation import evaluation, final_evaluation, plot_importance


class TestEvaluation:
    """Testes para funções de avaliação"""

    def test_evaluation_runs_without_error(self, split_datasets, mocker):
        """Verifica se evaluation roda sem erros"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=scale_pos_weight)
        train_model(model, x_train, y_train, x_cv, y_cv)

        # Mock MLflow para não interferir
        mocker.patch("src.evaluation.mlflow")

        try:
            evaluation(model, x_cv, y_cv, x_test, y_test)
            assert True
        except Exception as e:
            pytest.fail(f"Evaluation falhou: {e}")

    def test_final_evaluation_returns_predictions(self, split_datasets):
        """Verifica se final_evaluation retorna predições válidas"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=scale_pos_weight)
        train_model(model, x_train, y_train, x_cv, y_cv)

        preds = final_evaluation(model, x_test)

        assert preds is not None
        assert len(preds) == len(x_test)
        assert np.all(preds >= 0) and np.all(preds <= 1)


class TestMetricsCalculation:
    """Testes para cálculo de métricas de avaliação"""

    def test_auc_calculation_valid(self, split_datasets):
        """Verifica se AUC é calculado corretamente"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=scale_pos_weight)
        train_model(model, x_train, y_train, x_cv, y_cv)

        preds = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        assert 0 < auc < 1
        assert isinstance(auc, (float, np.floating))

    def test_precision_recall_valid(self, split_datasets):
        """Verifica se precision e recall são válidos"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=scale_pos_weight)
        train_model(model, x_train, y_train, x_cv, y_cv)

        preds = model.predict(x_test)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1

    @pytest.mark.smoke
    def test_metrics_consistent_across_splits(self, split_datasets):
        """Verifica se métricas são consistentes entre CV e Test"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=scale_pos_weight)
        train_model(model, x_train, y_train, x_cv, y_cv)

        cv_preds = model.predict_proba(x_cv)[:, 1]
        test_preds = model.predict_proba(x_test)[:, 1]

        cv_auc = roc_auc_score(y_cv, cv_preds)
        test_auc = roc_auc_score(y_test, test_preds)

        # Ambos devem estar razoáveis
        assert cv_auc > 0.7
        assert test_auc > 0.7


class TestPredictionDistribution:
    """Testes sobre distribuição de predições"""

    def test_prediction_probabilities_have_variance(self, split_datasets):
        """Verifica se probabilidades têm variância (não constantes)"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=scale_pos_weight)
        train_model(model, x_train, y_train, x_cv, y_cv)

        preds = model.predict_proba(x_test)[:, 1]

        # Deve haver variação nas probabilidades
        assert preds.std() > 0
        assert preds.min() < preds.max()


class TestModelCalibration:
    """Testes de calibração do modelo"""

    def test_positive_predictions_lower_than_mean_probability(self, split_datasets):
        """Verifica se predições positivas têm probabilidade > 0.5"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=scale_pos_weight)
        train_model(model, x_train, y_train, x_cv, y_cv)

        preds_proba = model.predict_proba(x_test)[:, 1]
        preds_binary = model.predict(x_test)

        # Quando binary prediction é 1, proba deve ser > 0.5
        positive_predicted = preds_binary == 1
        if positive_predicted.any():
            assert (preds_proba[positive_predicted] > 0.5).all()
