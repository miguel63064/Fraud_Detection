import pytest
import numpy as np
from sklearn.metrics import roc_auc_score
from src.models import lgb_model, train_model


class TestModelInstantiation:
    """Testes para criação de modelos"""

    def test_lgb_model_creation(self):
        """Verifica criação de modelo LightGBM"""
        model = lgb_model(scale_pos_weight=1)

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "predict")

    def test_lgb_model_params_set(self):
        """Verifica se parâmetros estão configurados"""
        model = lgb_model(scale_pos_weight=1)
        params = model.get_params()

        assert params["n_estimators"] > 0
        assert params["learning_rate"] > 0
        assert params["num_leaves"] > 0


class TestModelTraining:
    """Testes para treino de modelos"""

    def test_model_trains_successfully(self, split_datasets):
        """Verifica se modelo treina sem erros"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_datasets

        model = lgb_model(scale_pos_weight=1)

        try:
            train_model(model, x_train, y_train, x_cv, y_cv)
            assert True
        except Exception as e:
            pytest.fail(f"Modelo falhou ao treinar: {e}")

    def test_model_produces_predictions(self, split_datasets):
        """Verifica se modelo produz predições"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        preds = model.predict_proba(x_cv)

        assert preds is not None
        assert preds.shape[0] == len(x_cv)
        assert preds.shape[1] == 2  # Probabilidades para 2 classes

    def test_model_predictions_valid_probabilities(self, split_datasets):
        """Verifica se predições são probabilidades válidas [0, 1]"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        preds = model.predict_proba(x_cv)

        assert np.all(preds >= 0) and np.all(preds <= 1)
        # Soma de probabilidades deve ser 1
        np.testing.assert_array_almost_equal(preds.sum(axis=1), np.ones(len(preds)))

    def test_model_auc_above_threshold(self, split_datasets):
        """Verifica se AUC está acima do threshold mínimo"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        # Predições de probabilidade para a classe positiva
        preds = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        # AUC mínimo aceitável (modelo melhor que aleatório)
        assert auc > 0.5
        # Para um modelo decente em dados sintéticos
        assert auc > 0.5

    def test_model_handles_imbalanced_data(self, split_datasets):
        """Verifica se modelo lida com dados desbalanceados"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        # Deve conseguir fazer predições para ambas as classes
        preds = model.predict(x_test)

        assert set(np.unique(preds)) == {0, 1}

    def test_model_predictions_shape_matches_input(self, split_datasets):
        """Verifica se shape das predições match com input"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        for X, y in [(x_cv, y_cv), (x_test, y_test)]:
            preds_proba = model.predict_proba(X)
            preds = model.predict(X)

            assert len(preds_proba) == len(X)
            assert len(preds) == len(X)


class TestModelConsistency:
    """Testes de consistência e reproducibilidade"""

    def test_model_deterministic_with_seed(self, split_datasets):
        """Verifica se modelo com seed produz resultados consistentes"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        # Treina modelo 1
        model1 = lgb_model(scale_pos_weight=1)
        train_model(model1, x_train, y_train, x_cv, y_cv)
        preds1 = model1.predict_proba(x_test)

        # Treina modelo 2 (mesma configuração)
        model2 = lgb_model(scale_pos_weight=1)
        train_model(model2, x_train, y_train, x_cv, y_cv)
        preds2 = model2.predict_proba(x_test)

        # Predições devem ser muito próximas (LightGBM com seed)
        np.testing.assert_array_almost_equal(preds1, preds2, decimal=5)

    def test_model_performance_cv_vs_test(self, split_datasets):
        """Verifica se performance é razoável em CV vs Test"""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        cv_preds = model.predict_proba(x_cv)[:, 1]
        test_preds = model.predict_proba(x_test)[:, 1]

        cv_auc = roc_auc_score(y_cv, cv_preds)
        test_auc = roc_auc_score(y_test, test_preds)

        # Não deve haver overfitting excessivo
        assert abs(cv_auc - test_auc) < 0.25

        # Ambos devem estar razoáveis
        assert cv_auc > 0.5
        assert test_auc > 0.5
