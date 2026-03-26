import pytest
import numpy as np
from sklearn.metrics import roc_auc_score
from src.models import lgb_model, train_model


class TestModelInstantiation:
    """Tests for model instantiation"""

    def test_lgb_model_creation(self):
        """Verify that a LightGBM model is created with the expected interface."""
        model = lgb_model(scale_pos_weight=1)

        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "predict")

    def test_lgb_model_params_set(self):
        """Verify that key hyperparameters are set to valid values."""
        model = lgb_model(scale_pos_weight=1)
        params = model.get_params()

        assert params["n_estimators"] > 0
        assert params["learning_rate"] > 0
        assert params["num_leaves"] > 0


class TestModelTraining:
    """Tests for model training"""

    @pytest.mark.smoke
    def test_model_trains_successfully(self, split_datasets):
        """Verify that the model trains without raising an exception."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_datasets

        model = lgb_model(scale_pos_weight=1)

        try:
            train_model(model, x_train, y_train, x_cv, y_cv)
            assert True
        except Exception as e:
            pytest.fail(f"Model failed to train: {e}")

    def test_model_produces_predictions(self, split_datasets):
        """Verify that predict_proba returns an array of the expected shape."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        preds = model.predict_proba(x_cv)

        assert preds is not None
        assert preds.shape[0] == len(x_cv)
        assert preds.shape[1] == 2  # Probabilities for 2 classes

    def test_model_predictions_valid_probabilities(self, split_datasets):
        """Verify that output probabilities are in [0, 1] and sum to 1."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        preds = model.predict_proba(x_cv)

        assert np.all(preds >= 0) and np.all(preds <= 1)
        # Probabilities for both classes must sum to 1
        np.testing.assert_array_almost_equal(preds.sum(axis=1), np.ones(len(preds)))

    def test_model_auc_above_threshold(self, split_datasets):
        """Verify AUC is above the minimum acceptable threshold."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        # Positive class probability predictions
        preds = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        # AUC must exceed random baseline
        assert auc > 0.5
        # For a reasonably trained model on synthetic data
        assert auc > 0.5

    def test_model_handles_imbalanced_data(self, split_datasets):
        """Verify the model predicts both classes on imbalanced data."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        # Deve conseguir fazer predições para ambas as classes
        preds = model.predict(x_test)

        assert set(np.unique(preds)) == {0, 1}

    def test_model_predictions_shape_matches_input(self, split_datasets):
        """Verify prediction array shapes match the input size."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        for X, y in [(x_cv, y_cv), (x_test, y_test)]:
            preds_proba = model.predict_proba(X)
            preds = model.predict(X)

            assert len(preds_proba) == len(X)
            assert len(preds) == len(X)


class TestModelConsistency:
    """Tests for model consistency and reproducibility"""

    def test_model_deterministic_with_seed(self, split_datasets):
        """Verify that two models trained identically produce the same predictions."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        # Train model 1
        model1 = lgb_model(scale_pos_weight=1)
        train_model(model1, x_train, y_train, x_cv, y_cv)
        preds1 = model1.predict_proba(x_test)

        # Train model 2 (same configuration)
        model2 = lgb_model(scale_pos_weight=1)
        train_model(model2, x_train, y_train, x_cv, y_cv)
        preds2 = model2.predict_proba(x_test)

        # Predictions should be nearly identical (LightGBM with fixed seed)
        np.testing.assert_array_almost_equal(preds1, preds2, decimal=5)

    def test_model_performance_cv_vs_test(self, split_datasets):
        """Verify that CV and test AUC are reasonably close (no severe overfitting)."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight = split_datasets

        model = lgb_model(scale_pos_weight=1)
        train_model(model, x_train, y_train, x_cv, y_cv)

        cv_preds = model.predict_proba(x_cv)[:, 1]
        test_preds = model.predict_proba(x_test)[:, 1]

        cv_auc = roc_auc_score(y_cv, cv_preds)
        test_auc = roc_auc_score(y_test, test_preds)

        # No excessive overfitting
        assert abs(cv_auc - test_auc) < 0.25

        # Both should be reasonable
        assert cv_auc > 0.5
        assert test_auc > 0.5
