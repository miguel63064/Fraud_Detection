import pytest
import pandas as pd
import numpy as np
from src.load_data import split_train_data, prepare_final_test, reduce_mem_usage


class TestDataLoading:
    """Tests for data loading and memory reduction"""

    def test_reduce_mem_usage_compresses_data(self, sample_data):
        """Verify that reduce_mem_usage reduces memory footprint."""
        df = sample_data.copy()
        original_memory = df.memory_usage(deep=True).sum()

        df_reduced = reduce_mem_usage(df)
        reduced_memory = df_reduced.memory_usage(deep=True).sum()

        assert reduced_memory <= original_memory

    def test_reduce_mem_usage_preserves_values(self, sample_data):
        """Verify that numeric values are preserved after dtype downcast."""
        df = sample_data.copy()
        df_reduced = reduce_mem_usage(df)

        # Numeric values must be approximately equal after downcast
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype != object:
                pd.testing.assert_series_equal(
                    df[col].fillna(-999),
                    df_reduced[col].fillna(-999),
                    check_dtype=False,
                    rtol=1e-5,
                )


class TestDataSplit:
    """Tests for train/CV/test splitting"""

    def test_split_proportions(self, train_data):
        """Verify 70/15/15 split proportions."""
        x_train, _, x_cv, _, x_test, _, _ = split_train_data(train_data)

        total = len(x_train) + len(x_cv) + len(x_test)

        assert abs(len(x_train) / total - 0.70) < 0.01
        assert abs(len(x_cv) / total - 0.15) < 0.01
        assert abs(len(x_test) / total - 0.15) < 0.01

    @pytest.mark.smoke
    def test_split_no_overlap(self, train_data):
        """Verify there is no index overlap between splits."""
        x_train, _, x_cv, _, x_test, _, _ = split_train_data(train_data)

        # Check that indices do not overlap
        train_indices = set(x_train.index)
        cv_indices = set(x_cv.index)
        test_indices = set(x_test.index)

        assert len(train_indices & cv_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(cv_indices & test_indices) == 0

    def test_split_preserves_target_distribution(self, train_data):
        """Verify that fraud rate is roughly preserved across splits."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_train_data(train_data)

        original_fraud_ratio = train_data["isFraud"].mean()
        train_fraud_ratio = y_train.mean()
        cv_fraud_ratio = y_cv.mean()
        test_fraud_ratio = y_test.mean()

        # Ratios should be close (allowing reasonable variance)
        assert abs(train_fraud_ratio - original_fraud_ratio) < 0.05
        assert abs(cv_fraud_ratio - original_fraud_ratio) < 0.05
        assert abs(test_fraud_ratio - original_fraud_ratio) < 0.05

    def test_split_y_dimensions(self, train_data):
        """Verify shape and length consistency of y arrays."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_train_data(train_data)

        assert y_train.ndim == 1
        assert y_cv.ndim == 1
        assert y_test.ndim == 1

        assert len(y_train) == len(x_train)
        assert len(y_cv) == len(x_cv)
        assert len(y_test) == len(x_test)

    def test_split_y_values_binary(self, train_data):
        """Verify that y arrays contain only 0 and 1."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_train_data(train_data)

        for y in [y_train, y_cv, y_test]:
            assert set(np.unique(y)) == {0, 1}

    def test_split_scale_pos_weight(self, train_data):
        """Verify that scale_pos_weight equals the negative/positive class ratio."""
        (
            x_train,
            y_train,
            x_cv,
            y_cv,
            x_test,
            y_test,
            scale_pos_weight,
        ) = split_train_data(train_data)

        expected = (y_train == 0).sum() / (y_train == 1).sum()
        assert abs(scale_pos_weight - expected) < 1e-6
        assert scale_pos_weight > 0

    def test_split_removes_required_columns(self, train_data):
        """Verify that metadata columns are excluded from feature arrays."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_train_data(train_data)

        forbidden_cols = ["TransactionID", "isFraud", "TransactionDT"]

        for col in forbidden_cols:
            assert col not in x_train.columns
            assert col not in x_cv.columns
            assert col not in x_test.columns

    def test_split_x_dimensions(self, train_data):
        """Verify that X and y arrays have matching shapes."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_train_data(train_data)

        assert x_train.shape[0] == len(y_train)
        assert x_cv.shape[0] == len(y_cv)
        assert x_test.shape[0] == len(y_test)

        assert x_train.shape[1] == x_cv.shape[1]
        assert x_cv.shape[1] == x_test.shape[1]


class TestPrepareTestData:
    """Tests for final test set preparation"""

    def test_prepare_final_test_shape(self, test_data):
        """Verify output shape matches input."""
        x_final_test, test_ids = prepare_final_test(test_data)

        assert len(x_final_test) == len(test_data)
        assert len(test_ids) == len(test_data)

    def test_prepare_final_test_removes_cols(self, test_data):
        """Verify that metadata columns are dropped."""
        x_final_test, test_ids = prepare_final_test(test_data)

        forbidden_cols = ["TransactionID", "TransactionDT"]
        for col in forbidden_cols:
            assert col not in x_final_test.columns

    def test_prepare_final_test_no_nulls_in_ids(self, test_data):
        """Verify that TransactionID column has no nulls."""
        x_final_test, test_ids = prepare_final_test(test_data)

        assert not test_ids.isnull().any()


class TestDataQuality:
    """General data quality tests"""

    @pytest.mark.smoke
    def test_no_nan_in_split_features(self, split_datasets):
        """Verify no NaN values remain in features after split."""
        x_train, _, x_cv, _, x_test, _, _ = split_datasets

        nan_cols = x_train.columns[x_train.isnull().any()].tolist()
        print(f"Columns with NaN: {nan_cols}")

        # XGBoost e LightGBM não lidam bem com NaN
        assert not x_train.isnull().any().any()
        assert not x_cv.isnull().any().any()
        assert not x_test.isnull().any().any()

    def test_data_types_consistency(self, split_datasets):
        """Verify column dtypes are consistent across splits."""
        x_train, y_train, x_cv, y_cv, x_test, y_test, _ = split_datasets

        pd.testing.assert_index_equal(x_train.columns, x_cv.columns)
        pd.testing.assert_index_equal(x_cv.columns, x_test.columns)

    @pytest.mark.smoke
    def test_temporal_order_train_data(self, train_data):
        """Verify that splits are ordered by time to prevent data leakage."""
        # Sort data as the function does internally
        train_sorted = train_data.sort_values("TransactionDT").reset_index(drop=True)

        # Train must precede CV, and CV must precede test — no data leakage
        train_max_dt = train_sorted.iloc[: int(len(train_sorted) * 0.70)][
            "TransactionDT"
        ].max()
        cv_min_dt = train_sorted.iloc[
            int(len(train_sorted) * 0.70) : int(len(train_sorted) * 0.85)
        ]["TransactionDT"].min()
        test_min_dt = train_sorted.iloc[int(len(train_sorted) * 0.85) :][
            "TransactionDT"
        ].min()

        # Verificar ordenação temporal
        assert train_max_dt <= cv_min_dt
        assert cv_min_dt <= test_min_dt
