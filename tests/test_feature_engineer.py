import pytest
import numpy as np
import pandas as pd
from src.feature_engineer import (
    fit_transform,
    transform,
    CAT_COLS,
    FREQ_COLS,
)

# =============================================================================
# Testes do fit_transform
# =============================================================================


class TestFitTransform:
    def test_returns_three_values(self, train_data, test_data):
        """fit_transform deve devolver (train, test, artifacts)."""
        result = fit_transform(train_data.copy(), test_data.copy())
        assert len(result) == 3

    def test_artifacts_has_required_keys(self, fitted_data):
        """artifacts deve conter todas as chaves necessárias para a API."""
        _, _, artifacts = fitted_data
        for key in ["medians", "enc", "uid1_agg", "freq_maps", "dt_max"]:
            assert key in artifacts, f"Chave '{key}' em falta nos artifacts"

    def test_shape_preserved(self, train_data, test_data):
        """fit_transform não deve alterar o número de linhas."""
        n_train = len(train_data)
        n_test = len(test_data)
        train_out, test_out, _ = fit_transform(train_data.copy(), test_data.copy())
        assert len(train_out) == n_train
        assert len(test_out) == n_test

    def test_no_missing_values_after_fit(self, fitted_data):
        """Não devem existir NaN após fit_transform."""
        train, test, _ = fitted_data
        assert not train.isnull().any().any(), "Train tem NaN após fit_transform"
        assert not test.isnull().any().any(), "Test tem NaN após fit_transform"

    def test_cat_cols_are_numeric(self, fitted_data):
        """Colunas categóricas devem ser numéricas após encoding."""
        train, _, _ = fitted_data
        for col in CAT_COLS:
            if col in train.columns:
                assert pd.api.types.is_numeric_dtype(
                    train[col]
                ), f"Coluna '{col}' não é numérica após encoding"

    def test_uid_columns_created(self, fitted_data):
        """uid1, uid2, uid3 devem ser criados em train e test."""
        train, test, _ = fitted_data
        for uid in ["uid1", "uid2", "uid3"]:
            assert uid in train.columns, f"'{uid}' em falta no train"
            assert uid in test.columns, f"'{uid}' em falta no test"

    def test_time_features_created(self, fitted_data):
        """Features temporais devem ser criadas."""
        train, _, _ = fitted_data
        for col in ["day", "day_of_week", "hour", "dt_normalized"]:
            assert col in train.columns, f"'{col}' em falta no train"

    def test_amt_features_created(self, fitted_data):
        """Features de amount e email devem ser criadas."""
        train, _, _ = fitted_data
        for col in [
            "amt_log",
            "amt_is_round",
            "amt_decimal",
            "same_email",
            "P_email_suffix",
        ]:
            assert col in train.columns, f"'{col}' em falta no train"

    def test_amt_log_non_negative(self, fitted_data):
        """amt_log não deve ter valores negativos."""
        train, _, _ = fitted_data
        assert (train["amt_log"] >= 0).all()

    def test_amt_is_round_binary(self, fitted_data):
        """amt_is_round deve ser binário."""
        train, _, _ = fitted_data
        assert set(train["amt_is_round"].unique()).issubset({0, 1})

    def test_same_email_binary(self, fitted_data):
        """same_email deve ser binário."""
        train, _, _ = fitted_data
        assert set(train["same_email"].unique()).issubset({0, 1})

    def test_uid1_agg_columns_created(self, fitted_data):
        """Colunas de agregação uid1 devem ser criadas."""
        train, _, _ = fitted_data
        agg_cols = [c for c in train.columns if c.startswith("uid1_")]
        assert len(agg_cols) > 0

    def test_no_data_leakage(self, train_data, test_data):
        """dt_max deve ser calculado apenas no train."""
        _, _, artifacts = fit_transform(train_data.copy(), test_data.copy())
        assert artifacts["dt_max"] == train_data["TransactionDT"].max()

    def test_freq_maps_values_between_0_and_1(self, fitted_data):
        """Frequency maps devem ter valores entre 0 e 1."""
        _, _, artifacts = fitted_data
        for col, freq in artifacts["freq_maps"].items():
            assert (freq >= 0).all() and (
                freq <= 1
            ).all(), f"freq_maps['{col}'] tem valores fora de [0, 1]"


# =============================================================================
# Testes do transform (produção / API)
# =============================================================================


class TestTransform:
    def test_transform_single_row(self, sample_data, artifacts):
        """transform deve funcionar com uma única transação."""
        single = sample_data.iloc[[0]].drop(columns=["TransactionID", "isFraud"]).copy()
        result = transform(single, artifacts)
        assert len(result) == 1

    def test_transform_no_missing(self, sample_data, artifacts):
        """transform não deve deixar NaN no resultado."""
        single = sample_data.iloc[[0]].drop(columns=["TransactionID", "isFraud"]).copy()
        result = transform(single, artifacts)
        assert not result.isnull().any().any()

    def test_transform_minimal_input(self, artifacts):
        """transform deve funcionar com apenas os campos obrigatórios."""
        minimal = pd.DataFrame(
            [
                {
                    "TransactionDT": 86400,
                    "TransactionAmt": 150.0,
                    "card1": 9500.0,
                    "addr1": 315.0,
                    "card2": 111.0,
                    "P_emaildomain": "gmail.com",
                    "R_emaildomain": "gmail.com",
                }
            ]
        )
        result = transform(minimal, artifacts)
        assert len(result) == 1

    def test_transform_missing_cat_cols_filled(self, artifacts):
        """Colunas categóricas em falta devem ser preenchidas com 'unknown'."""
        minimal = pd.DataFrame(
            [
                {
                    "TransactionDT": 86400,
                    "TransactionAmt": 150.0,
                    "card1": 9500.0,
                    "addr1": 315.0,
                    "card2": 111.0,
                    "P_emaildomain": "gmail.com",
                    "R_emaildomain": "gmail.com",
                }
            ]
        )
        # Não deve crashar mesmo sem as colunas categóricas
        result = transform(minimal, artifacts)
        assert len(result) == 1

    def test_transform_unknown_user_gets_fallback(self, artifacts):
        """Utilizador novo deve receber -999 nas colunas de agregação."""
        new_user = pd.DataFrame(
            [
                {
                    "TransactionDT": 86400,
                    "TransactionAmt": 999.0,
                    "card1": 999999.0,  # nunca visto no treino
                    "addr1": 1.0,
                    "card2": 1.0,
                    "P_emaildomain": "new-unknown.com",
                    "R_emaildomain": "new-unknown.com",
                }
            ]
        )
        result = transform(new_user, artifacts)
        assert result["uid1_TransactionAmt_mean"].iloc[0] == -999

    def test_transform_same_features_as_fit(self, fitted_data, sample_data, artifacts):
        """transform deve produzir as mesmas colunas base que o fit_transform."""
        train, _, _ = fitted_data
        single = sample_data.iloc[[0]].drop(columns=["TransactionID", "isFraud"]).copy()
        result = transform(single, artifacts)

        fit_cols = set(train.columns) - {"isFraud", "TransactionID"}
        trans_cols = set(result.columns)
        missing = fit_cols - trans_cols
        assert len(missing) == 0, f"Colunas em falta no transform: {missing}"


# =============================================================================
# Testes de integração
# =============================================================================


class TestIntegration:
    def test_full_pipeline_does_not_crash(self, train_data, test_data, sample_data):
        """Pipeline completo fit_transform → transform não deve crashar."""
        train, test, artifacts = fit_transform(train_data.copy(), test_data.copy())
        single = sample_data.iloc[[0]].drop(columns=["TransactionID", "isFraud"]).copy()
        transform(single, artifacts)

    def test_transform_consistent_with_fit(self, train_data, test_data, sample_data):
        """
        amt_log de uma linha transformada via transform() deve ser igual
        ao mesmo valor produzido pelo fit_transform().
        """
        train_out, _, artifacts = fit_transform(train_data.copy(), test_data.copy())

        single = sample_data.iloc[[0]].drop(columns=["TransactionID", "isFraud"]).copy()
        result = transform(single, artifacts)

        expected = train_out.iloc[0]["amt_log"]
        actual = result.iloc[0]["amt_log"]
        assert (
            abs(expected - actual) < 1e-5
        ), f"amt_log difere: fit={expected:.5f} transform={actual:.5f}"
