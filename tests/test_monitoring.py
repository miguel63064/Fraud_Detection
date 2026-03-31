import numpy as np
import pandas as pd
import pytest

from src.monitoring import compute_reference_stats, detect_drift


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_df():
    """DataFrame com distribuições bem definidas para servir de referência."""
    np.random.seed(0)
    n = 1_000
    return pd.DataFrame(
        {
            "amount": np.random.normal(100, 10, n),  # numérica
            "age": np.random.normal(35, 5, n),  # numérica
            "card_type": np.random.choice(
                ["visa", "mastercard", "amex"], n
            ),  # categórica
            "device": np.random.choice(["mobile", "desktop"], n),  # categórica
        }
    )


@pytest.fixture
def reference_stats(reference_df):
    return compute_reference_stats(reference_df)


# ---------------------------------------------------------------------------
# compute_reference_stats
# ---------------------------------------------------------------------------


class TestComputeReferenceStats:
    def test_keys_present(self, reference_stats):
        assert "numerical" in reference_stats
        assert "categorical" in reference_stats

    def test_numerical_columns_detected(self, reference_stats):
        assert "amount" in reference_stats["numerical"]
        assert "age" in reference_stats["numerical"]

    def test_categorical_columns_detected(self, reference_stats):
        assert "card_type" in reference_stats["categorical"]
        assert "device" in reference_stats["categorical"]

    def test_numerical_fields(self, reference_stats):
        col = reference_stats["numerical"]["amount"]
        for field in (
            "mean",
            "std",
            "min",
            "max",
            "p25",
            "p50",
            "p75",
            "values_sample",
        ):
            assert field in col, f"Campo '{field}' ausente"

    def test_values_sample_not_empty(self, reference_stats):
        sample = reference_stats["numerical"]["amount"]["values_sample"]
        assert isinstance(sample, list)
        assert len(sample) > 0

    def test_values_sample_capped_at_5000(self):
        np.random.seed(0)
        big_df = pd.DataFrame({"x": np.random.normal(0, 1, 10_000)})
        stats = compute_reference_stats(big_df)
        assert len(stats["numerical"]["x"]["values_sample"]) <= 5_000

    def test_categorical_frequencies_sum_to_one(self, reference_stats):
        freqs = reference_stats["categorical"]["card_type"]["frequencies"]
        total = sum(freqs.values())
        assert abs(total - 1.0) < 1e-6

    def test_empty_column_skipped(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, 2.0]})
        stats = compute_reference_stats(df)
        assert "a" not in stats["numerical"]
        assert "b" in stats["numerical"]


# ---------------------------------------------------------------------------
# detect_drift — sem drift
# ---------------------------------------------------------------------------


class TestDetectDriftNoDrift:
    def test_no_drift_same_distribution(self, reference_stats):
        np.random.seed(1)
        batch = pd.DataFrame(
            {
                "amount": np.random.normal(100, 10, 500),
                "age": np.random.normal(35, 5, 500),
                "card_type": np.random.choice(["visa", "mastercard", "amex"], 500),
                "device": np.random.choice(["mobile", "desktop"], 500),
            }
        )
        report = detect_drift(batch, reference_stats)
        assert report["drift_detected"] is False
        assert report["drifted_features"] == []

    def test_n_samples_correct(self, reference_stats):
        batch = pd.DataFrame({"amount": np.random.normal(100, 10, 50)})
        report = detect_drift(batch, reference_stats)
        assert report["n_samples"] == 50

    def test_numerical_results_structure(self, reference_stats):
        batch = pd.DataFrame({"amount": np.random.normal(100, 10, 200)})
        report = detect_drift(batch, reference_stats)
        col = report["numerical"].get("amount", {})
        for field in ("ks_statistic", "p_value", "drift", "new_mean", "ref_mean"):
            assert field in col


# ---------------------------------------------------------------------------
# detect_drift — com drift
# ---------------------------------------------------------------------------


class TestDetectDriftWithDrift:
    def test_numerical_drift_detected(self, reference_stats):
        # Distribuição muito diferente da referência (média deslocada 10 sigma)
        batch = pd.DataFrame({"amount": np.random.normal(200, 10, 500)})
        report = detect_drift(batch, reference_stats, ks_threshold=0.05)
        assert "amount" in report["drifted_features"]
        assert report["drift_detected"] is True

    def test_categorical_drift_detected(self, reference_stats):
        # Só uma categoria (desequilíbrio extremo)
        batch = pd.DataFrame({"card_type": ["visa"] * 500})
        report = detect_drift(batch, reference_stats, chi2_threshold=0.05)
        assert "card_type" in report["drifted_features"]
        assert report["drift_detected"] is True

    def test_drifted_feature_flag_true(self, reference_stats):
        batch = pd.DataFrame({"amount": np.random.normal(200, 10, 500)})
        report = detect_drift(batch, reference_stats)
        assert report["numerical"]["amount"]["drift"] is True


# ---------------------------------------------------------------------------
# detect_drift — casos limite
# ---------------------------------------------------------------------------


class TestDetectDriftEdgeCases:
    def test_column_not_in_batch_is_skipped(self, reference_stats):
        # Batch sem nenhuma coluna do reference — não deve levantar excepção
        batch = pd.DataFrame({"unrelated_col": [1, 2, 3]})
        report = detect_drift(batch, reference_stats)
        assert report["numerical"] == {}
        assert report["categorical"] == {}

    def test_batch_with_single_row_skipped(self, reference_stats):
        # Menos de 2 valores numéricos — KS não é executado
        batch = pd.DataFrame({"amount": [105.0]})
        report = detect_drift(batch, reference_stats)
        assert "amount" not in report["numerical"]

    def test_empty_reference_stats(self):
        batch = pd.DataFrame({"amount": [1, 2, 3]})
        report = detect_drift(batch, {})
        assert report["drift_detected"] is False
        assert report["drifted_features"] == []

    def test_return_keys(self, reference_stats):
        batch = pd.DataFrame({"amount": np.random.normal(100, 10, 100)})
        report = detect_drift(batch, reference_stats)
        for key in (
            "drift_detected",
            "n_samples",
            "drifted_features",
            "numerical",
            "categorical",
        ):
            assert key in report
