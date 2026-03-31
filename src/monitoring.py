"""
Drift detection utilities for the fraud detection model.

Two statistical tests are used:
- Kolmogorov-Smirnov (KS) test for numerical features
- Chi-squared test for categorical features

Reference statistics are computed once during training (main.py) and
persisted as ``train_artifacts/reference_stats.json`` in the MLflow run.
The API loads these stats at startup and exposes a ``POST /monitor/drift``
endpoint that compares incoming batch data against the reference.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Any


# ---------------------------------------------------------------------------
# Reference stats — computed once at training time
# ---------------------------------------------------------------------------


def compute_reference_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute summary statistics from the raw training DataFrame.

    Called in main.py immediately after load_data(), before fit_transform(),
    so the reference captures the true raw distributions.

    Returns a JSON-serialisable dict with:
    - ``numerical``:   {col: {mean, std, min, max, p25, p50, p75, values_sample}}
    - ``categorical``: {col: {frequencies}}
    """
    result: dict[str, Any] = {"numerical": {}, "categorical": {}}

    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        if pd.api.types.is_numeric_dtype(series):
            # Store up to 5 000 random values for the KS test at inference time
            sample = series.sample(min(len(series), 5_000), random_state=42).tolist()
            result["numerical"][col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.median()),
                "p75": float(series.quantile(0.75)),
                "values_sample": sample,
            }
        else:
            freq = series.value_counts(normalize=True).to_dict()
            result["categorical"][col] = {
                "frequencies": {str(k): float(v) for k, v in freq.items()}
            }

    return result


# ---------------------------------------------------------------------------
# Drift detection — called at inference time by the API
# ---------------------------------------------------------------------------


def detect_drift(
    batch: pd.DataFrame,
    reference_stats: dict[str, Any],
    ks_threshold: float = 0.05,
    chi2_threshold: float = 0.05,
) -> dict[str, Any]:
    """
    Compare ``batch`` against ``reference_stats`` and return a drift report.

    Args:
        batch:            DataFrame of incoming transactions (1-N rows).
        reference_stats:  Dict produced by ``compute_reference_stats``.
        ks_threshold:     p-value threshold for the KS test (default 0.05).
        chi2_threshold:   p-value threshold for the chi-squared test (default 0.05).

    Returns:
        drift_detected   – True if any feature shows significant drift.
        n_samples        – Number of rows in the batch.
        drifted_features – List of feature names that triggered an alert.
        numerical        – Per-column KS test results.
        categorical      – Per-column chi-squared test results.
    """
    numerical_results: dict[str, Any] = {}
    categorical_results: dict[str, Any] = {}
    drifted: list[str] = []

    num_ref = reference_stats.get("numerical", {})
    cat_ref = reference_stats.get("categorical", {})

    # --- Numerical features: two-sample KS test ---
    for col, ref in num_ref.items():
        if col not in batch.columns:
            continue
        new_vals = batch[col].dropna().values
        if len(new_vals) < 2:
            continue

        ref_sample = np.array(ref["values_sample"])
        ks_stat, p_value = stats.ks_2samp(ref_sample, new_vals)
        drifted_flag = bool(p_value < ks_threshold)

        numerical_results[col] = {
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "drift": drifted_flag,
            "new_mean": round(float(new_vals.mean()), 4),
            "ref_mean": round(ref["mean"], 4),
        }
        if drifted_flag:
            drifted.append(col)

    # --- Categorical features: chi-squared test ---
    for col, ref in cat_ref.items():
        if col not in batch.columns:
            continue
        new_counts = batch[col].fillna("unknown").value_counts()
        ref_freq = ref["frequencies"]

        # Only consider categories present in the reference distribution
        categories = list(ref_freq.keys())
        observed = np.array([new_counts.get(c, 0) for c in categories], dtype=float)
        expected_freq = np.array([ref_freq[c] for c in categories], dtype=float)

        total_observed = observed.sum()
        if total_observed == 0:
            continue

        expected = expected_freq * total_observed

        # Chi-squared requires expected counts >= 1; drop bins below that
        mask = expected >= 1
        observed_f = observed[mask]
        expected_f = expected[mask]

        if len(observed_f) < 2:
            continue

        # Rescale expected so its sum matches observed (required by scipy after bin filtering)
        expected_f = expected_f * (observed_f.sum() / expected_f.sum())

        chi2_stat, p_value = stats.chisquare(f_obs=observed_f, f_exp=expected_f)
        drifted_flag = bool(p_value < chi2_threshold)

        categorical_results[col] = {
            "chi2_statistic": round(float(chi2_stat), 6),
            "p_value": round(float(p_value), 6),
            "drift": drifted_flag,
            "new_top_category": str(new_counts.idxmax())
            if len(new_counts) > 0
            else None,
        }
        if drifted_flag:
            drifted.append(col)

    return {
        "drift_detected": len(drifted) > 0,
        "n_samples": len(batch),
        "drifted_features": drifted,
        "numerical": numerical_results,
        "categorical": categorical_results,
    }
