"""
Tests for scripts/simulate_drift.py

The script has no external runtime deps beyond stdlib, so all HTTP calls are
mocked with unittest.mock.  The batch-generator functions are tested directly.
"""

import importlib.util
import json
import sys
import unittest.mock
from io import BytesIO
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the module from scripts/ (not on sys.path by default)
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parent.parent / "scripts" / "simulate_drift.py"
_spec = importlib.util.spec_from_file_location("simulate_drift", _SCRIPT)
simulate_drift = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(simulate_drift)


# ---------------------------------------------------------------------------
# _base_transaction
# ---------------------------------------------------------------------------


class TestBaseTransaction:
    def test_returns_dict(self):
        t = simulate_drift._base_transaction(0)
        assert isinstance(t, dict)

    def test_required_keys_present(self):
        t = simulate_drift._base_transaction(42)
        required = {
            "TransactionDT",
            "TransactionAmt",
            "ProductCD",
            "card1",
            "card2",
            "card4",
            "card6",
            "addr1",
            "P_emaildomain",
            "R_emaildomain",
            "C1",
            "D1",
            "M1",
            "M2",
            "M3",
            "DeviceType",
        }
        assert required.issubset(t.keys())

    def test_deterministic_with_same_seed(self):
        assert simulate_drift._base_transaction(7) == simulate_drift._base_transaction(
            7
        )

    def test_different_seeds_produce_different_results(self):
        assert simulate_drift._base_transaction(0) != simulate_drift._base_transaction(
            1
        )

    def test_transaction_amount_in_range(self):
        for seed in range(20):
            t = simulate_drift._base_transaction(seed)
            assert 10 <= t["TransactionAmt"] <= 500

    def test_valid_card4_value(self):
        valid = {"visa", "mastercard", "discover", "american express"}
        for seed in range(20):
            assert simulate_drift._base_transaction(seed)["card4"] in valid


# ---------------------------------------------------------------------------
# make_no_drift_batch
# ---------------------------------------------------------------------------


class TestNoDriftBatch:
    def test_default_length(self):
        assert len(simulate_drift.make_no_drift_batch()) == 50

    def test_custom_length(self):
        assert len(simulate_drift.make_no_drift_batch(10)) == 10

    def test_each_element_is_dict(self):
        batch = simulate_drift.make_no_drift_batch(5)
        assert all(isinstance(t, dict) for t in batch)

    def test_amounts_in_baseline_range(self):
        batch = simulate_drift.make_no_drift_batch(50)
        for t in batch:
            assert 10 <= t["TransactionAmt"] <= 500


# ---------------------------------------------------------------------------
# make_moderate_drift_batch
# ---------------------------------------------------------------------------


class TestModerateDriftBatch:
    def test_default_length(self):
        assert len(simulate_drift.make_moderate_drift_batch()) == 50

    def test_amounts_shifted_upward(self):
        batch = simulate_drift.make_moderate_drift_batch(50)
        for t in batch:
            assert 800 <= t["TransactionAmt"] <= 2_000

    def test_other_fields_still_present(self):
        batch = simulate_drift.make_moderate_drift_batch(5)
        for t in batch:
            assert "card4" in t and "DeviceType" in t


# ---------------------------------------------------------------------------
# make_severe_drift_batch
# ---------------------------------------------------------------------------


class TestSevereDriftBatch:
    def test_default_length(self):
        assert len(simulate_drift.make_severe_drift_batch()) == 50

    def test_amounts_extreme(self):
        batch = simulate_drift.make_severe_drift_batch(50)
        for t in batch:
            assert 5_000 <= t["TransactionAmt"] <= 50_000

    def test_unseen_card_type(self):
        batch = simulate_drift.make_severe_drift_batch(10)
        assert all(t["card4"] == "unknown_processor" for t in batch)

    def test_unseen_email_domain(self):
        batch = simulate_drift.make_severe_drift_batch(10)
        assert all(t["P_emaildomain"] == "suspicious-domain.ru" for t in batch)

    def test_d1_out_of_training_range(self):
        batch = simulate_drift.make_severe_drift_batch(20)
        for t in batch:
            assert 3_000 <= t["D1"] <= 9_999


# ---------------------------------------------------------------------------
# post_json  (stdlib HTTP — mocked)
# ---------------------------------------------------------------------------


class TestPostJson:
    def _mock_urlopen(self, response_body: dict):
        """Return a context-manager mock that yields a fake HTTP response."""
        encoded = json.dumps(response_body).encode()
        mock_resp = unittest.mock.MagicMock()
        mock_resp.read.return_value = encoded
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = unittest.mock.MagicMock(return_value=False)
        return mock_resp

    def test_returns_parsed_json(self):
        expected = {"drift_detected": False, "n_samples": 50}
        mock_resp = self._mock_urlopen(expected)
        with unittest.mock.patch("urllib.request.urlopen", return_value=mock_resp):
            result = simulate_drift.post_json(
                "http://localhost:8000/monitor/drift",
                {"transactions": []},
                "fraud",
            )
        assert result == expected

    def test_sends_correct_headers(self):
        mock_resp = self._mock_urlopen({})
        with unittest.mock.patch(
            "urllib.request.urlopen", return_value=mock_resp
        ) as mock_open:
            simulate_drift.post_json("http://localhost/drift", {}, "my-key")
            req = mock_open.call_args[0][0]
            assert req.get_header("Content-type") == "application/json"
            assert req.get_header("X-api-key") == "my-key"

    def test_sends_post_method(self):
        mock_resp = self._mock_urlopen({})
        with unittest.mock.patch(
            "urllib.request.urlopen", return_value=mock_resp
        ) as mock_open:
            simulate_drift.post_json("http://localhost/drift", {"k": "v"}, "key")
            req = mock_open.call_args[0][0]
            assert req.get_method() == "POST"


# ---------------------------------------------------------------------------
# run_scenario  (mocks post_json, captures stdout)
# ---------------------------------------------------------------------------


class TestRunScenario:
    def _patch_post(self, response: dict):
        return unittest.mock.patch.object(
            simulate_drift, "post_json", return_value=response
        )

    def test_no_drift_output(self, capsys):
        resp = {
            "drift_detected": False,
            "n_samples": 50,
            "drifted_features": [],
            "numerical": {},
            "categorical": {},
        }
        with self._patch_post(resp):
            simulate_drift.run_scenario(
                "no_drift", [{}] * 50, "http://localhost:8000", "fraud"
            )
        out = capsys.readouterr().out
        assert "No drift" in out
        assert "50" in out

    def test_drift_detected_output(self, capsys):
        resp = {
            "drift_detected": True,
            "n_samples": 50,
            "drifted_features": ["TransactionAmt"],
            "numerical": {
                "TransactionAmt": {
                    "drift": True,
                    "ks_statistic": 0.8,
                    "p_value": 0.0001,
                    "new_mean": 9000.0,
                    "ref_mean": 200.0,
                }
            },
            "categorical": {},
        }
        with self._patch_post(resp):
            simulate_drift.run_scenario(
                "severe", [{}] * 50, "http://localhost:8000", "fraud"
            )
        out = capsys.readouterr().out
        assert "DRIFT DETECTED" in out
        assert "TransactionAmt" in out

    def test_http_error_does_not_crash(self, capsys):
        import urllib.error

        with unittest.mock.patch.object(
            simulate_drift,
            "post_json",
            side_effect=urllib.error.HTTPError(
                "url", 422, "Unprocessable", {}, BytesIO(b"err")
            ),
        ):
            simulate_drift.run_scenario("bad", [], "http://localhost:8000", "fraud")
        out = capsys.readouterr().out
        assert "422" in out

    def test_generic_exception_does_not_crash(self, capsys):
        with unittest.mock.patch.object(
            simulate_drift,
            "post_json",
            side_effect=ConnectionRefusedError("refused"),
        ):
            simulate_drift.run_scenario("bad", [], "http://localhost:8000", "fraud")
        out = capsys.readouterr().out
        assert "Request failed" in out
