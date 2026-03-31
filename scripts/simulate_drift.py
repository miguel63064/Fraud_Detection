"""
Drift simulation script for the Fraud Detection API.

Sends three batches to POST /monitor/drift with increasing levels of
synthetic distribution shift, so you can verify the drift detector works.

Usage
-----
Start the API first (the model must be trained so reference_stats.json exists):

    python -m uvicorn app.api:app --reload

Then run this script:

    python scripts/simulate_drift.py
    python scripts/simulate_drift.py --url http://localhost:8000 --api-key fraud

Scenarios
---------
no_drift        Normal amounts / common email domains — should NOT trigger alerts.
moderate_drift  Transaction amounts 2x higher than baseline — may trigger some features.
severe_drift    Amounts 10x higher + unseen card type + unseen domains — should alert.
"""

import argparse
import json
import random
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Transaction generators
# ---------------------------------------------------------------------------

def _base_transaction(seed: int) -> dict:
    random.seed(seed)
    return {
        "TransactionDT": float(random.randint(86_400, 15_000_000)),
        "TransactionAmt": round(random.uniform(10, 500), 2),
        "ProductCD": random.choice(["W", "H", "C", "S", "R"]),
        "card1": float(random.randint(1_000, 18_000)),
        "card2": float(random.randint(100, 600)),
        "card4": random.choice(["visa", "mastercard", "discover", "american express"]),
        "card6": random.choice(["debit", "credit"]),
        "addr1": float(random.randint(100, 500)),
        "P_emaildomain": random.choice(
            ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]
        ),
        "R_emaildomain": random.choice(
            ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]
        ),
        "C1": round(random.uniform(0, 5), 1),
        "D1": float(random.randint(0, 365)),
        "M1": random.choice(["T", "F"]),
        "M2": random.choice(["T", "F"]),
        "M3": random.choice(["T", "F"]),
        "DeviceType": random.choice(["desktop", "mobile"]),
    }


def make_no_drift_batch(n: int = 50) -> list[dict]:
    """Normal transactions matching the training distribution."""
    return [_base_transaction(i) for i in range(n)]


def make_moderate_drift_batch(n: int = 50) -> list[dict]:
    """Transaction amounts shifted ~2x above the training baseline."""
    batch = []
    for i in range(n):
        t = _base_transaction(i + 1_000)
        t["TransactionAmt"] = round(random.uniform(800, 2_000), 2)
        batch.append(t)
    return batch


def make_severe_drift_batch(n: int = 50) -> list[dict]:
    """Extreme amounts + unseen card type + unseen email domains."""
    batch = []
    for i in range(n):
        t = _base_transaction(i + 2_000)
        t["TransactionAmt"] = round(random.uniform(5_000, 50_000), 2)
        t["card4"] = "unknown_processor"
        t["P_emaildomain"] = "suspicious-domain.ru"
        t["D1"] = float(random.randint(3_000, 9_999))
        batch.append(t)
    return batch


# ---------------------------------------------------------------------------
# HTTP helper (stdlib-only, no extra deps)
# ---------------------------------------------------------------------------

def post_json(url: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(name: str, transactions: list[dict], base_url: str, api_key: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Scenario : {name}  ({len(transactions)} transactions)")
    print("=" * 60)

    try:
        report = post_json(
            f"{base_url}/monitor/drift",
            {"transactions": transactions},
            api_key,
        )
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()}")
        return
    except Exception as exc:
        print(f"Request failed: {exc}")
        return

    label = "DRIFT DETECTED" if report.get("drift_detected") else "No drift"
    print(f"Result   : {label}")
    print(f"Samples  : {report.get('n_samples')}")
    drifted = report.get("drifted_features", [])
    print(f"Drifted  : {', '.join(drifted) if drifted else 'none'}")

    num = report.get("numerical", {})
    if any(v["drift"] for v in num.values()):
        print("\nNumerical (drifted only):")
        for col, info in num.items():
            if info["drift"]:
                print(
                    f"  {col:<32s}  KS={info['ks_statistic']:.4f}  "
                    f"p={info['p_value']:.4f}  "
                    f"new_mean={info['new_mean']:.2f}  ref_mean={info['ref_mean']:.2f}"
                )

    cat = report.get("categorical", {})
    if any(v["drift"] for v in cat.values()):
        print("\nCategorical (drifted only):")
        for col, info in cat.items():
            if info["drift"]:
                print(
                    f"  {col:<32s}  chi2={info['chi2_statistic']:.4f}  "
                    f"p={info['p_value']:.4f}  "
                    f"top={info.get('new_top_category')}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate data drift against the Fraud API.")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", default="fraud", help="Value of the X-API-Key header")
    args = parser.parse_args()

    scenarios = [
        ("no_drift",       make_no_drift_batch()),
        ("moderate_drift", make_moderate_drift_batch()),
        ("severe_drift",   make_severe_drift_batch()),
    ]

    for name, batch in scenarios:
        run_scenario(name, batch, args.url, args.api_key)

    print("\nDone.")


if __name__ == "__main__":
    main()
