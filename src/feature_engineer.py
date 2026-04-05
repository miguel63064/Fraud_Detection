import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


CAT_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "DeviceType",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "id_33",
    "id_34",
]

FREQ_COLS = [
    "P_emaildomain",
    "P_email_suffix",
    "DeviceInfo",
    "uid1",
    "uid2",
    "uid3",
    "R_emaildomain",
]


# =============================================================================
# Internal helpers — operate on a single DataFrame (train or test)
# =============================================================================


def _combine_features(df):
    new_cols = {
        "uid1": df["card1"].astype(str) + "_" + df["addr1"].astype(str),
        "uid2": df["card1"].astype(str) + "_" + df["card2"].astype(str),
        "uid3": df["card1"].astype(str) + "_" + df["P_emaildomain"].astype(str),
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _time_features(df):
    day = df["TransactionDT"] // (3600 * 24)
    new_cols = {
        "day": day,
        "day_of_week": day % 7,
        "hour": (df["TransactionDT"] // 3600) % 24,
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _amt_email_features(df):
    new_cols = {
        "amt_log": np.log1p(df["TransactionAmt"]),
        "amt_is_round": (df["TransactionAmt"] % 1 == 0).astype(int),
        "amt_vs_uid_mean": df["TransactionAmt"] / (df["uid1_TransactionAmt_mean"] + 1),
        "amt_decimal": df["TransactionAmt"] - np.floor(df["TransactionAmt"]),
        "P_email_suffix": df["P_emaildomain"].str.split(".").str[-1],
        "same_email": (df["P_emaildomain"] == df["R_emaildomain"]).astype(int),
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# =============================================================================
# FIT_TRANSFORM — called from main.py during training
# Learns parameters from train, applies them to train + test.
# Returns artifacts dict for saving to MLflow.
# =============================================================================


def fit_transform(train, test):
    """
    Called from main.py during training.

    Returns (train, test, artifacts) where artifacts contains everything
    the API needs to preprocess new transactions at inference time.
    """
    artifacts = {}

    # 1. Missing values — learn medians from training data
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [
        c for c in num_cols if c not in ["TransactionID", "isFraud", "TransactionDT"]
    ]
    medians = {col: train[col].median() for col in num_cols}
    artifacts["medians"] = medians

    for col, median_val in medians.items():
        train[col] = train[col].fillna(median_val)
        if col in test.columns:
            test[col] = test[col].fillna(median_val)

    cat_cols_obj = train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols_obj:
        train[col] = train[col].fillna("unknown")
        if col in test.columns:
            test[col] = test[col].fillna("unknown")

    # 2. Combine features (stateless — only creates new columns)
    train = _combine_features(train)
    test = _combine_features(test)

    # 3. Categorical encoding — fit on train, transform both
    cols_present = [c for c in CAT_COLS if c in train.columns]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train[cols_present] = enc.fit_transform(train[cols_present])
    test[cols_present] = enc.transform(test[cols_present])
    artifacts["enc"] = enc

    # 4. Aggregate user stats — fit on train, map to both
    agg_dict = {
        "TransactionAmt": ["mean", "std", "max", "min"],
        "D1": ["mean", "std"],
        "C1": ["mean", "sum"],
    }
    uid1_agg = train.groupby("uid1").agg(agg_dict)
    uid1_agg.columns = ["uid1_" + "_".join(c) for c in uid1_agg.columns]
    artifacts["uid1_agg"] = uid1_agg

    uid1_train = {
        col: train["uid1"].map(uid1_agg[col]).fillna(-999) for col in uid1_agg.columns
    }
    uid1_test = {
        col: test["uid1"].map(uid1_agg[col]).fillna(-999) for col in uid1_agg.columns
    }
    train = pd.concat([train, pd.DataFrame(uid1_train, index=train.index)], axis=1)
    test = pd.concat([test, pd.DataFrame(uid1_test, index=test.index)], axis=1)

    # 5. Time features
    dt_max = train["TransactionDT"].max()
    artifacts["dt_max"] = dt_max
    train = _time_features(train)
    test = _time_features(test)
    train = pd.concat(
        [
            train,
            pd.DataFrame(
                {"dt_normalized": train["TransactionDT"] / dt_max}, index=train.index
            ),
        ],
        axis=1,
    )
    test = pd.concat(
        [
            test,
            pd.DataFrame(
                {"dt_normalized": test["TransactionDT"] / dt_max}, index=test.index
            ),
        ],
        axis=1,
    )

    # 6. Amount + email features
    train = _amt_email_features(train)
    test = _amt_email_features(test)

    # 7. Frequency encoding — fit on train, map to both
    freq_maps = {}
    for col in FREQ_COLS:
        if col not in train.columns:
            continue
        freq = train[col].value_counts(normalize=True)
        freq_maps[col] = freq
        train[col] = train[col].map(freq).fillna(0)
        if col in test.columns:
            test[col] = test[col].map(freq).fillna(0)
    artifacts["freq_maps"] = freq_maps

    return train, test, artifacts


# =============================================================================
# TRANSFORM — called by the API at inference time
# Receives a single-row DataFrame and the training artifacts dict.
# =============================================================================


def transform(df, artifacts):
    """
    Called by the API to preprocess a new transaction.

    Args:
        df:        Single-row DataFrame representing the incoming transaction.
        artifacts: Dict returned by fit_transform and persisted in MLflow.
    """
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "unknown"

    # 1. Missing values — apply training medians
    medians = artifacts.get("medians", {})
    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)

    cat_cols_obj = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols_obj:
        df[col] = df[col].fillna("unknown")

    # 2. Combine features
    df = _combine_features(df)

    # 3. Categorical encoding — apply training encoder
    enc = artifacts.get("enc")
    cols_present = [c for c in CAT_COLS if c in df.columns]
    if enc is not None:
        df[cols_present] = enc.transform(df[cols_present])

    # 4. Aggregate user stats — map from training aggregation table
    uid1_agg = artifacts.get("uid1_agg")
    if uid1_agg is not None:
        uid1_cols = {
            col: df["uid1"].map(uid1_agg[col]).fillna(-999) for col in uid1_agg.columns
        }
        df = pd.concat([df, pd.DataFrame(uid1_cols, index=df.index)], axis=1)
    else:
        fallback = {
            col: -999
            for col in [
                "uid1_TransactionAmt_mean",
                "uid1_TransactionAmt_std",
                "uid1_TransactionAmt_max",
                "uid1_TransactionAmt_min",
                "uid1_D1_mean",
                "uid1_D1_std",
                "uid1_C1_mean",
                "uid1_C1_sum",
            ]
        }
        df = pd.concat([df, pd.DataFrame(fallback, index=df.index)], axis=1)

    # 5. Time features
    dt_max = artifacts.get("dt_max", 1)
    df = _time_features(df)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {"dt_normalized": df["TransactionDT"] / dt_max}, index=df.index
            ),
        ],
        axis=1,
    )

    # 6. Amount + email features
    df = _amt_email_features(df)

    # 7. Frequency encoding — apply training frequency maps
    freq_maps = artifacts.get("freq_maps", {})
    for col in FREQ_COLS:
        if col not in df.columns:
            continue
        fmap = freq_maps.get(col)
        df[col] = df[col].map(fmap).fillna(0) if fmap is not None else 0

    return df
