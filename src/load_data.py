import pandas as pd
import numpy as np


def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
    return df


def load_data():
    # Carregar dados
    train_identity = pd.read_csv("data/raw/train_identity.csv")
    train_identity = reduce_mem_usage(train_identity)
    train_transaction = pd.read_csv("data/raw/train_transaction.csv")
    train_transaction = reduce_mem_usage(train_transaction)
    test_identity = pd.read_csv("data/raw/test_identity.csv")
    test_identity = reduce_mem_usage(test_identity)
    test_transaction = pd.read_csv("data/raw/test_transaction.csv")
    test_transaction = reduce_mem_usage(test_transaction)

    train = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")
    test = pd.merge(test_transaction, test_identity, on="TransactionID", how="left")
    test.columns = test.columns.str.replace("-", "_")

    del train_transaction, train_identity, test_transaction, test_identity

    return train.copy(), test.copy()


def split_train_data(train):
    train = train.sort_values("TransactionDT").reset_index(drop=True)

    n = len(train)

    train_set = train.iloc[: int(n * 0.70)]  # 70% — train
    cv_set = train.iloc[int(n * 0.70) : int(n * 0.85)]  # 15% - validation
    test_set = train.iloc[int(n * 0.85) :]  # 15% — test

    print(f"Train: {len(train_set):,}")
    print(f"CV:    {len(cv_set):,}")
    print(f"Test:  {len(test_set):,}")

    y_train = train_set[["isFraud"]].values.ravel()
    x_train = train_set.drop(columns=["TransactionID", "isFraud", "TransactionDT"])
    y_test = test_set[["isFraud"]].values.ravel()
    x_test = test_set.drop(columns=["TransactionID", "isFraud", "TransactionDT"])
    y_cv = cv_set[["isFraud"]].values.ravel()
    x_cv = cv_set.drop(columns=["TransactionID", "isFraud", "TransactionDT"])

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    return x_train, y_train, x_cv, y_cv, x_test, y_test, scale_pos_weight


def prepare_final_test(test):
    test_ids = test["TransactionID"]
    x_final_test = test.drop(columns=["TransactionID", "TransactionDT"])

    return x_final_test, test_ids
