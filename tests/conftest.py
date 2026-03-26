import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def sample_data():
    """Cria dados de amostra para testes"""
    np.random.seed(42)

    # Cria dataset sintético similar ao dataset real
    n_samples = 5000
    X, y = make_classification(
        n_samples=n_samples,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        random_state=42,
        weights=[0.95, 0.05],  # Desbalanceado como dados reais
    )

    # Cria colunas obrigatórias
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(50)])
    df["TransactionID"] = range(1, n_samples + 1)
    df["TransactionDT"] = np.random.randint(0, 10000, n_samples)
    df["isFraud"] = y
    df["TransactionAmt"] = np.random.exponential(100, n_samples)
    df["addr1"] = np.random.randint(1, 100, n_samples)
    df["card1"] = np.random.randint(1, 50, n_samples)
    df["card2"] = np.random.randint(1, 50, n_samples)
    df["card4"] = np.random.choice(["visa", "mastercard", "amex"], n_samples)
    df["card6"] = np.random.choice(["credit", "debit"], n_samples)
    df["ProductCD"] = np.random.choice(["C", "H", "W", "R", "S"], n_samples)
    df["P_emaildomain"] = np.random.choice(
        ["gmail.com", "yahoo.com", "hotmail.com"], n_samples
    ).astype(str)
    df["R_emaildomain"] = np.random.choice(
        ["gmail.com", "yahoo.com", "hotmail.com"], n_samples
    ).astype(str)
    df["DeviceType"] = np.random.choice(["mobile", "desktop"], n_samples)
    df["DeviceInfo"] = np.random.choice(
        ["Windows", "iOS", "Android", "Mac"], n_samples
    ).astype(str)
    df["D1"] = np.random.randint(0, 400, n_samples)
    df["C1"] = np.random.randint(0, 100, n_samples)

    # Adiciona colunas categóricas
    for col in [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
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
    ]:
        df[col] = np.random.choice(["A", "B", "C", "D"], n_samples)

    return df


@pytest.fixture
def train_data(sample_data):
    """Training split (70% of the sample)"""
    train = sample_data.iloc[: int(len(sample_data) * 0.70)].copy()
    return train


@pytest.fixture
def test_data(sample_data):
    """Test split (30% of the sample)"""
    test = sample_data.iloc[int(len(sample_data) * 0.70) :].copy()
    return test


@pytest.fixture
def fitted_data(train_data, test_data):
    """Data processed through the feature engineering pipeline."""
    from src.feature_engineer import fit_transform

    train = train_data.copy()
    test = test_data.copy()

    train, test, artifacts = fit_transform(train, test)

    return train, test, artifacts


@pytest.fixture
def artifacts(fitted_data):
    """Training artifacts only — convenient for transform() tests."""
    _, _, arts = fitted_data
    return arts


@pytest.fixture
def split_datasets(fitted_data):
    """Split datasets ready for model training."""
    from src.load_data import split_train_data

    train, _, _ = fitted_data
    return split_train_data(train)
