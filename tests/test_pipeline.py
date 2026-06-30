import pytest
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.csv')

EXPECTED_COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
CRITICAL_COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']


@pytest.fixture
def df():
    return pd.read_csv(DATA_PATH)


def test_dataset_has_expected_columns(df):
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Colonnes attendues: {EXPECTED_COLUMNS}, colonnes trouvées: {list(df.columns)}"
    )


def test_no_critical_column_is_empty(df):
    for col in CRITICAL_COLUMNS:
        assert col in df.columns, f"Colonne critique manquante: {col}"
        assert df[col].notna().all(), f"La colonne '{col}' contient des valeurs manquantes"
