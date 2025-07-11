import pandas as pd
from src.data_preprocessing import data_preprocessing

def test_data_preprocessing_runs():
    df = data_preprocessing()
    assert isinstance(df, pd.DataFrame)
    assert "class" in df.columns
    assert "Description" in df.columns
    assert df.shape[0] > 1000
