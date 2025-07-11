import pandas as pd
from src.data_ingestion import data_ingestion

def test_data_ingestion_runs():
    df = data_ingestion()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[0] > 1000  
    assert df.shape[1] >= 2  