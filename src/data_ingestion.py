import pandas as pd
import os
from src.logger import get_logger  

logger = get_logger(__name__,"data_ingestion.log")

def data_ingestion():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, "data", "ecommerceDataset.csv")
        
        logger.info(f"Reading data from {train_path}")
        df = pd.read_csv(train_path)
        logger.info(f"Data shape: {df.shape}")
        return df

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise

if __name__ == '__main__':
    data = data_ingestion()
    print(data.columns)
