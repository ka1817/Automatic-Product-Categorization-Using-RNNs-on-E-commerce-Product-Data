from src.data_ingestion import data_ingestion
from src.logger import get_logger

logger = get_logger(__name__, "data_preprocessing.log")

def data_preprocessing():
    try:
        df = data_ingestion()
        logger.info("Data ingested successfully")

        logger.info(f"Original columns: {df.columns.tolist()}")

        if df.columns.size == 2:
            df.columns = ['class', 'Description']
            logger.info("Columns renamed to ['class', 'Description'].")
        else:
            logger.warning("Unexpected number of columns. Attempting to fix by renaming.")
            df = df.rename(columns={
                df.columns[0]: 'class',
                df.columns[1]: 'Description'
            })
            logger.info("Columns forcibly renamed based on index.")

        logger.info(f"Processed Data Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    data = data_preprocessing()
    print(data.columns)
