import os
import sys
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)


class DataIngestion:

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = read_yaml(self.config_path)["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(
            f"Data Ingestion started with {self.bucket_name} and file is {self.file_name}"
        )

    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            with open(RAW_FILE_PATH, "wb") as f:
                blob.download_to_file(f)
            logger.info(f"CSV file successfully downloaded to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"CSV File failed to download : {e}")
            raise CustomException("Failed to download CSV", sys)

    def split_data(self):
        try:
            logger.info("Starting Train Test Split")
            data = pd.read_csv(RAW_FILE_PATH)

            train, test = train_test_split(
                data, train_size=self.train_ratio, random_state=42
            )

            train.to_csv(TRAIN_FILE_PATH, index=False)
            logger.info(f"Train File Saved in {TRAIN_FILE_PATH}")
            test.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Test File Saved in {TEST_FILE_PATH}")

        except Exception as e:
            logger.error(f"Failed to train test split : {e}")
            raise CustomException("Failed to perform train test split", sys)

    def run(self):
        try:
            logger.info("Data Ingestion started")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data Ingestion successfull")
        except CustomException as ce:
            logger.error(f"Data Ingestion Failed : {str(ce)}")


if __name__ == "__main__":
    data_ingestion = DataIngestion(CONFIG_PATH)
    data_ingestion.run()
