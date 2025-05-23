import os
import sys
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml

logger = get_logger(__name__)


def read_yaml(file_path: str) -> dict:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path : {file_path}")

        with open(file_path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info("Successfully loaded YAML file")
            return content

    except Exception as e:
        logger.error(f"Error while reading YAML File : {e}")
        raise CustomException("Failed to read YAML file", sys)


def load_data(path: str) -> pd.DataFrame:
    try:
        logger.info("Loading Data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error occured while loading file : {e}")
        raise CustomException("Error Loading file", sys)
