import os
import sys
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_functions import read_yaml, load_data
from config.paths_config import *


logger = get_logger(__name__)


class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        os.makedirs(self.processed_dir, exist_ok=True)

    def preprocess_data(self):
        try:
            logger.info("Starting Data Preprocessing step")

            logger.info("Loading training file as dataframe")
            train = load_data(self.train_path)

            logger.info("Dropping Unnecessary columns")
            train = train.drop(columns=["Booking_ID"])

            logger.info("Dropping Duplicate rows")
            train = train.drop_duplicates(keep="first").reset_index(drop=True)

            preprocess_config = self.config["data_preprocessing"]
            target_col = preprocess_config["target_column"]
            cat_cols = preprocess_config["categorical_columns"]
            num_cols = preprocess_config["numerical_columns"]

            logger.info("Seperating Target columns")
            xtrain = train.drop(columns=target_col)
            ytrain = train[target_col].values.flatten()

            logger.info("Preprocessing data")
            num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
            cat_pipe = make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(),
                StandardScaler(),
            )
            preprocessor = ColumnTransformer(
                [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
            )
            xtrain_pre = preprocessor.fit_transform(xtrain)

            logger.info("SMOTE Oversampling")
            smote = SMOTE(random_state=42)
            xtrain_res, ytrain_res = smote.fit_resample(xtrain_pre, ytrain)  # type: ignore

            logger.info("Saving preprocessed train data")
            np.save(XTRAIN_DATA, xtrain_res)
            np.save(YTRAIN_DATA, ytrain_res)

            logger.info("Loading and transforming test data")
            test = load_data(self.test_path)
            xtest = test.drop(columns=["Booking_ID"] + target_col)
            ytest = test[target_col].values.flatten()
            xtest_pre = preprocessor.transform(xtest)

            logger.info("Saving preprocessed test data")
            np.save(XTEST_DATA, xtest_pre)  # type: ignore
            np.save(YTEST_DATA, ytest)

            logger.info("Saving Preprocessor to joblib file")
            joblib.dump(preprocessor, PREPROCESSOR_PATH)

        except Exception as e:
            logger.error(f"Error occured during data preprocessing : {e}")
            raise CustomException("Failed to Preprocess Data", sys)


if __name__ == "__main__":
    try:
        data_preprocessor = DataProcessor(
            train_path=TRAIN_FILE_PATH,
            test_path=TEST_FILE_PATH,
            processed_dir=PROCESSED_DIR,
            config_path=CONFIG_PATH,
        )
        data_preprocessor.preprocess_data()
    except CustomException as ce:
        logger.error(f"Error occured while data preprocessing : {ce}")
