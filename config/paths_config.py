import os

# Data Ingestion
RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"

# Data Preprocessing
PROCESSED_DIR = "artifacts/processed"
XTRAIN_DATA = os.path.join(PROCESSED_DIR, "xtrain.npy")
YTRAIN_DATA = os.path.join(PROCESSED_DIR, "ytrain.npy")
XTEST_DATA = os.path.join(PROCESSED_DIR, "xtest.npy")
YTEST_DATA = os.path.join(PROCESSED_DIR, "ytest.npy")
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "preprocessor.joblib")

# Model training
MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model.joblib"
