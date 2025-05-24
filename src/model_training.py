import os
import sys
import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *


logger = get_logger(__name__)


class ModelTraining:

    def __init__(self, xtrain_path, xtest_path, ytrain_path, ytest_path) -> None:
        self.xtrain = np.load(xtrain_path)
        self.xtest = np.load(xtest_path)
        self.ytrain = np.load(ytrain_path, allow_pickle=True)
        self.ytest = np.load(ytest_path, allow_pickle=True)

        self.model_output_path = MODEL_OUTPUT_PATH
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def train_model(self):
        try:
            logger.info("Initializing Model")
            lgbm_model = lgb.LGBMClassifier(random_state=42)

            logger.info("Starting Hyper Parameter tuning")
            random_search = RandomizedSearchCV(
                estimator=lgbm_model, **self.random_search_params  # type: ignore
            )
            random_search.fit(self.xtrain, self.ytrain)

            best_params = random_search.best_params_
            best_score = random_search.best_score_
            best_model = random_search.best_estimator_
            logger.info(f"Best Params : {best_params}")
            logger.info(f"Best Score : {best_score}")

            logger.info("Saving model object")
            model_dir = os.path.dirname(self.model_output_path)
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(best_model, self.model_output_path)
            return best_model
        except Exception as e:
            logger.error(f"Failed to train model : {e}")
            raise CustomException("Failed to train model", sys)

    def evaluate_model(self, model):
        try:
            logger.info(f"Evaluating model")
            ypred_test = model.predict(self.xtest)
            accuracy = accuracy_score(self.ytest, ypred_test)
            precision = precision_score(self.ytest, ypred_test, average="macro")
            recall = recall_score(self.ytest, ypred_test, average="macro")
            f1_macro = f1_score(self.ytest, ypred_test, average="macro")
            logger.info(f"Accuracy : {accuracy:.2%}")
            logger.info(f"Precision Macro : {precision:.2%}")
            logger.info(f"Recall Macro : {recall:.2%}")
            logger.info(f"F1 Macro : {f1_macro}")
            return {
                "accuracy": accuracy,
                "recall_macro": recall,
                "precision_macro": precision,
                "f1_macro": f1_macro,
            }
        except Exception as e:
            logger.error(f"Error occured while evaluating model : {e}")
            raise CustomException("Failed While Model Evaluation", sys)

    def run(self):
        try:
            model = self.train_model()
            self.evaluate_model(model)
        except Exception as e:
            logger.error(f"Error While Model Training")
            raise CustomException(f"Failed to train the model", sys)


if __name__ == "__main__":
    try:
        model_trainer = ModelTraining(
            xtrain_path=XTRAIN_DATA,
            xtest_path=XTEST_DATA,
            ytrain_path=YTRAIN_DATA,
            ytest_path=YTEST_DATA,
        )
        model_trainer.run()
    except CustomException as ce:
        logger.error(f"Failed to train model : {str(ce)}")
