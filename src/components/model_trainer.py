import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
            self.model_trainer_config = ModelTrainerConfig()



    def initiate_model_Trainer(self,train_array, test_array):
            try:
                logging.info("Initiating the model trainer")
                X_Train, Y_Train, X_Test, Y_Test = (
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1],
                )
                models = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "XGBClassifier": XGBRegressor(),
                    "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                }  

                model_report: dict = evaluate_models(X_Train, Y_Train, X_Test, Y_Test, models)


                best_model_score = max(sorted(model_report.values()))

                best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]

                best_model = models[best_model_name]

                if best_model_score < 0.6:
                    raise CustomException("No best model found")
                logging.info("Best model found on both training and testing dataset")

                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                predicted = best_model.predict(X_Test)

                r2_square = r2_score(Y_Test, predicted)
                return r2_square
            except Exception as e:
                raise CustomException(e, sys)