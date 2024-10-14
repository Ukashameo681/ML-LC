import os
import sys 
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd

import pickle
from exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Training set performance
            model_train_score  = r2_score(y_train, y_train_pred)
            model_test_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] =  model_test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)        