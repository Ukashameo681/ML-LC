import os
import sys
sys.path.append(r"D:\Data Science Projects life\Machine learning life Cycle\src")
from exception import CustomException
from logger import logging
import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from components.model_trainer import ModelTrainer
from components.data_transformation import DataTransformationConfig
from components.data_transformation import DataTransformation

@dataclass

class DataIngestionConfigration:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfigration()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("read the dataset into dataframe")
            
            # creating data path
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False, header=True)

            logging.info("Train test split is initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False, header=True)


            logging.info("Ingestion of the data is completed")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr, preprocessing_obj =  data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_Trainer(train_arr, test_arr))

