import os # To create file path
import sys # System Error
sys.path.append(r'D:\Projects\House_Price_Prediction')
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

##Initialise the Data Ingestion Configuration i.e. we give path as an input to configuration class

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

## creating a class for Data Ingestion

class DataIngestion:
    def __init__(self):
        self.ingestionConfig=DataIngestionConfig()

    def initiateDataIngesion(self):
        logging.info("Data Ingestion Method Starts")


        try:
            df=pd.read_csv('notebooks\data\gemstone.csv')
            logging.info('Dataset read as pandas DataFrame')

            os.makedirs(os.path.dirname(self.ingestionConfig.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestionConfig.raw_data_path,index=False)
            logging.info('Train Test Split')

            train_set,test_set=train_test_split(df,test_size=0.30,random_state=45)

            train_set.to_csv(self.ingestionConfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestionConfig.test_data_path,index=False,header=True)

            logging.info("Data Ingestion Completed")


            return (
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path
            )



        except Exception as e:
            logging.info("Exception occured at Data Ingestion Stage")
            raise CustomException(e,sys)

## Run Data Ingestion

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiateDataIngesion()