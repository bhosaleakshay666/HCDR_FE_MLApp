import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from src.components.data_transformtion import DataTransformation, DataTransformationConfig

#from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"application_train.csv")
    test_data_path: str=os.path.join('artifacts',"application_test.csv")
    bureau_data_path: str=os.path.join('artifacts',"bureau.csv")
    burbal_data_path: str=os.path.join('artifacts',"bureau_balance.csv")
    pos_data_path: str=os.path.join('artifacts',"POS_CASH_balance.csv")
    payments_data_path: str=os.path.join('artifacts',"installments_payments.xlsx")
    cc_data_path: str=os.path.join('artifacts',"credit_card_balance.csv")
    prev_app_path: str=os.path.join('artifacts',"previous_application.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    
    def initiate_data_ingestion(self, train, test, bureau, bureau_bal, cc_bal, prev_app, payments, cash):
        logging.info("Entered the  data ingestion method or component")
        try:
            
        #Ingestion address


            train=pd.read_csv(r"C:\Users\aksha\notebooks\data\application_train.csv")
            test=pd.read_csv(r"C:\Users\aksha\notebooks\data\application_test.csv")
            bureau=pd.read_csv(r"C:\Users\aksha\notebooks\data\bureau.csv")
            bureau_bal=pd.read_csv(r"C:\Users\aksha\notebooks\data\bureau_balance.csv")
            cc_bal=pd.read_csv(r"C:\Users\aksha\notebooks\data\credit_card_balance.csv")
            payments=pd.read_csv(r"C:\Users\aksha\notebooks\data\installments_payments.csv")
            cash=pd.read_csv(r"C:\Users\aksha\notebooks\data\POS_CASH_balance.csv")
            prev_app=pd.read_csv(r"C:\Users\aksha\notebooks\data\previous_application.csv")
            
            train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info('Read the train dataset as dataframe')
            test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Read the test dataset as dataframe') 

            bureau.to_csv(self.ingestion_config.bureau_data_path,index=False,header=True)
            logging.info('Read the bureau dataset as dataframe')
            
            bureau_bal.to_csv(self.ingestion_config.burbal_data_path,index=False,header=True)
            logging.info('Read the bureau balance dataset as dataframe')

            cash.to_csv(self.ingestion_config.pos_data_path,index=False,header=True)
            logging.info('Read the POS CASH balance dataset as dataframe')
            
            cc_bal.to_csv(self.ingestion_config.cc_data_path,index=False,header=True)
            logging.info('Read the Credit Card Balance dataset as dataframe')

            payments.to_csv(self.ingestion_config.payments_data_path,index=False,header=True)
            logging.info('Read the Installment Payment dataset as dataframe')

            prev_app.to_csv(self.ingestion_config.prev_app_path,index=False,header=True)
            logging.info('Read the previous application dataset as dataframe')

            
            #os.makedirs(os.path.dirname(self.ingestion_config.all_data_path), exist_ok=True)
            logging.info("Transformation Intitated")
            

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.bureau_data_path,
                self.ingestion_config.burbal_data_path,
                self.ingestion_config.cc_data_path,
                self.ingestion_config.pos_data_path,
                self.ingestion_config.payments_data_path,
                self.ingestion_config.prev_app_path)
        
        
        except Exception as e:
            raise CustomException(e,sys)
        

    
        

    

if __name__=="__main__":
    di=DataIngestion()
    di.initiate_data_ingestion()
    

    