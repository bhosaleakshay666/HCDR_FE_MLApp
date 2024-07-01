import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline


from src.exception import CustomException
from src.logger import logging
from src.pipeline.transform_pipeline import TransformPipeline
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def add_trx2(self, df):
        df['MA_10'] = df['Adj Close'].rolling(10).mean()
        df = df[df['MA_10'].notna()]
        df['MA_20'] = df['Adj Close'].rolling(20).mean()
        df = df[df['MA_20'].notna()]
        df['MA_60'] = df['Adj Close'].rolling(60).mean()
        df = df[df['MA_60'].notna()]
        df['Daily Return'] = df['Adj Close'].pct_change()
        df = df[df['Daily Return'].notna()]
        
        return df
        
    
        
    def get_data_transformer_object(self, data):
        
        try:
            prep_ob = self.add_trx2(data)
            return prep_ob
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def transform_data(self,train_path,test_path, bureau_path, burbal_path, payments_path, pos_path, cc_path, prev_app_path):

        try:
            train=pd.read_csv(train_path)
            test=pd.read_csv(test_path)
            bureau=pd.read_csv(bureau_path)
            burbal=pd.read_csv(burbal_path)
            payments=pd.read_csv(payments_path)
            cash=pd.read_csv(pos_path)
            cc=pd.read_csv(cc_path)
            prev_app=pd.read_csv(prev_app_path)

            logging.info("Reading all data and performing initial transformations to combine the training set")

            trx=TransformPipeline()
            df = trx.application(train,test,bureau)
            bureau_2 = trx.bureaubal(bureau, burbal)
            payments_ = trx.installment(payments)
            cash_ = trx.pos_cash(cash)
            cc_ = trx.credit_card(cc)
            prev_app_ = trx.previous_application(prev_app)

            logging.info("Obtaining preprocessing object")

            #preprocessing_obj=self.get_data_transformer_object()

            
            # Concatenate all the stock dataframes into one
            data = trx.merge(df, bureau_2, cash_, payments_, cc_, prev_app_)
            trxobj=self.get_data_transformer_object(data)

            logging.info(
                f"Applying preprocessing object on combined dataframe"
            )

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=trxobj

            )

            return (
                trxobj,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

   