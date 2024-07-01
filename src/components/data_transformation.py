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

    def add_ma4eda(self, df):
        df['MA_10'] = df['Adj Close'].rolling(10).mean()
        df = df[df['MA_10'].notna()]
        df['MA_20'] = df['Adj Close'].rolling(20).mean()
        df = df[df['MA_20'].notna()]
        df['MA_60'] = df['Adj Close'].rolling(60).mean()
        df = df[df['MA_60'].notna()]
        df['Daily Return'] = df['Adj Close'].pct_change()
        df = df[df['Daily Return'].notna()]
        
        return df
        
    
        
    def get_data_transformer_object(self, all_stocks):
        
        try:
            prep_ob = self.add_ma4eda(all_stocks)
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
            '''all_stocks = pd.concat([train, test, bureau, burbal], axis=0)
            all_stocks.reset_index(drop=True, inplace=True)
            concat_obj=self.get_data_transformer_object(all_stocks)'''

            logging.info(
                f"Applying preprocessing object on combined dataframe"
            )

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=concat_obj

            )

            return (
                concat_obj,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

   