import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline


from src.exception import CustomException
from src.logger import logging
from src.pipeline.transform_pipeline import TransformPipeline
from src.components.model_trainer import TrainPipeline
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessed.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    
             
    def get_data_transformed(self, data):
        
        try:
            lgbm1=TrainPipeline()
            sub1, feat = lgbm1.model1(data)
            data1 = data.drop(columns=feat)
            
            data1.to_csv(self.data_transformation_config.preprocessor_obj_file_path,index=False,header=True)
            logging.info('Read the processed dataset as dataframe')
            return data1
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_transform_data(self,train_path,test_path, bureau_path, burbal_path, payments_path, pos_path, cc_path, prev_app_path):

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
            data = trx.remove_missing_columns(data, threshold=77)
            noninformative_cols = []
            for col in data.columns:
                if len(data[col].value_counts()) < 2:
                    noninformative_cols.append(col)
            data=data.drop(columns=noninformative_cols)
            data = trx.remove_missing_columns(data)
            trxobj=self.get_data_transformed(data)

            logging.info(
                f"Applying preprocessing object on combined dataframe"
            )

            logging.info(f"Saved preprocessed dataframe.")

            '''save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=trxobj

            )'''

            return (
                trxobj,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

   