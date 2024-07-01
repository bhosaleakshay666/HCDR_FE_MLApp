import sys
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import gc
import re
from src.pipeline.feature_engineering import FeaturePipeline

class TrainPipeline:
    def __init__(self):
        pass

    def data_post_processing(dataframe):
        fe=FeaturePipeline
        print(f'---=> the DATA POST-PROCESSING is beginning, the dataset has {dataframe.shape[1]} features')
        # keep index related columns
        index_cols = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']

        dataframe = dataframe.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
        
        feature_num = dataframe.shape[1]
        all_features = dataframe.columns.tolist()
        print(f'{feature_num - dataframe.shape[1]} features are eliminated by LightGBM classifier in step I')
        print(f'---=> {dataframe.shape[1]} features are remained after removing features not interesting for LightGBM classifier')


        # generate new columns with risk_groupanizer
        start_feats_num = dataframe.shape[1]
        cat_cols = [col for col in dataframe.columns if 3 < len(dataframe[col].value_counts()) < 20 and col not in index_cols]
        dataframe, _ = fe.risk_groupanizer(dataframe, column_names=cat_cols, upper_limit_ratio=8.1, lower_limit_ratio=8.1)
        print(f'---=> {dataframe.shape[1] - start_feats_num} features are generated with the risk_groupanizer')


        # ending message of DATA POST-PROCESSING
        print(f'---=> the DATA POST-PROCESSING is ended!, now the dataset has a total {dataframe.shape[1]} features')

        gc.collect()
        return dataframe