import sys
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import gc
import re

class FeaturePipeline:
    def __init__(self):
        pass

    #Feature Engineering
    nan_as_category = True   
    def one_hot_encoder(self,df, nan_as_category=True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns


    def do_sum(self, dataframe, group_cols, counted, agg_name):
        gp = dataframe[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(columns={counted: agg_name})
        dataframe = dataframe.merge(gp, on=group_cols, how='left')
        return dataframe


    '''def reduce_mem_usage(dataframe):
        m_start = dataframe.memory_usage().sum() / 1024 ** 2
        for col in dataframe.columns:
            col_type = dataframe[col].dtype
            if col_type != object:
                c_min = dataframe[col].min()
                c_max = dataframe[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        dataframe[col] = dataframe[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        dataframe[col] = dataframe[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        dataframe[col] = dataframe[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        dataframe[col] = dataframe[col].astype(np.int64)
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        dataframe[col] = dataframe[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        dataframe[col] = dataframe[col].astype(np.float32)
                    else:
                        dataframe[col] = dataframe[col].astype(np.float64)

        m_end = dataframe.memory_usage().sum() / 1024 ** 2
        return dataframe'''


    def risk_groupanizer(self, dataframe, column_names, target_val=1, upper_limit_ratio=8.2, lower_limit_ratio=8.2):
    # one-hot encoder killer :-)
        all_cols = dataframe.columns
        for col in column_names:

            temp_df = dataframe.groupby([col] + ['TARGET'])[['SK_ID_CURR']].count().reset_index()
            temp_df['ratio%'] = round(temp_df['SK_ID_CURR']*100/temp_df.groupby([col])['SK_ID_CURR'].transform('sum'), 1)
            col_groups_high_risk = temp_df[(temp_df['TARGET'] == target_val) &
                                        (temp_df['ratio%'] >= upper_limit_ratio)][col].tolist()
            col_groups_low_risk = temp_df[(temp_df['TARGET'] == target_val) &
                                        (lower_limit_ratio >= temp_df['ratio%'])][col].tolist()
            if upper_limit_ratio != lower_limit_ratio:
                col_groups_medium_risk = temp_df[(temp_df['TARGET'] == target_val) &
                    (upper_limit_ratio > temp_df['ratio%']) & (temp_df['ratio%'] > lower_limit_ratio)][col].tolist()

                for risk, col_groups in zip(['_high_risk', '_medium_risk', '_low_risk'],
                                            [col_groups_high_risk, col_groups_medium_risk, col_groups_low_risk]):
                    dataframe[col + risk] = [1 if val in col_groups else 0 for val in dataframe[col].values]
            else:
                for risk, col_groups in zip(['_high_risk', '_low_risk'], [col_groups_high_risk, col_groups_low_risk]):
                    dataframe[col + risk] = [1 if val in col_groups else 0 for val in dataframe[col].values]
            if dataframe[col].dtype == 'O' or dataframe[col].dtype == 'object':
                dataframe.drop(col, axis=1, inplace=True)
        
        return dataframe, list(set(dataframe.columns).difference(set(all_cols)))

    
    