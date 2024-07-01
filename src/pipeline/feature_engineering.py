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
    def one_hot_encoder(df, nan_as_category=True):
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
        new_columns = [c for c in df.columns if c not in original_columns]
        return df, new_columns

    def do_sum(dataframe, group_cols, counted, agg_name):
        gp = dataframe[group_cols + [counted]].groupby(group_cols)[counted].sum().reset_index().rename(columns={counted: agg_name})
        dataframe = dataframe.merge(gp, on=group_cols, how='left')
        return dataframe

    def reduce_mem_usage(dataframe):
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
        return dataframe

    


    def risk_groupanizer(dataframe, column_names, target_val=1, upper_limit_ratio=8.2, lower_limit_ratio=8.2):
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


    
    def model1(df, n_folds = 5):
        
        features = df[df['TARGET'].notnull()]
        test_features = df[df['TARGET'].isnull()]
        #features, test_features = train.align(test_features, join = 'inner', axis = 1)
        # Extract the ids
        train_ids = features['SK_ID_CURR']
        test_ids = test_features['SK_ID_CURR']
        
        # Extract the labels for training
        labels = features['TARGET']
        
        # Remove the ids and target
        features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
        test_features = test_features.drop(columns = ['SK_ID_CURR', 'TARGET'])

        # Extract feature names
        feature_names = list(features.columns)
        
        # Convert to np arrays
        features = np.array(features)
        test_features = np.array(test_features)
        
        # Create the kfold object
        k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
        
        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))
        
        # Empty array for test predictions
        test_predictions = np.zeros(test_features.shape[0])
        
        # Empty array for out of fold validation predictions
        out_of_fold = np.zeros(features.shape[0])
        
        # Lists for recording validation and training scores
        valid_scores = []
        train_scores = []
        
        # Iterate through each fold
        for train_indices, valid_indices in k_fold.split(features):
            
            # Training data for the fold
            train_features, train_labels = features[train_indices], labels[train_indices]
            # Validation data for the fold
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]
            
            # Create the model
            model = LGBMClassifier(n_estimators=2266, objective = 'binary', 
                                    class_weight = 'balanced', learning_rate = 0.01, 
                                    reg_alpha = 0.02, reg_lambda = 0.9, 
                                    subsample = 0.86667, n_jobs = -1, random_state = 500)
            
            # Train the model
            model.fit(train_features, train_labels, eval_metric = 'auc',
                    eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                    eval_names = ['valid', 'train'])
            
            # Record the best iteration
            best_iteration = model.best_iteration_
            
            # Record the feature importances
            feature_importance_values += model.feature_importances_ / k_fold.n_splits
            
            # Make predictions
            test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
            
            # Record the out of fold predictions
            out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
            
            # Record the best score
            valid_score = model.best_score_['valid']['auc']
            train_score = model.best_score_['train']['auc']
            
            valid_scores.append(valid_score)
            train_scores.append(train_score)
            
            # Clean up memory
            gc.enable()
            del model, train_features, valid_features
            gc.collect()
            
        # Make the submission dataframe
        submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
        
        # Make the feature importance dataframe
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
        
        # Overall validation score
        valid_auc = roc_auc_score(labels, out_of_fold)
        
        # Add the overall scores to the metrics
        valid_scores.append(valid_auc)
        train_scores.append(np.mean(train_scores))
        
        # Needed for creating dataframe of validation scores
        fold_names = list(range(n_folds))
        fold_names.append('overall')
        
        # Dataframe of validation scores
        ''' metrics = pd.DataFrame({'fold': fold_names,
                                'train': train_scores,
                                'valid': valid_scores})''' 
        fi_drop=feature_importances[feature_importances['importance']< 12]
        dropfeat=fi_drop['feature'].tolist()
        return submission, dropfeat