import sys
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import gc
import re
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,df):
        try:
            logging.info("Split training and test input data")
            
            print('===============================================', '\n', '##### the ML in processing...')

            # loading predicted result 
            df_subx = pd.read_csv(r"C:\Users\aksha\HCDR\notebook\submission.csv")
            df_sub = df_subx[['SK_ID_CURR', 'TARGET']]
            df_sub.columns = ['SK_ID_CURR', 'TARGET']
            
            # split train, and test datasets
            train_df = df[df['TARGET'].notnull()]
            test_df = df[df['TARGET'].isnull()]
            # delete main dataframe for saving memory
            del df_subx
            gc.collect()

                # Expand train dataset with two times of test dataset including predicted results
            for i in test_df.shape[0]:
                test_df.TARGET = np.where(df_sub.TARGET > 0.75, 1, 0)
                train_df = pd.concat([train_df, test_df], axis=0)
                train_df = pd.concat([train_df, test_df], axis=0)
                train_df = pd.concat([train_df, test_df], axis=0)
                print(f'Train shape: {train_df.shape}, test shape: {test_df.shape} are loaded.')

            # Cross validation model
            folds = KFold(n_splits=6, shuffle=True, random_state=667)

            # Create arrays and dataframes to store results
            oof_preds = np.zeros(train_df.shape[0])
            sub_preds = np.zeros(test_df.shape[0])

            # limit number of feature to only 174!!!
            feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV']]
            
            # print final shape of dataset to evaluate by LightGBM
            print(f'only {len(feats)} features from a total {train_df.shape[1]} features are used for ML analysis')

            for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
                train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
                valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
                clf = LGBMClassifier(nthread=-1,
                                    #device_type='gpu',
                                    n_estimators=5000,
                                    learning_rate=0.01,
                                    max_depth=11,
                                    num_leaves=58,
                                    colsample_bytree=0.613,
                                    subsample=0.708,
                                    max_bin=407,
                                    reg_alpha=3.564,
                                    reg_lambda=4.930,
                                    min_child_weight=6,
                                    min_child_samples=165,
                                    #keep_training_booster=True,
                                    silent=-1,
                                    verbose=-1,)

                clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc')

                oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
                sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

                print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
                '''del clf, train_x, train_y, valid_x, valid_y
                gc.collect()'''

            print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

            # create submission file
            test_df['TARGET'] = sub_preds
            test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission.csv', index=False)
            print('a submission file is created')
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=clf.best_iteration_
            )

            predicted=test_df

            roc_auc_score(train_df['TARGET'], oof_preds)
            return roc_auc_score
            



            
        except Exception as e:
            raise CustomException(e,sys)
        

class TrainPipeline:
    def __init__(self):
        pass

    def model1(self, df, n_folds):
        
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