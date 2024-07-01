import sys
import os
import numpy as np
import pandas as pd
from src.pipeline.feature_engineering import FeaturePipeline
import gc
import re

class TransformPipeline:

    

    def __init__(self):
        pass
    
    def group(self, df_to_agg, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
        agg_df = df_to_agg.groupby(aggregate_by).agg(aggregations)
        agg_df.columns = pd.Index(['{}{}_{}'.format(prefix, e[0], e[1].upper())
                                for e in agg_df.columns.tolist()])
        return agg_df.reset_index()

    
    def group_and_merge(self, df_to_agg, df_to_merge, prefix, aggregations, aggregate_by= 'SK_ID_CURR'):
        agg_df = self.group(df_to_agg, prefix, aggregations, aggregate_by= aggregate_by)
        return df_to_merge.merge(agg_df, how='left', on= aggregate_by)


    def application(self, df1, df2,bureau):
        

        df = pd.concat([df1, df2], ignore_index=True)
        fe=FeaturePipeline()
        # general cleaning procedures
        df = df[df['CODE_GENDER'] != 'XNA']
        df = df[df['AMT_INCOME_TOTAL'] < 20000000] # remove a outlier 117M
        # NaN values for DAYS_EMPLOYED: 365.243 -> nan
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True) # set null value
        df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True) # set null value

        # Categorical features with Binary encode (0 or 1; two categories)
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], uniques = pd.factorize(df[bin_feature])
        
        # Categorical features with One-Hot encode
        df, cat_cols = fe.one_hot_encoder(df, nan_as_category=True)

        # Flag_document features - count and kurtosis
        docs = [f for f in df.columns if 'FLAG_DOC' in f]
        df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
        df['NEW_DOC_KURT'] = df[docs].kurtosis(axis=1)

        def get_age_label(days_birth):
            """ Return the age group label (int). """
            age_years = -days_birth / 365
            if age_years < 27: return 1
            elif age_years < 40: return 2
            elif age_years < 50: return 3
            elif age_years < 65: return 4
            elif age_years < 99: return 5
            else: return 0
        # Categorical age - based on target=1 plot
        df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_label(x))

        # New features based on External sources
        df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['EXT_SOURCES_WEIGHTED'] = df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 1 + df.EXT_SOURCE_3 * 3
        #np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
            feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
            df[feature_name] = eval('np.{}'.format(function_name))(
                df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

        # Some simple new features (percentages)
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

        # Credit ratios
        df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        
        # Income ratios
        df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
        df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
        
        # Time ratios
        df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
        df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
        df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']

        # EXT_SOURCE_X FEATURE
        df['APPS_EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['APPS_EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        df['APPS_EXT_SOURCE_STD'] = df['APPS_EXT_SOURCE_STD'].fillna(df['APPS_EXT_SOURCE_STD'].mean())
        df['APP_SCORE1_TO_BIRTH_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_BIRTH'] / 365.25)
        df['APP_SCORE2_TO_BIRTH_RATIO'] = df['EXT_SOURCE_2'] / (df['DAYS_BIRTH'] / 365.25)
        df['APP_SCORE3_TO_BIRTH_RATIO'] = df['EXT_SOURCE_3'] / (df['DAYS_BIRTH'] / 365.25)
        df['APP_SCORE1_TO_EMPLOY_RATIO'] = df['EXT_SOURCE_1'] / (df['DAYS_EMPLOYED'] / 365.25)
        df['APP_EXT_SOURCE_2*EXT_SOURCE_3*DAYS_BIRTH'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['DAYS_BIRTH']
        df['APP_SCORE1_TO_FAM_CNT_RATIO'] = df['EXT_SOURCE_1'] / df['CNT_FAM_MEMBERS']
        df['APP_SCORE1_TO_GOODS_RATIO'] = df['EXT_SOURCE_1'] / df['AMT_GOODS_PRICE']
        df['APP_SCORE1_TO_CREDIT_RATIO'] = df['EXT_SOURCE_1'] / df['AMT_CREDIT']
        df['APP_SCORE1_TO_SCORE2_RATIO'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_2']
        df['APP_SCORE1_TO_SCORE3_RATIO'] = df['EXT_SOURCE_1'] / df['EXT_SOURCE_3']
        df['APP_SCORE2_TO_CREDIT_RATIO'] = df['EXT_SOURCE_2'] / df['AMT_CREDIT']
        df['APP_SCORE2_TO_REGION_RATING_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_RATING_CLIENT']
        df['APP_SCORE2_TO_CITY_RATING_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_RATING_CLIENT_W_CITY']
        df['APP_SCORE2_TO_POP_RATIO'] = df['EXT_SOURCE_2'] / df['REGION_POPULATION_RELATIVE']
        df['APP_SCORE2_TO_PHONE_CHANGE_RATIO'] = df['EXT_SOURCE_2'] / df['DAYS_LAST_PHONE_CHANGE']
        df['APP_EXT_SOURCE_1*EXT_SOURCE_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
        df['APP_EXT_SOURCE_1*EXT_SOURCE_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
        df['APP_EXT_SOURCE_2*EXT_SOURCE_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['APP_EXT_SOURCE_1*DAYS_EMPLOYED'] = df['EXT_SOURCE_1'] * df['DAYS_EMPLOYED']
        df['APP_EXT_SOURCE_2*DAYS_EMPLOYED'] = df['EXT_SOURCE_2'] * df['DAYS_EMPLOYED']
        df['APP_EXT_SOURCE_3*DAYS_EMPLOYED'] = df['EXT_SOURCE_3'] * df['DAYS_EMPLOYED']

        # AMT_INCOME_TOTAL : income
        # CNT_FAM_MEMBERS  : the number of family members
        df['APPS_GOODS_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
        df['APPS_CNT_FAM_INCOME_RATIO'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        
        # DAYS_BIRTH : Client's age in days at the time of application
        # DAYS_EMPLOYED : How many days before the application the person started current employment
        df['APPS_INCOME_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']

        # other feature from better than 0.8
        df['CREDIT_TO_GOODS_RATIO_2'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['APP_AMT_INCOME_TOTAL_12_AMT_ANNUITY_ratio'] = df['AMT_INCOME_TOTAL'] / 12. - df['AMT_ANNUITY']
        df['APP_INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
        df['APP_DAYS_LAST_PHONE_CHANGE_DAYS_EMPLOYED_ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
        df['APP_DAYS_EMPLOYED_DAYS_BIRTH_diff'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
        
        # Groupby the client id (SK_ID_CURR), count the number of previous loans, and rename the column
        previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
        #Merging cell
        df = df.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')


        dropfeatures = ['FLAG_OWN_CAR','OWN_CAR_AGE','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL','HOUR_APPR_PROCESS_START','APARTMENTS_AVG','BASEMENTAREA_AVG','YEARS_BEGINEXPLUATATION_AVG','YEARS_BUILD_AVG','COMMONAREA_AVG','ELEVATORS_AVG','ENTRANCES_AVG','FLOORSMAX_AVG','FLOORSMIN_AVG','LANDAREA_AVG','LIVINGAPARTMENTS_AVG','LIVINGAREA_AVG','NONLIVINGAPARTMENTS_AVG','NONLIVINGAREA_AVG','APARTMENTS_MODE','BASEMENTAREA_MODE','YEARS_BEGINEXPLUATATION_MODE','YEARS_BUILD_MODE','COMMONAREA_MODE','ELEVATORS_MODE','ENTRANCES_MODE','FLOORSMAX_MODE','FLOORSMIN_MODE','LANDAREA_MODE','LIVINGAPARTMENTS_MODE','LIVINGAREA_MODE','NONLIVINGAPARTMENTS_MODE','APARTMENTS_MEDI','BASEMENTAREA_MEDI','YEARS_BEGINEXPLUATATION_MEDI','YEARS_BUILD_MEDI','COMMONAREA_MEDI','ELEVATORS_MEDI','ENTRANCES_MEDI','FLOORSMAX_MEDI','FLOORSMIN_MEDI','LANDAREA_MEDI','LIVINGAPARTMENTS_MEDI','LIVINGAREA_MEDI','NONLIVINGAPARTMENTS_MEDI','NONLIVINGAREA_MEDI','TOTALAREA_MODE','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
        df=df.drop(columns=dropfeatures)
        print('Final shape:', df.shape)


        return df


    def bureaubal(self, bureau, bb):

        fe=FeaturePipeline()
        # Credit duration and credit/account end date difference
        bureau['CREDIT_DURATION'] = -bureau['DAYS_CREDIT'] + bureau['DAYS_CREDIT_ENDDATE']
        bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
        
        # Credit to debt ratio and difference
        bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
        bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
        bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']
        bureau['BUREAU_CREDIT_FACT_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
        bureau['BUREAU_CREDIT_ENDDATE_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
        bureau['BUREAU_CREDIT_DEBT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']

        # CREDIT_DAY_OVERDUE :
        bureau['BUREAU_IS_DPD'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 0 else 0)
        bureau['BUREAU_IS_DPD_OVER120'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 120 else 0)

        bb, bb_cat = fe.one_hot_encoder(bb, nan_as_category = True)
        bureau, bureau_cat = fe. one_hot_encoder(bureau, nan_as_category = True)

        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size', 'mean']}
        for col in bb_cat:
            bb_aggregations[col] = ['mean']

        #Status of Credit Bureau loan during the month
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')

        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean', 'min'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max', 'sum'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean', 'sum'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
            'SK_ID_BUREAU': ['count'],
            'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
            'ENDDATE_DIF': ['min', 'max', 'mean'],
            'BUREAU_CREDIT_FACT_DIFF': ['min', 'max', 'mean'],
            'BUREAU_CREDIT_ENDDATE_DIFF': ['min', 'max', 'mean'],
            'BUREAU_CREDIT_DEBT_RATIO': ['min', 'max', 'mean'],
            'DEBT_CREDIT_DIFF': ['min', 'max', 'mean'],
            'BUREAU_IS_DPD': ['mean', 'sum'],
            'BUREAU_IS_DPD_OVER120': ['mean', 'sum']
            }

        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat: cat_aggregations[cat] = ['mean']
        for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

        # Bureau: Active credits - using only numerical aggregations
        active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')

        # Bureau: Closed credits - using only numerical aggregations
        closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

        print('"Bureau/Bureau Balance" final shape:', bureau_agg.shape)
        return bureau_agg


    def previous_application(self, prev):
        
        fe=FeaturePipeline()
        prev, cat_cols = fe.one_hot_encoder(prev, nan_as_category=True)

        # Days 365.243 values -> nan
        prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

        # Add feature: value ask / value received percentage
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

        # Feature engineering: ratios and difference
        prev['APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
        prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
        prev['DOWN_PAYMENT_TO_CREDIT'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']

        # Interest ratio on previous application (simplified)
        total_payment = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
        prev['SIMPLE_INTERESTS'] = (total_payment / prev['AMT_CREDIT'] - 1) / prev['CNT_PAYMENT']

        # Days last due difference (scheduled x done)
        prev['DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']

        # from off
        prev['PREV_GOODS_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
        prev['PREV_ANNUITY_APPL_RATIO'] = prev['AMT_ANNUITY']/prev['AMT_APPLICATION']
        prev['PREV_GOODS_APPL_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_APPLICATION']

        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
            'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
            'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
            'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
            'SK_ID_PREV': ['nunique'],
            'DAYS_TERMINATION': ['max'],
            'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
            'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean', 'sum'],
            'DOWN_PAYMENT_TO_CREDIT': ['mean'],
            'PREV_GOODS_DIFF': ['mean', 'max', 'sum'],
            'PREV_GOODS_APPL_RATIO': ['mean', 'max'],
            'DAYS_LAST_DUE_DIFF': ['mean', 'max', 'sum'],
            'SIMPLE_INTERESTS': ['mean', 'max']
        }

        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean']

        prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

        # Previous Applications: Approved Applications - only numerical features
        approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

        # Previous Applications: Refused Applications - only numerical features
        refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

        print('"Previous Applications" final shape:', prev_agg.shape)
        return prev_agg


    def pos_cash(self, pos):
        
        fe=FeaturePipeline()
        pos, cat_cols = fe.one_hot_encoder(pos, nan_as_category=True)

        # Flag months with late payment
        pos['LATE_PAYMENT'] = pos['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
        pos['POS_IS_DPD'] = pos['SK_DPD'].apply(lambda x: 1 if x > 0 else 0) # <-- same with ['LATE_PAYMENT']
        pos['POS_IS_DPD_UNDER_120'] = pos['SK_DPD'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
        pos['POS_IS_DPD_OVER_120'] = pos['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)

        # Features
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size', 'min'],
            'SK_DPD': ['max', 'mean', 'sum', 'var', 'min'],
            'SK_DPD_DEF': ['max', 'mean', 'sum'],
            'SK_ID_PREV': ['nunique'],
            'LATE_PAYMENT': ['mean'],
            'SK_ID_CURR': ['count'],
            'CNT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
            'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum'],
            'POS_IS_DPD': ['mean', 'sum'],
            'POS_IS_DPD_UNDER_120': ['mean', 'sum'],
            'POS_IS_DPD_OVER_120': ['mean', 'sum'],
        }

        for cat in cat_cols:
            aggregations[cat] = ['mean']

        pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

        # Count pos cash accounts
        pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()


        sort_pos = pos.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
        gp = sort_pos.groupby('SK_ID_PREV')
        df_pos = pd.DataFrame()
        df_pos['SK_ID_CURR'] = gp['SK_ID_CURR'].first()
        df_pos['MONTHS_BALANCE_MAX'] = gp['MONTHS_BALANCE'].max()

        # Percentage of previous loans completed and completed before initial term
        df_pos['POS_LOAN_COMPLETED_MEAN'] = gp['NAME_CONTRACT_STATUS_Completed'].mean()
        df_pos['POS_COMPLETED_BEFORE_MEAN'] = gp['CNT_INSTALMENT'].first() - gp['CNT_INSTALMENT'].last()
        df_pos['POS_COMPLETED_BEFORE_MEAN'] = df_pos.apply(lambda x: 1 if x['POS_COMPLETED_BEFORE_MEAN'] > 0 \
                                                                        and x['POS_LOAN_COMPLETED_MEAN'] > 0 else 0, axis=1)
        # Number of remaining installments (future installments) and percentage from total
        df_pos['POS_REMAINING_INSTALMENTS'] = gp['CNT_INSTALMENT_FUTURE'].last()
        df_pos['POS_REMAINING_INSTALMENTS_RATIO'] = gp['CNT_INSTALMENT_FUTURE'].last()/gp['CNT_INSTALMENT'].last()

        # Group by SK_ID_CURR and merge
        df_gp = df_pos.groupby('SK_ID_CURR').sum().reset_index()
        df_gp.drop(['MONTHS_BALANCE_MAX'], axis=1, inplace= True)
        pos_agg = pd.merge(pos_agg, df_gp, on= 'SK_ID_CURR', how= 'left')

        # Percentage of late payments for the 3 most recent applications
        pos = fe.do_sum(pos, ['SK_ID_PREV'], 'LATE_PAYMENT', 'LATE_PAYMENT_SUM')

        # Last month of each application
        last_month_df = pos.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()

        # Most recent applications (last 3)
        sort_pos = pos.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
        gp = sort_pos.iloc[last_month_df].groupby('SK_ID_CURR').tail(3)
        gp_mean = gp.groupby('SK_ID_CURR').mean().reset_index()
        pos_agg = pd.merge(pos_agg, gp_mean[['SK_ID_CURR', 'LATE_PAYMENT_SUM']], on='SK_ID_CURR', how='left')

        print('"Pos-Cash" balance final shape:', pos_agg.shape) 
        return pos_agg


    def installment(self, ins):
        
        fe=FeaturePipeline()
        ins, cat_cols = fe.one_hot_encoder(ins, nan_as_category=True)

        # Group payments and get Payment difference
        ins = fe.do_sum(ins, ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], 'AMT_PAYMENT', 'AMT_PAYMENT_GROUPED')
        ins['PAYMENT_DIFFERENCE'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT_GROUPED']
        ins['PAYMENT_RATIO'] = ins['AMT_INSTALMENT'] / ins['AMT_PAYMENT_GROUPED']
        ins['PAID_OVER_AMOUNT'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
        ins['PAID_OVER'] = (ins['PAID_OVER_AMOUNT'] > 0).astype(int)

        # Percentage and difference paid in each installment (amount paid and installment value)
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

        # Days past due and days before due (no negative values)
        ins['DPD_diff'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD_diff'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD_diff'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD_diff'].apply(lambda x: x if x > 0 else 0)

        # Flag late payment
        ins['LATE_PAYMENT'] = ins['DBD'].apply(lambda x: 1 if x > 0 else 0)
        ins['INSTALMENT_PAYMENT_RATIO'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['LATE_PAYMENT_RATIO'] = ins.apply(lambda x: x['INSTALMENT_PAYMENT_RATIO'] if x['LATE_PAYMENT'] == 1 else 0, axis=1)

        # Flag late payments that have a significant amount
        ins['SIGNIFICANT_LATE_PAYMENT'] = ins['LATE_PAYMENT_RATIO'].apply(lambda x: 1 if x > 0.05 else 0)
        
        # Flag k threshold late payments
        ins['DPD_7'] = ins['DPD'].apply(lambda x: 1 if x >= 7 else 0)
        ins['DPD_15'] = ins['DPD'].apply(lambda x: 1 if x >= 15 else 0)

        ins['INS_IS_DPD_UNDER_120'] = ins['DPD'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
        ins['INS_IS_DPD_OVER_120'] = ins['DPD'].apply(lambda x: 1 if (x >= 120) else 0)

        # Features: Perform aggregations
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum', 'var'],
            'DBD': ['max', 'mean', 'sum', 'var'],
            'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
            'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum', 'min'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'min'],
            'SK_ID_PREV': ['size', 'nunique'],
            'PAYMENT_DIFFERENCE': ['mean'],
            'PAYMENT_RATIO': ['mean', 'max'],
            'LATE_PAYMENT': ['mean', 'sum'],
            'SIGNIFICANT_LATE_PAYMENT': ['mean', 'sum'],
            'LATE_PAYMENT_RATIO': ['mean'],
            'DPD_7': ['mean'],
            'DPD_15': ['mean'],
            'PAID_OVER': ['mean'],
            'DPD_diff':['mean', 'min', 'max'],
            'DBD_diff':['mean', 'min', 'max'],
            'DAYS_INSTALMENT': ['mean', 'max', 'sum'],
            'INS_IS_DPD_UNDER_120': ['mean', 'sum'],
            'INS_IS_DPD_OVER_120': ['mean', 'sum']
        }

        for cat in cat_cols:
            aggregations[cat] = ['mean']
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

        # Count installments accounts
        ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

        # from oof (DAYS_ENTRY_PAYMENT)
        cond_day = ins['DAYS_ENTRY_PAYMENT'] >= -365
        ins_d365_grp = ins[cond_day].groupby('SK_ID_CURR')
        ins_d365_agg_dict = {
            'SK_ID_CURR': ['count'],
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DAYS_ENTRY_PAYMENT': ['mean', 'max', 'sum'],
            'DAYS_INSTALMENT': ['mean', 'max', 'sum'],
            'AMT_INSTALMENT': ['mean', 'max', 'sum'],
            'AMT_PAYMENT': ['mean', 'max', 'sum'],
            'PAYMENT_DIFF': ['mean', 'min', 'max', 'sum'],
            'PAYMENT_PERC': ['mean', 'max'],
            'DPD_diff': ['mean', 'min', 'max'],
            'DPD': ['mean', 'sum'],
            'INS_IS_DPD_UNDER_120': ['mean', 'sum'],
            'INS_IS_DPD_OVER_120': ['mean', 'sum']}

        ins_d365_agg = ins_d365_grp.agg(ins_d365_agg_dict)
        ins_d365_agg.columns = ['INS_D365' + ('_').join(column).upper() for column in ins_d365_agg.columns.ravel()]

        ins_agg = ins_agg.merge(ins_d365_agg, on='SK_ID_CURR', how='left')

        print('"Installments Payments" final shape:', ins_agg.shape)
        return ins_agg


    def credit_card(self, cc):    
    
        fe=FeaturePipeline()
        cc, cat_cols = fe.one_hot_encoder(cc, nan_as_category=True)

        # Amount used from limit
        cc['LIMIT_USE'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
        # Current payment / Min payment
        cc['PAYMENT_DIV_MIN'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY']
        # Late payment <-- 'CARD_IS_DPD'
        cc['LATE_PAYMENT'] = cc['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
        # How much drawing of limit
        cc['DRAWING_LIMIT_RATIO'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']

        cc['CARD_IS_DPD_UNDER_120'] = cc['SK_DPD'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
        cc['CARD_IS_DPD_OVER_120'] = cc['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)

        # General aggregations
        cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

        # Count credit card lines
        cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

        # Last month balance of each credit card application
        last_ids = cc.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()
        last_months_df = cc[cc.index.isin(last_ids)]
        cc_agg = self.group_and_merge(last_months_df,cc_agg,'CC_LAST_', {'AMT_BALANCE': ['mean', 'max']})

        CREDIT_CARD_TIME_AGG = {
            'AMT_BALANCE': ['mean', 'max'],
            'LIMIT_USE': ['max', 'mean'],
            'AMT_CREDIT_LIMIT_ACTUAL':['max'],
            'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
            'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
            'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['max','sum'],
            'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
            'CNT_DRAWINGS_ATM_CURRENT': ['max','sum', 'mean'],
            'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
            'CNT_DRAWINGS_POS_CURRENT': ['mean'],
            'SK_DPD': ['mean', 'max', 'sum'],
            'LIMIT_USE': ['min', 'max'],
            'DRAWING_LIMIT_RATIO': ['min', 'max'],
            'LATE_PAYMENT': ['mean', 'sum'],
            'CARD_IS_DPD_UNDER_120': ['mean', 'sum'],
            'CARD_IS_DPD_OVER_120': ['mean', 'sum']
        }

        for months in [12, 24, 48]:
            cc_prev_id = cc[cc['MONTHS_BALANCE'] >= -months]['SK_ID_PREV'].unique()
            cc_recent = cc[cc['SK_ID_PREV'].isin(cc_prev_id)]
            prefix = 'INS_{}M_'.format(months)
            cc_agg = self.group_and_merge(cc_recent, cc_agg, prefix, CREDIT_CARD_TIME_AGG)


        print('"Credit Card Balance" final shape:', cc_agg.shape)
        return cc_agg
    

    def merge(self, df, b, cash, ins, cc, pa):

        data = df.merge(b, how='left', on='SK_ID_CURR')
        data = data.merge(cash, how='left', on='SK_ID_CURR')
        data = data.merge(ins, how='left', on='SK_ID_CURR')
        data = data.merge(cc, how='left', on='SK_ID_CURR')
        data = data.merge(pa, how='left', on='SK_ID_CURR')

        return data