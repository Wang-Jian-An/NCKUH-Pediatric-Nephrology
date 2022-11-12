import os
import gzip
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 輸入資料
raw_data = pd.read_pickle("preprocess_data/Merge_Vital_signs_and_Dialysis-Merge-2.gzip", "gzip") 

# 定義 Dialysis Data 中所有欄位、欲分析的所有變數與目標
Dialysis_features = ["Arterial", "Venous", "PF", "PU", "TMP"]
target = "Predict_IDH" if "Predict_IDH" in raw_data.columns else "IDH"
inputFeatures = ["Patient", "Predict_Time"] +\
                [i for i in ["end_of_the_time_of_data", "start_of_the_time_of_data"] if target == "Predict_IDH"] +\
                ["Heart Rate" if target == "Predict_IDH" else None] +\
                Dialysis_features 
inputFeatures = [i for i in inputFeatures if i is not None]

# Imputation
def impute_sequence(one_sequence: pd.DataFrame):
    
    """
    輸入的 one_sequence: 要有時間與值
    輸出的 one_sequence: 一個 Series，index 為時間、value 為值
    """
    
    column_name = one_sequence.columns
    if one_sequence.columns.tolist().__len__() == 2:
        one_sequence = one_sequence.set_index(keys = one_sequence[column_name[0]]).drop(columns = column_name[0])
        column_name = column_name.copy()[-1:]     
        pass

    one_sequence = one_sequence[column_name[0]].interpolate(methods = "linear")   
    return one_sequence 

# 定義一個 function，輸入一個病人後，可以完成所有 Imputation 的流程
def imputation_flow(one_patient):
    # 把特定病人所有資料抓出來
    select_raw_data = raw_data.query("Patient == @one_patient").reset_index(drop = True)
    
    if "Heart Rate" in inputFeatures:
        select_raw_data.loc[:, "Heart Rate"] = impute_sequence(one_sequence = select_raw_data[["Time", "Heart Rate"]]).tolist()
        select_raw_data = select_raw_data.query("`Heart Rate`.notnull()") # [select_raw_data["Heart Rate"].isna() == False]

    # 把美筆資料的透析特徵的序列進行 Imputation
    for one_dialysis_features in Dialysis_features:
        select_raw_data.loc[:, one_dialysis_features] = select_raw_data.apply(lambda x:\
            impute_sequence(one_sequence = pd.DataFrame(x[["time", one_dialysis_features]].to_dict() )).tolist() , axis = 1)
    return select_raw_data

imputed_data = pd.concat([
    imputation_flow(one_patient = one_patient) for one_patient in raw_data["Patient"].unique()
], axis = 0)

# 定義一個 Function，經序列資料進行統計量計算
def compute_sequence_statistics(one_sequence: list,
                                methods = "mean",
                                diff_number: int = 1,
                                diff_retain_sequence = True):
    
    if one_sequence.__len__() == 0:
        return 0
    else:
        if methods == "mean": 
            return np.mean(one_sequence)
        elif methods == "std": 
            return np.std(one_sequence)
        elif methods == "max": 
            return np.max(one_sequence)
        elif methods == "min": 
            return np.min(one_sequence)
        elif methods == "median": 
            return np.median(one_sequence)

        if one_sequence.__len__() <= diff_number:
            return 0
        else:
            diff_one_sequence = np.array(one_sequence[diff_number:]) - np.array(one_sequence[:-diff_number])
            if methods == "diff_mean": 
                return np.mean(diff_one_sequence) 
            elif methods == "diff_std": 
                return np.std(diff_one_sequence) 
            elif methods == "diff_max": 
                return np.max(diff_one_sequence) 
            elif methods == "diff_min": 
                return np.min(diff_one_sequence) 
            elif methods == "diff_median": 
                return np.median(diff_one_sequence) 

# 定義一個 Function，把時間序列資料統計量計算寫進去
def ML_feature_generation_flow(): 
    global imputed_data
    """
    整個流程：
    1. 挑選出欲分析的特徵與目標
    2. 把 Dialysis Features 進行差分運算（要額外建立 Function）
    """
    
    imputed_data = imputed_data[inputFeatures+[target]].copy().reset_index(drop = True)
        
    statistics_dict = {
        "{}_{}".format(one_dialysis_feature, statistics): imputed_data[one_dialysis_feature].apply(lambda x:\
            compute_sequence_statistics(one_sequence = x, 
                                        methods = statistics)).tolist() \
                for one_dialysis_feature, statistics in itertools.product(Dialysis_features, 
                                                                          ["mean", "std", "max", "min", "median"])
    }
        
    diff_statistics_dict = {
        "{}_diff_{}_{}".format(one_dialysis_feature, diff_number, diff_statistics): imputed_data[one_dialysis_feature].apply(lambda x:\
            compute_sequence_statistics(one_sequence = x, 
                                        methods = "diff_{}".format(diff_statistics), 
                                        diff_number = diff_number, 
                                        diff_retain_sequence = True)) \
                for one_dialysis_feature, diff_statistics, diff_number in itertools.product(Dialysis_features, 
                                                                                            ["mean", "std", "max", "min", "median"], 
                                                                                            list(range(1, 21, 1)) )
    } 
    
    num_of_seq_dict = {
        "{}_sequence_length".format(one_dialysis_feature): imputed_data[one_dialysis_feature].apply(lambda x: x.__len__()) \
            for one_dialysis_feature in Dialysis_features
    }

    
    imputed_data = pd.concat([
        imputed_data, pd.DataFrame(statistics_dict), pd.DataFrame(diff_statistics_dict), pd.DataFrame(num_of_seq_dict)
    ], axis = 1)
    
    imputed_data = imputed_data.drop(columns = Dialysis_features)
    
    return imputed_data

feature_engineered_data = ML_feature_generation_flow()

writer = pd.ExcelWriter("preprocess_data/Feature_Engineer_Merge2.xlsx") 
feature_engineered_data.iloc[:100, :].to_excel(writer, index = None)
writer.close()

feature_engineered_data.to_pickle("preprocess_data/Feature_Engineer_Merge2.gzip", "gzip") 