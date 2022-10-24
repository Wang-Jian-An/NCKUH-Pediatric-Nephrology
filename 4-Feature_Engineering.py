import os
import gzip
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_multiprocess import multi_process

# 輸入資料
with gzip.GzipFile("preprocess_data/Merge_Vital_signs_and_Dialysis.gzip", "rb") as f:
    raw_data = pickle.load(f)
    pass

# 定義 Dialysis Data 中所有欄位、欲分析的所有變數與目標
Dialysis_features = ["Arterial", "Venous", "PF", "PU", "TMP"]
target = "IDH"
inputFeatures = ["Patient", "Time", "Heart Rate"] + Dialysis_features


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
    select_raw_data.loc[:, "Heart Rate"] = impute_sequence(one_sequence = select_raw_data[["Time", "Heart Rate"]]).tolist()
    select_raw_data = select_raw_data[select_raw_data["Heart Rate"].isna() == False]

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
                                diff_number: int = 1):
    
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
    elif methods == "diff_mean":
        diff_one_sequence = np.array(one_sequence[diff_number:]) - np.array(one_sequence[:-diff_number])
        return np.mean(diff_one_sequence)
    elif methods == "diff_std":
        diff_one_sequence = np.array(one_sequence[diff_number:]) - np.array(one_sequence[:-diff_number])
        return np.std(diff_one_sequence)
    elif methods == "diff_max":
        diff_one_sequence = np.array(one_sequence[diff_number:]) - np.array(one_sequence[:-diff_number])
        return np.max(diff_one_sequence)
    elif methods == "diff_min":
        diff_one_sequence = np.array(one_sequence[diff_number:]) - np.array(one_sequence[:-diff_number])
        return np.min(diff_one_sequence)
    elif methods == "diff_median":
        diff_one_sequence = np.array(one_sequence[diff_number:]) - np.array(one_sequence[:-diff_number])
        return np.median(diff_one_sequence)
    return 


# 定義一個 Function，把時間序列資料統計量計算寫進去
def ML_feature_generation_flow():
    global imputed_data
    """
    整個流程：
    1. 挑選出欲分析的特徵與目標
    2. 把 Dialysis Features 進行差分運算（要額外建立 Function）
    """
    
    imputed_data = imputed_data[inputFeatures+[target]].copy()
    
    for one_dialysis_feature in Dialysis_features:
        for statistics in ["mean", "std", "max", "min", "median"]:
            imputed_data["{}_{}".format(one_dialysis_feature, statistics)] = imputed_data[one_dialysis_feature].apply(lambda x:\
                compute_sequence_statistics(one_sequence = x, methods = statistics))
            
        for diff_statistics, diff_number in itertools.product(["mean", "std", "max", "min", "median"], list(range(1, 3, 1))):
            imputed_data["{}_diff_{}_{}".format(one_dialysis_feature, diff_number, diff_statistics)] = imputed_data[one_dialysis_feature].apply(lambda x:\
                compute_sequence_statistics(one_sequence = x, methods = "diff_{}".format(diff_statistics), diff_number = diff_number))           
    imputed_data = imputed_data.drop(columns = Dialysis_features)
    
    return imputed_data

feature_engineered_data = ML_feature_generation_flow()

writer = pd.ExcelWriter("preprocess_data/Feature_Engineer_Data.xlsx")
feature_engineered_data.to_excel(writer, index = None)
writer.close()

with gzip.GzipFile("preprocess_data/Feature_Engineer_Data.gzip", "wb") as f:
    pickle.dump(feature_engineered_data, f)
