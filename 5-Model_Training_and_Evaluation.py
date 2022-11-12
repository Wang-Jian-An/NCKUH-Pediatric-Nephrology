import os
import gzip
import pickle
import itertools
import numpy as np
import pandas as pd
import tqdm.contrib.itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer 

import ML_two_class_for_autogluon

# 輸入原始資料
with gzip.GzipFile("preprocess_data/Feature_Engineer_Data-Flow-2.gzip", "rb") as f:
    raw_data = pickle.load(f)
ID_split = pd.read_pickle("preproceseed_data/ID_split_MergeData-4.gzip", "gzip")


# 定義許多變數
PK = "Patient"
Time_column = "Time"
target = "IDH" 
standardization_list = [None, "standardization", "normalization", "min-max_scaler"]
decomposition_list = [None] 
split_data_stratify_list = reversed([None, PK, target, f"{PK}_{target}"])
inputFeatures = [i for i in raw_data.columns if i not in [PK, Time_column, target]]

ML_totalResult, FI_totalResult = list(), list()

for standardization_method, decomposition_method, data_id in itertools.product(standardization_list, decomposition_list, range(1, 31, 1)): 
    trainData = raw_data.loc[ID_split.query("data_id == @data_id and split == @train")["ID_split"].iloc[0], inputFeatures + [target]]
    valiData = raw_data.loc[ID_split.query("data_id == @data_id and split == @vali")["ID_split"].iloc[0], inputFeatures + [target]]
    testData = raw_data.loc[ID_split.query("data_id == @data_id and split == @test")["ID_split"].iloc[0], inputFeatures + [target]] 

    if standardization_method == "standardization":
        ### Standardization ###
        standardization = StandardScaler().fit(trainData[inputFeatures])
        trainData = pd.concat([
            pd.DataFrame(standardization.transform(trainData[inputFeatures]), columns = inputFeatures), trainData[target]
        ], axis = 1)
        valiData = pd.concat([
            pd.DataFrame(standardization.transform(valiData[inputFeatures]), columns = inputFeatures), valiData[target]
        ], axis = 1)
        testData = pd.concat([
            pd.DataFrame(standardization.transform(testData[inputFeatures]), columns = inputFeatures), testData[target]
        ], axis = 1)
        ### Standardization ###
    elif standardization_method == "normalization":
        ### Normalization ###
        normalization = Normalizer().fit(trainData[inputFeatures])
        trainData = pd.concat([
            pd.DataFrame(normalization.transform(trainData[inputFeatures]), columns = inputFeatures), trainData[target]
        ], axis = 1)
        valiData = pd.concat([
            pd.DataFrame(normalization.transform(valiData[inputFeatures]), columns = inputFeatures), valiData[target]
        ], axis = 1)
        testData = pd.concat([
            pd.DataFrame(normalization.transform(testData[inputFeatures]), columns = inputFeatures), testData[target]
        ], axis = 1)
        ### Normalization ###            
    
    # 模型訓練
    totalResult = ML_two_class_for_autogluon.model_fit(
        data_id = data_id,
        trainData = trainData,
        valiData = valiData,
        testData = testData,
        input_features = inputFeatures,
        target_label = target
    )    
    basic_result = {
        "Data_ID": data_id,
        "Split_Data_Stratify": one_split_data_stratify,
        "Standardization": standardization_method,
        "Decomposition": decomposition_method 
    }
    ML_totalResult.extend([{**basic_result, **i} for i in totalResult[0]])
    FI_totalResult.extend([{**basic_result, **i} for i in totalResult[1]])

writer = pd.ExcelWriter("result/ML-result-Flow-6_and_10_and_14_and_18_and_22.xlsx")
pd.DataFrame(ML_totalResult).to_excel(writer, index = None, sheet_name = "Model_Evaluation")
pd.DataFrame(FI_totalResult).to_excel(writer, index = None, sheet_name = "Permutation Importance")
writer.close() 