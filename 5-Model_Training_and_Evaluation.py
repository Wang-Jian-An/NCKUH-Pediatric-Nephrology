import os
import gzip
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import ML_two_class_for_autogluon

# 輸入原始資料
with gzip.GzipFile("preprocess_data/Feature_Engineer_Data-Flow-2.gzip", "rb") as f:
    raw_data = pickle.load(f)

# 定義許多變數
PK = "Patient"
Time_column = "Time"
target = "IDH"
inputFeatures = [i for i in raw_data.columns if i not in [PK, Time_column, target]]

# 切割資料
def train_vali_test_split(data: pd.DataFrame, stratify_baseline = None):
    
    if stratify_baseline == "Patient":
        trainData, testData = train_test_split(data, test_size = 0.2, shuffle = True, stratify = data["Patient"])
        trainData, valiData = train_test_split(trainData, test_size = 0.25, shuffle = True, stratify = trainData["Patient"])
    else:
        trainData, testData = train_test_split(data, test_size = 0.2, shuffle = True)
        trainData, valiData = train_test_split(trainData, test_size = 0.25, shuffle = True)
    
    trainData = trainData.reset_index(drop = True)
    valiData = valiData.reset_index(drop = True)
    testData = testData.reset_index(drop = True)
    
    return trainData, valiData, testData

ML_totalResult, FI_totalResult = list(), list()
for data_id in range(1, 11, 1):
    trainData, valiData, testData = train_vali_test_split(data = raw_data, stratify_baseline = "Patient")

    # 模型訓練
    totalResult = ML_two_class_for_autogluon.model_fit(
        data_id = data_id,
        trainData = trainData,
        valiData = valiData,
        testData = testData,
        input_features = inputFeatures,
        target_label = target
    )
    
    ML_totalResult.extend([{**{"Data_ID": data_id}, **i} for i in totalResult[0]])
    FI_totalResult.extend([{**{"Data_ID": data_id}, **i} for i in totalResult[1]])

writer = pd.ExcelWriter("result/ML-result-Flow-2.xlsx")
pd.DataFrame(ML_totalResult).to_excel(writer, index = None, sheet_name = "Model_Evaluation")
pd.DataFrame(FI_totalResult).to_excel(writer, index = None, sheet_name = "Permutation Importance")
writer.close()