import os
import zipfile
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# 輸入資料
raw_data = pd.read_pickle("preprocessed_data/Merge_Vital_signs_and_Dialysis-Merge-3.gzip", "gzip") 

# 定義變數
PK = "Patient"
target = "IDH" 

# 執行資料切割
def split_train_vali_test_data(data_id, data, stratify: str, test_from_all_size = 0.2, vali_from_train = 0.25):
    
    if stratify == "patient":
        trainData, testData = train_test_split(data, test_size = test_from_all_size, shuffle = True, random_state = data_id, stratify = data[PK])
        trainData, valiData = train_test_split(trainData, test_size = vali_from_train, shuffle = True, random_state = data_id, stratify = trainData[PK]) 
    elif stratify == "IDH":
        trainData, testData = train_test_split(data, test_size = test_from_all_size, shuffle = True, random_state = data_id, stratify = data[target])
        trainData, valiData = train_test_split(trainData, test_size = vali_from_train, shuffle = True, random_state = data_id, stratify = trainData[target]) 
    
    return [
        {"data_id": data_id, "split": "train", "id_list": trainData.index.tolist()},
        {"data_id": data_id, "split": "vali", "id_list": valiData.index.tolist()},
        {"data_id": data_id, "split": "test", "id_list": testData.index.tolist()}
    ]
    
split_result = pd.DataFrame(
    [i for one_data_id in range(1, 31, 1) for i in split_train_vali_test_data(data_id = one_data_id, 
                                                                              data = raw_data, 
                                                                              stratify = "IDH")]
)

split_result.to_excel("preprocessed_data/ID_split_Merge-3-Split-2.xlsx", index = None)
split_result.to_pickle("preprocessed_data/ID_split_Merge-3-Split-2.gzip", "gzip") 

