import os
import gzip
import pickle
import itertools
import numpy as np
import pandas as pd

import ML_two_class
from FT_D_Pipeline import *

# 定義許多變數
PK = "Patient"
standardization_list = [None, "standardization", "normalization", "min-max_scaler"]
decomposition_list = [None, "PCA", "KernelPCA", "IPCA"] 
feature_selection_method_list = [None, "SFS", "SBS", "SFFS", "SFBS", "RFECV"]
mergeData_list = ["Merge-3"]
split_id_list = ["Split-1"]
metaData_list = ["MetaData-55"]

ML_totalResult, FI_totalResult, HT_totalResult, PI_totalResult = list(), list(), list(), list()
for one_metaData, one_mergeData, one_splitData in zip(metaData_list, mergeData_list, split_id_list):
    
    # 輸入原始資料
    raw_data = pd.read_pickle(os.path.join("preprocessed_data", f"Feature_Engineer_{one_metaData}.gzip"), "gzip")
    ID_split = pd.read_pickle(os.path.join("preprocessed_data", f"ID_split_{one_mergeData}-{one_splitData}.gzip"), "gzip")
    target = "Predict_IDH" if "Predict_IDH" in raw_data.columns else "IDH"
    Time_column = [i for i in raw_data.columns if "Time" in i or "time" in i]
    
    for standardization_method, decomposition_method, feature_selection_method, data_id in itertools.product(standardization_list[1:], 
                                                                                                             decomposition_list[1:], 
                                                                                                             feature_selection_method_list,
                                                                                                             range(1, 31, 1)): 
        inputFeatures = [i for i in raw_data.columns if i not in [PK, *Time_column, target]]
        trainData = raw_data.loc[ID_split.query("data_id == @data_id and split == 'train'")["id_list"].iloc[0], [*inputFeatures, target]].reset_index(drop = True)
        valiData = raw_data.loc[ID_split.query("data_id == @data_id and split == 'vali'")["id_list"].iloc[0], [*inputFeatures, target]].reset_index(drop = True)
        testData = raw_data.loc[ID_split.query("data_id == @data_id and split == 'test'")["id_list"].iloc[0], [*inputFeatures, target]].reset_index(drop = True)
        ml_pipeline_obj = ML_Pipeline(ml_methods = [str(standardization_method), str(decomposition_method)], inputFeatures = inputFeatures, target = target)
        ml_pipeline_obj.fit_Pipeline(trainData, decomposition_result_file_name = "test.xlsx")
        trainData, valiData, testData = [
            ml_pipeline_obj.transform_Pipeline(one_data) for one_data in [trainData, valiData, testData]
        ]
        with gzip.GzipFile(os.path.join("ML_Flow_Obj", f"ML_Pipeline_{one_mergeData}_{one_splitData}_{standardization_method}_{decomposition_method}_{feature_selection_method}_{data_id}.gzip"), "wb") as f:
            pickle.dump(ml_pipeline_obj, f)
        inputFeatures = [i for i in trainData.columns if i != target]

        # 模型訓練
        totalResult = ML_two_class.model_fit(data_id = data_id,
                                            trainData = trainData,
                                            valiData = valiData,
                                            testData = testData,
                                            input_features = inputFeatures,
                                            target_label = target,
                                            target_type = "classification",
                                            main_metric = "f1",
                                            feature_selection_method = feature_selection_method,
                                            model_file_name = None) 
        basic_result = {
            "Data_ID": data_id,
            "MetaData_ID": one_metaData,
            "MergeData_ID": one_mergeData,
            "Split_ID": one_splitData,
            "Standardization": standardization_method,
            "Decomposition": decomposition_method,
            "FeatureSelection": feature_selection_method,
        }
        ML_totalResult.extend([{**basic_result, **i} for i in totalResult[0]])
        FI_totalResult.extend([{**basic_result, **i} for i in totalResult[1]])
        HT_totalResult.extend([{**basic_result, **i} for i in totalResult[2]])
        PI_totalResult.extend([{**basic_result, **totalResult[3]}])


        writer = pd.ExcelWriter("result/ML-result-test.xlsx")
        pd.DataFrame(ML_totalResult).to_excel(writer, index = None, sheet_name = "Model_Evaluation")
        pd.DataFrame(FI_totalResult).to_excel(writer, index = None, sheet_name = "Permutation Importance")
        pd.DataFrame(HT_totalResult).to_excel(writer, index = None, sheet_name = "Hyperparameter Tuning")
        writer.close() 

        # pd.DataFrame(ML_totalResult).to_pickle(os.path.join("result", "ML-result-test-Model_Evaluation.gzip"), "gzip")
        # pd.DataFrame(FI_totalResult).to_pickle(os.path.join("result", "ML-result-test-Permutation Importance.gzip"), "gzip")
        # pd.DataFrame(HT_totalResult).to_pickle(os.path.join("result", "ML-result-test-Hyperparameter Tuning.gzip"), "gzip")

        # # 建立一個專屬於 PI 存放的資料夾
        # if os.path.exists("PI_plot") == False :
        #     os.mkdir("PI_plot")

        # # 把所有 PI 圖片存放到資料夾中
        # pd.DataFrame(PI_totalResult).to_pickle("PI_plot/ML-result-test.xlsx")
        break
    break