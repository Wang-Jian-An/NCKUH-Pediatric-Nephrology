import os
import gzip
import pickle
import shutil
import itertools
import numpy as np
import pandas as pd
import tqdm.contrib.itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer 

import ML_two_class



# 定義許多變數
PK = "Patient"
Time_column = "Predict_Time"
target = "IDH" 
standardization_list = [None, "standardization", "normalization", "min-max_scaler"]
decomposition_list = [None] 
feature_selection_method_list = [None, "SFS", "SBS", "SFFS", "SFBS", "RFECV"]
metaData_list = ["MetaData-1", "MetaData-2", "MetaData-3"] * 2
mergeData_list = [*["Merge-1-patient"]*3, *["Merge-1-IDH"]*3]

ML_totalResult, FI_totalResult, HT_totalResult, PI_totalResult = list(), list(), list(), list()
for one_metaData, one_mergeData in zip(metaData_list, mergeData_list):
    
    # 輸入原始資料
    raw_data = pd.read_pickle(os.path.join("preprocessed_data", f"Feature_Engineer_{one_metaData}.gzip"), "gzip")
    ID_split = pd.read_pickle(os.path.join("preprocessed_data", f"ID_split_{one_mergeData}.gzip"), "gzip")
    inputFeatures = [i for i in raw_data.columns if i not in [PK, Time_column, target]]
    
    for standardization_method, decomposition_method, feature_selection_method, data_id in itertools.product(standardization_list[:1], 
                                                                                                             decomposition_list, 
                                                                                                             feature_selection_method_list,
                                                                                                             range(1, 2, 1)): 
        trainData = raw_data.loc[ID_split.query("data_id == @data_id and split == 'train'")["id_list"].iloc[0], [*inputFeatures, target]]
        valiData = raw_data.loc[ID_split.query("data_id == @data_id and split == 'vali'")["id_list"].iloc[0], [*inputFeatures, target]]
        testData = raw_data.loc[ID_split.query("data_id == @data_id and split == 'test'")["id_list"].iloc[0], [*inputFeatures, target]] 

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
        totalResult = ML_two_class.model_fit(data_id = 1,
                                            trainData = trainData,
                                            valiData = valiData,
                                            testData = testData,
                                            input_features = inputFeatures,
                                            target_label = target,
                                            target_type = "classification",
                                            main_metric = "f1",
                                            feature_selection_method = feature_selection_method)
        basic_result = {
            "Data_ID": data_id,
            "MetaData_ID": one_metaData,
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

pd.DataFrame(ML_totalResult).to_pickle(os.path.join("result", "ML-result-test-Model_Evaluation.gzip"), "gzip")
pd.DataFrame(FI_totalResult).to_pickle(os.path.join("result", "ML-result-test-Permutation Importance.gzip"), "gzip")
pd.DataFrame(HT_totalResult).to_pickle(os.path.join("result", "ML-result-test-Hyperparameter Tuning.gzip"), "gzip")

# 建立一個專屬於 PI 存放的資料夾
if os.path.exists("PI_plot") == False :
    os.mkdir("PI_plot")

# 把所有 PI 圖片存放到資料夾中
for one_image_dict in PI_totalResult:
    print(one_image_dict)
    for one_model_name, one_model in list(one_image_dict.items())[5:]:
        if one_model is not None:
            one_model.write_image(os.path.join("PI_plot", "{}_{}_{}_{}_{}_{}.png".format( *[*list(one_image_dict.values())[:5], one_model_name] )))


zip_name = "PI_plot-20221124.zip"
folder_name = "PI_plot"
shutil.make_archive(zip_name, "zip", folder_name)    