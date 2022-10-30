import os
import pickle
import numpy as np
import pandas as pd
from datetime import timedelta
import itertools

"""
撰寫 Prediction Flow 的重點：把輸入模型的資料的模樣弄得跟訓練資料一模一樣
Step1. 輸入 Vital Signs、Dialysis Data 
Step2. 開啟預測結果欄位
Step3. 確認是否有遺失值（尤其是 Heart Rate），如 Heart Rate 無資料則該預測無效；如 Dialysis Data 有遺失值則進行補值
Step4. 以時間序列觀點進行資料合併 
Step5. 取序列的統計量 
Step6. 放入模型中預測 
Step7. 輸入預測結果的 Excel 表 
"""

### Step1. 輸入 Vital Signs、Dialysis Data ###
raw_vital_signs_data = pd.read_excel("raw_data/sample_prediction/Sample_Vital_Signs.xlsx")
raw_dialysis_data = pd.read_excel("raw_data/sample_prediction/Sample_Dialysis_Data.xlsx")
### Step1. 輸入 Vital Signs、Dialysis Data ###

### Step2. 開啟預測結果欄位 ###
raw_vital_signs_data["Prediction_IDH"] = np.nan
### Step2. 開啟預測結果欄位 ###

### Step3. 確認是否有遺失值（尤其是 Heart Rate），如 Heart Rate 無資料則該預測無效，如所有資料皆有缺漏，則直接結束程式 ###
raw_vital_signs_data["Prediction_IDH"] = raw_vital_signs_data.apply(lambda x: "無法預測" if x.isna().sum() > 1 else np.nan, axis = 1)
if raw_vital_signs_data["Prediction_IDH"].isna().sum() == 0:
    writer = pd.ExcelWriter("result/Sample_Prediction_Result.xlsx")
    raw_vital_signs_data.to_excel(writer, index = None)
    writer.close()
### Step3. 確認是否有遺失值（尤其是 Heart Rate），如 Heart Rate 無資料則該預測無效，如所有資料皆有缺漏，則直接結束程式 ###

### Step4. 以時間序列觀點進行資料合併 ###
# 定義出每筆資料應該要用哪個時間區段當作訓練資料
one_hour_timedelta = timedelta(hours = 1)
raw_vital_signs_data = raw_vital_signs_data.rename(columns = {"Time": "end_of_the_time_of_data"})
raw_vital_signs_data["Predict_Time"] = raw_vital_signs_data["end_of_the_time_of_data"].apply(lambda x: x + one_hour_timedelta)
raw_vital_signs_data["start_of_the_time_of_data"] = raw_vital_signs_data["end_of_the_time_of_data"].apply(lambda x: x - one_hour_timedelta)

def match_data(start_of_the_time, end_of_the_time):
    select_pressure_data = raw_dialysis_data.copy()
    select_pressure_data = select_pressure_data[(select_pressure_data["time"] >= start_of_the_time) & (select_pressure_data["time"] <= end_of_the_time)]
    # select_pressure_data["time"] = select_pressure_data["time"].apply(lambda x: str(x))
    select_pressure_data = select_pressure_data.dropna()
    
    if select_pressure_data.shape[0] > 0:
        return select_pressure_data.to_dict("list")
    else:
        # print("{} 病人沒有任何序列".format(one_patient))
        return np.nan
    
raw_vital_signs_data.loc[:, "Match_Pressure"] = raw_vital_signs_data.apply(lambda x: match_data(start_of_the_time = x["start_of_the_time_of_data"], 
                                                                                                end_of_the_time = x["end_of_the_time_of_data"]), axis = 1)      

raw_vital_signs_data = raw_vital_signs_data[raw_vital_signs_data["Match_Pressure"].isna() == False].reset_index(drop = True)

# 把合併後的 Dialysis Data 與原先的 Vital Signs 資料合併
raw_vital_signs_data = pd.concat([
    raw_vital_signs_data, pd.DataFrame(raw_vital_signs_data["Match_Pressure"].tolist())
], axis = 1).drop(columns = ["Match_Pressure"])
### Step4. 以時間序列觀點進行資料合併 ###

### Step5. 取序列的統計量 ###
# 定義一個 Function，經序列資料進行統計量計算
Dialysis_features = ["Arterial", "Venous", "PF", "PU", "TMP"]

def compute_sequence_statistics(one_sequence: list,
                                methods = "mean",
                                diff_number: int = 1):
    
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
            if methods == "diff_mean": 
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

for one_dialysis_feature in Dialysis_features:
    for statistics in ["mean", "std", "max", "min", "median"]:
        raw_vital_signs_data["{}_{}".format(one_dialysis_feature, statistics)] = raw_vital_signs_data[one_dialysis_feature].apply(lambda x:\
            compute_sequence_statistics(one_sequence = x, methods = statistics))
        
    for diff_statistics, diff_number in itertools.product(["mean", "std", "max", "min", "median"], list(range(2, 4, 1))):
        raw_vital_signs_data["{}_diff_{}_{}".format(one_dialysis_feature, diff_number, diff_statistics)] = raw_vital_signs_data[one_dialysis_feature].apply(lambda x:\
            compute_sequence_statistics(one_sequence = x, methods = "diff_{}".format(diff_statistics), diff_number = diff_number))
    raw_vital_signs_data["{}_sequence_length".format(one_dialysis_feature)] = raw_vital_signs_data[one_dialysis_feature].apply(lambda x: x.__len__())    
### Step5. 取序列的統計量 ###

### Step6. 放入模型中預測 ###
target = "Prediction_IDH"
time_list = ["Predict_Time", "end_of_the_time_of_data", "start_of_the_time_of_data", "time"]
inputFeatures = [i for i in raw_vital_signs_data.columns if i not in [target]+time_list+Dialysis_features]

with open(os.path.join("Final_Model", "model_2022-10-30.pkl"), "rb") as f:
    model = pickle.load(f)

yhat = model.predict(raw_vital_signs_data[inputFeatures])
raw_vital_signs_data = raw_vital_signs_data[["end_of_the_time_of_data", "Predict_Time", "Heart Rate", target]].copy()
raw_vital_signs_data[target] = [{0: "沒有IDH", 1:"IDH"}[i] for i in yhat]
### Step6. 放入模型中預測 ### 


writer = pd.ExcelWriter("result/Sample_Prediction_Result.xlsx")
raw_vital_signs_data.to_excel(writer, index = None)
writer.close()