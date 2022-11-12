import os
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

import tqdm
from pandas_multiprocess import multi_process
import pickle
import gzip

main_raw_data_path = "raw_data/IDH data（分析用）" 

# 輸入重要資訊與透析資料
vital_signs_data = pd.read_excel(os.path.join(main_raw_data_path, "Vital_signs.xlsx"))
dialysis_data = pd.read_excel(os.path.join(main_raw_data_path, "Dialysis.xlsx"))

def match_data(select_pressure_data: pd.DataFrame, start_of_the_time, end_of_the_time):
    """
    本函數目的：挑選出特定時間的 Dialysis Data
    """
    select_pressure_data = select_pressure_data[(select_pressure_data["time"] >= start_of_the_time) & (select_pressure_data["time"] <= end_of_the_time)].dropna()
    
    if select_pressure_data.shape[0] > 0:
        return select_pressure_data.drop(columns = ["Patient"]).to_dict("list")
    else:
        return np.nan 

# 定義出每筆資料應該要用哪個時間區段當作訓練資料
one_hour_timedelta = timedelta(hours = 1) 
vital_signs_data = vital_signs_data.rename(columns = {"Time": "Predict_Time"})
vital_signs_data["end_of_the_time_of_data"] = vital_signs_data["Predict_Time"].apply(lambda x: x - one_hour_timedelta)
vital_signs_data["start_of_the_time_of_data"] = vital_signs_data["end_of_the_time_of_data"].apply(lambda x: x - one_hour_timedelta)

# ### Vital Signs 中其中一筆作為預測資料、前一筆為訓練資料 ###
# totalResult = list()
# for one_patient in vital_signs_data["Patient"].unique().tolist():
#     select_vital_signs = vital_signs_data.query("Patient == @one_patient").rename(columns = {"Time": "end_of_the_time_of_data"}).reset_index(drop = True)
#     select_vital_signs["start_of_the_time_of_data"] = select_vital_signs["end_of_the_time_of_data"].apply(lambda x: x - one_hour_timedelta)
#     print(select_vital_signs["end_of_the_time_of_data"].tolist()[:-1])
#     predict_data_dict = {
#         "Predict_Time": select_vital_signs["end_of_the_time_of_data"].tolist()[1:] + [np.nan],
#         "Predict_IDH": select_vital_signs["IDH"].tolist()[1:] + [np.nan]
#     }
#     select_vital_signs = {**select_vital_signs.to_dict("list"), **predict_data_dict}
#     totalResult.append(pd.DataFrame(select_vital_signs).query("Predict_Time.notnull()"))
# vital_signs_data = pd.concat(totalResult, axis = 0)
# ### Vital Signs 中其中一筆作為預測資料、前一筆為訓練資料 ### 


### 針對每位病患個別進行合併 ###
total_vital_signs_data = list()
for one_patient in tqdm.tqdm(vital_signs_data["Patient"].unique().tolist()):
    select_vital_signs_data = vital_signs_data.query("Patient == @one_patient").reset_index(drop = True)
    select_pressure_data = dialysis_data.query("Patient == @one_patient")
    
    match_pressure = [
        match_data(select_pressure_data = select_pressure_data, 
                   start_of_the_time = one_vital_signs_data["start_of_the_time_of_data"],
                   end_of_the_time = one_vital_signs_data["end_of_the_time_of_data"]) for _, one_vital_signs_data in select_vital_signs_data.iterrows()
    ]

    match_pressure = pd.Series(match_pressure).dropna().to_dict() 
  
    # 把合併後的 Dialysis Data 與原先的 Vital Signs 資料合併
    select_vital_signs_data = pd.concat([
        select_vital_signs_data.loc[list(match_pressure.keys()), :], pd.DataFrame(list(match_pressure.values()), index = list(match_pressure.keys()))
    ], axis = 1)
    total_vital_signs_data.append(select_vital_signs_data)
### 針對每位病患個別進行合併 ### 

total_vital_signs_data = pd.concat(total_vital_signs_data, axis = 0).reset_index(drop = True)
total_vital_signs_data.iloc[:50, :].to_excel("preprocess_data/Merge_Vital_signs_and_Dialysis-Merge-1-sample（欲分析者請用gzip檔案）.xlsx", index = None)
total_vital_signs_data.to_pickle("preprocess_data/Merge_Vital_signs_and_Dialysis-Merge-1.gzip", "gzip")