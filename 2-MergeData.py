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

# 定義出每筆資料應該要用哪個時間區段當作訓練資料
one_hour_timedelta = timedelta(hours = 1)
vital_signs_data["end_of_the_time_of_data"] = vital_signs_data["Time"].apply(lambda x: x - one_hour_timedelta)
vital_signs_data["start_of_the_time_of_data"] = vital_signs_data["end_of_the_time_of_data"].apply(lambda x: x - one_hour_timedelta)

### 針對每位病患個別進行合併 ###
def match_data(one_patient, start_of_the_time, end_of_the_time):
    select_pressure_data = dialysis_data.query("Patient == @one_patient").reset_index(drop = True)
    select_pressure_data = select_pressure_data[(select_pressure_data["time"] >= start_of_the_time) & (select_pressure_data["time"] <= end_of_the_time)]
    # select_pressure_data["time"] = select_pressure_data["time"].apply(lambda x: str(x))
    select_pressure_data = select_pressure_data.dropna()
    
    if select_pressure_data.shape[0] > 0:
        return select_pressure_data.drop(columns = ["Patient"]).to_dict("list")
    else:
        # print("{} 病人沒有任何序列".format(one_patient))
        return np.nan

total_vital_signs_data = list()
for one_patient in tqdm.tqdm(vital_signs_data["Patient"].unique().tolist()):
    select_vital_signs_data = vital_signs_data.query("Patient == @one_patient").reset_index(drop = True)
    select_vital_signs_data.loc[:, "Match_Pressure"] = select_vital_signs_data.apply(lambda x: match_data(one_patient = one_patient,
                                                                                                          start_of_the_time = x["start_of_the_time_of_data"], 
                                                                                                          end_of_the_time = x["end_of_the_time_of_data"]), axis = 1)      

    select_vital_signs_data = select_vital_signs_data[select_vital_signs_data["Match_Pressure"].isna() == False].reset_index(drop = True)
  
    # 把合併後的 Dialysis Data 與原先的 Vital Signs 資料合併
    select_vital_signs_data = pd.concat([
        select_vital_signs_data, pd.DataFrame(select_vital_signs_data["Match_Pressure"].tolist())
    ], axis = 1).drop(columns = ["Match_Pressure"])

    total_vital_signs_data.append(select_vital_signs_data)
### 針對每位病患個別進行合併 ###

total_vital_signs_data = pd.concat(total_vital_signs_data, axis = 0)
total_vital_signs_data.to_excel("preprocess_data/Merge_Vital_signs_and_Dialysis（欲分析者請用gzip檔案）.xlsx", index = None)

with gzip.GzipFile("preprocess_data/Merge_Vital_signs_and_Dialysis.gzip", "wb") as f:
    pickle.dump(total_vital_signs_data, f)