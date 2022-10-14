import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# 建立每個子目錄路徑
main_raw_data_path = "raw_data/IDH data（分析用）"
each_patient_path = [i for i in os.listdir(main_raw_data_path) if i not in ["Patient_data", "Dialysis_data"]]

try:
    os.mkdir(os.path.join(main_raw_data_path, "Patient_data"))
except:
    print("已經擁有 Patient 資料夾")
    
try:
    os.mkdir(os.path.join(main_raw_data_path, "Dialysis_data"))
except:
    print("已經擁有 Dialysis 資料夾")

# 針對每個病人目錄進行資料合併
for one_patient_path in tqdm(each_patient_path):
    
    patient_ID = one_patient_path.split(" ")[-1]
    
    one_patient_data_list = os.listdir( os.path.join(main_raw_data_path, one_patient_path) )
    
    # 把透析資料與病人資料分開來
    dialysis_machine_data = [i for i in one_patient_data_list if "Patient" not in i]
    patient_data = [i for i in one_patient_data_list if "Patient" in i]
    
    # 把透析資料全部讀取並合併
    all_dialysis_data = pd.concat([
        pd.read_excel( os.path.join(main_raw_data_path, one_patient_path, i), sheet_name = "Pressure" ) for i in dialysis_machine_data
    ], axis = 0).reset_index(drop = True)
    original_columns = all_dialysis_data.columns.tolist()
    all_dialysis_data["Patient"] = patient_ID
    all_dialysis_data = all_dialysis_data[["Patient"] + original_columns]
    all_dialysis_data.to_excel( os.path.join(main_raw_data_path, "Dialysis_data/Dialysis_data_Patient-{}.xlsx".format(patient_ID)), index = None )
    
    # 讀取病人基本資料，把病人 ID 輸入進去        
    vital_sign_data = pd.read_excel( os.path.join(main_raw_data_path, one_patient_path, patient_data[0]), sheet_name = "Vital signs")
    measure_data = pd.read_excel( os.path.join(main_raw_data_path, one_patient_path, patient_data[0]), sheet_name = "Lab" )
    vital_sign_data["Patient"] = patient_ID
    measure_data["Patient"] = patient_ID
    vital_sign_data = vital_sign_data[vital_sign_data.columns.tolist()[-1:] + vital_sign_data.columns.tolist()[:-1]]
    measure_data = measure_data[measure_data.columns.tolist()[-1:] + measure_data.columns.tolist()[:-1]]
    
    writer = pd.ExcelWriter(os.path.join(main_raw_data_path, "Patient_data/Patient_data_{}.xlsx".format(patient_ID)))
    vital_sign_data.to_excel(writer, sheet_name = "Vital signs", index = None)
    measure_data.to_excel(writer, sheet_name = "Lab", index = None)
    writer.close()
    
# 讀取彙整的資料夾，等會兒要把所有檔案合併成三個：Vital signs、Lab、Dialysis
patient_data_list = os.listdir(os.path.join(main_raw_data_path, "Patient_data"))
dialysis_data_list = os.listdir(os.path.join(main_raw_data_path, "Dialysis_data"))

vital_signs_data = pd.concat([
    pd.read_excel(os.path.join(main_raw_data_path, "Patient_data", i), sheet_name = "Vital signs") for i in patient_data_list
], axis = 0)

lab_data = pd.concat([
    pd.read_excel(os.path.join(main_raw_data_path, "Patient_data", i), sheet_name = "Lab") for i in patient_data_list
], axis = 0)

dialysis_data = pd.concat([
    pd.read_excel(os.path.join(main_raw_data_path, "Dialysis_data", i)) for i in dialysis_data_list
], axis = 0)

for file_name, file in zip(["Vital_signs", "Lab", "Dialysis"], [vital_signs_data, lab_data, dialysis_data]):
    file.to_excel(os.path.join(main_raw_data_path, "{}.xlsx".format(file_name)), index = None)