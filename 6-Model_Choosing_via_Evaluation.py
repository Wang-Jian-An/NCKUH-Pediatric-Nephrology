import os
import pickle
import shutil
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 定義主要評估指標
main_metric = "F1-Score_for_1"

# 匯入模型評估結果
raw_result = pd.read_excel("result/ML-result-20221025.xlsx", sheet_name = "Model_Evaluation")

# 篩選出測試資料且 Model 中有 FULL 的部分
select_result = raw_result.query("(Set == 'test') and (Model.str.contains('FULL'))").reset_index(drop = True)

# 計算每個模型在 10 次模擬後的平均數與標準差
model_list = sorted(select_result["Model"].unique().tolist())
groupby_result_mean = select_result.groupby(by = "Model").mean()[main_metric]
groupby_result_std = select_result.groupby(by = "Model").std()[main_metric]

# 找到主要評估指標中，平均最高且標準差最低的模型
great_model = model_list[np.argmax(groupby_result_mean / groupby_result_std)]

# 找到最好的模型後，根據 Data_id 找到多次中最好的那一次
select_result = select_result.query("Model == @great_model").reset_index(drop = True)
great_data_id = np.argmax(select_result[main_metric])+1

# 把最好的模型檔案複製到最終模型的資料夾中
model_path = os.path.join("AutoML", f"Data_id_{great_data_id}", "models", great_model, "model.pkl")
shutil.copyfile(model_path, os.path.join("Final_Model", "model_2022-10-30.pkl"))