import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas_multiprocess import multi_process
import tqdm.contrib.itertools

# 輸入資料
with gzip.GzipFile("preprocess_data/Feature_Engineer_Data-Flow-1-11.gzip", "rb") as f:
    raw_data = pickle.load(f)

# 定義特徵
Dialysis_features = ["Arterial", "Venous", "PF", "PU", "TMP"]
related_Dialysis_features = [i for i in raw_data if "diff" in i and "mean" in i]
print(related_Dialysis_features)

# # 描述性統計-計算美個病人 IDH 數量
# plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
# sns.countplot(data = raw_data, x = "IDH", hue = "Patient")
# plt.title("各病人 IDH 次數")
# plt.legend(loc = "center right", bbox_to_anchor = (1.2, 0.5))
# plt.tight_layout()
# plt.savefig("result/各病人IDH次數.png")
# # plt.show()

# 描述性統計-各病人時間序列圖（要依照是否 IDH 標記顏色）
matplotlib.rcParams["figure.max_open_warning"] = 0
plt.rcParams['axes.unicode_minus']=False
for one_patient, one_feature in tqdm.contrib.itertools.product(raw_data["Patient"].unique(), related_Dialysis_features):
    plt.figure(figsize = (12, 7))
    for _, one_data in raw_data.query("Patient == @one_patient").iterrows():
        line_style = {0: "dotted", 1: "solid"} 
        # print(one_data[one_feature])
        if type(one_data[one_feature]) != list:
            plt.plot(list(range([one_data[one_feature]].__len__())), [one_data[one_feature]], linestyle = line_style[one_data["IDH"]])
        else:
            plt.plot(list(range(one_data[one_feature].__len__())), one_data[one_feature], linestyle = line_style[one_data["IDH"]])
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
        plt.title("{} 病人的低血壓與 {} 序列特徵概況".format(one_patient, one_feature))
        plt.savefig("plot/{} 病人的低血壓與 {} 序列特徵概況.png".format(one_patient, one_feature))

# # 描述性統計-各病人序列數量之 Histogram
# for one_patient in raw_data["Patient"].unique().tolist():
#     sequence_number = [one_data["time"].__len__() for _, one_data in raw_data.query("Patient == @one_patient").iterrows()]
#     plt.figure(figsize = (8, 6))
#     plt.hist(sequence_number)
#     plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
#     plt.title("{} 病人每次序列數量的直方圖".format(one_patient))
#     plt.savefig("plot/{} 病人每次序列數量的直方圖".format(one_patient))