import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.contrib import itertools
from tqdm import tqdm
tqdm.pandas()

from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import *
from lime import lime_tabular
from autogluon.tabular import TabularPredictor

import warnings
warnings.filterwarnings("ignore")
import two_class_model_evaulation_DC

"""
本程式碼主旨：針對一組資料，進行模型訓練後，進行模型評估
輸入：訓練資料、驗證資料、測試資料、輸入特徵、目標變數
輸出：報表

"""

def model_fit(trainData, 
              valiData, 
              testData, 
              input_features, 
              target_label, 
              hyperparameter_tuning = "bayesopt", 
              feature_importances = "PermutationImportance"):
    totalResult = list()
    totalFeatureImportanceResult = list()

    # Step2. 使用 TabularPredictor 進行模型訓練
    predictor = TabularPredictor(label = target_label, verbosity = 0, problem_type = "binary", path = "C://RECYCLE")\
                            .fit(train_data = trainData[input_features + [target_label]], 
                                 tuning_data = valiData[input_features + [target_label]], 
                                 hyperparameter_tune_kwargs = hyperparameter_tuning, 
                                 refit_full = True)

    # 把模型名稱取出來
    all_model_list = predictor.get_model_names()

    # Step4. 模型評估
    # ["test", "train", "vali"] [trainData, valiData, testData]
    for one_model_name, (set_name, set) in itertools.product(all_model_list, zip(["train", "vali", "test"], [trainData, valiData, testData])):
        # print(f"Get {one_model_name}, {set_name} evaluation")
        basic_information = {
            "Model": one_model_name,
            "Features": input_features,
            "Set": set_name,
            "Number_of_Data": set.shape[0]
        }
        # Step3. 將測試資料放入訓練好的模型作預測
        yhat_test = predictor.predict(data = set, model = one_model_name)
        yhat_proba_test = predictor.predict_proba(data = set, model = one_model_name)
        one_model_all_score = two_class_model_evaulation_DC.model_evaluation(ytrue = set[target_label],
                                                                        ypred = yhat_test,
                                                                        ypred_proba = yhat_proba_test.values[:, 1])
        totalResult.append({**basic_information, **one_model_all_score})

        # Step5. 變數重要性
        if feature_importances == "PermutationImportance":
            print(f"Get {one_model_name}, {set_name} feature importances")
            feature_importances_information = {
                "Model": [one_model_name] * input_features.__len__(),
                "Set": [set_name] * input_features.__len__()
            }

            feature_importance_result = predictor.feature_importance(
                data = set,
                model = one_model_name,
                features = input_features
            ).reset_index()

            feature_importance_result = pd.concat([
                pd.DataFrame(feature_importances_information), feature_importance_result
            ], axis = 1).to_dict("records")

            totalFeatureImportanceResult += [{**feature_importances_information, **one_feature_importance_information} for one_feature_importance_information in feature_importance_result]

        elif feature_importances == "LIME" and set_name == "test":
            lime_result = list()
            with open(f"E://AutoML//models//{one_model_name}//model.pkl", "rb") as f:
                one_model = pickle.load(f)
            lime_explainer = lime_tabular.LimeTabularExplainer(training_data = trainData[input_features].values,
                                                            training_labels = trainData[target_label].values,
                                                            feature_names = input_features,
                                                            class_names = ["Operation", "Biopsy"])
            for test_index in tqdm(list(testData.index), desc = f"{one_model_name}-LIME"):
                try:
                    exp = lime_explainer.explain_instance(testData[input_features].values[test_index, :], 
                                                        predict_fn = one_model.model.predict_proba)
                    exp.save_to_file(f"LIME_result//{one_model_name}_testData-{test_index}_OrininalResult-{int(testData.loc[test_index, target_label])}_HyperparameterTuning-{hyperparameter_tuning}.html")
                    one_lime_result_dict_list = [{"Model": one_model_name, 
                                                "TestID": test_index,
                                                "TrueResult": int(testData.loc[test_index, target_label]),
                                                "PredResult": one_model.model.predict_proba(testData.loc[test_index, input_features].values.reshape((1, -1))), 
                                                "Information": i[0],
                                                "Value": i[1]} for i in exp.as_list()]    
                    lime_result.extend(one_lime_result_dict_list)
                except: 
                    pass
    
    if feature_importances == "PermutationImportance":
        return totalResult, totalFeatureImportanceResult
    elif feature_importances == "LIME":
        return totalResult, lime_result
    else:
        return totalResult