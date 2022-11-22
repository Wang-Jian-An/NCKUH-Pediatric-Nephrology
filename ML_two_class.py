import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.contrib import itertools
from tqdm import tqdm
tqdm.pandas()

from sklearn.metrics import *
from sklearn.inspection import permutation_importance 
from lime import lime_tabular

import warnings
warnings.filterwarnings("ignore")
import two_class_model_evaulation
from ML_Model_Training import model_training_and_hyperparameter_tuning

"""
本程式碼主旨：針對一組資料，進行模型訓練後，進行模型評估
輸入：訓練資料、驗證資料、測試資料、輸入特徵、目標變數
輸出：報表

"""

def model_fit(data_id,
              trainData: pd.DataFrame, 
              valiData: pd.DataFrame, 
              testData: pd.DataFrame, 
              input_features, 
              target_label, 
              target_type,
              main_metric, 
              feature_selection_method = None, 
              hyperparameter_tuning = "bayesopt", 
              feature_importances = "PermutationImportance"):
    
    totalResult = list()
    totalFeatureImportanceResult = list()

    # Step2. 使用 TabularPredictor 進行模型訓練
    model_name_list = ["Random Forest with Entropy", "Random Forest with Gini",
                       "Extra Tree with Entropy", "Extra Tree with Gini", "XGBoost", "LightGBM"]
    predictor = {
        model_name: model_training_and_hyperparameter_tuning(trainData = trainData,
                                                             valiData = valiData,
                                                             inputFeatures = input_features,
                                                             target = target_label,
                                                             target_type = target_type,
                                                             model_name = model_name,
                                                             feature_selection_method = feature_selection_method,
                                                             main_metric = main_metric).model_training() for model_name in model_name_list
    }

    # Step4. 模型評估
    for one_model_name, (set_name, set) in itertools.product(model_name_list, 
                                                             zip(["train", "vali", "test"], 
                                                                 [trainData, valiData, testData])):
        # print(f"Get {one_model_name}, {set_name} evaluation")
        basic_information = {
            "Model": one_model_name,
            "Features": predictor[one_model_name]["Features"],
            "Set": set_name,
            "Number_of_Data": set.shape[0]
        }
        # Step3. 將測試資料放入訓練好的模型作預測
        yhat_test = predictor[one_model_name]["Model"].predict(set[predictor[one_model_name]["Features"]])
        yhat_proba_test = predictor[one_model_name]["Model"].predict_proba(set[predictor[one_model_name]["Features"]])
        one_model_all_score = two_class_model_evaulation.model_evaluation(ytrue = set[target_label],
                                                                        ypred = yhat_test,
                                                                        ypred_proba = yhat_proba_test[:, 1])
        totalResult.append({**basic_information, **one_model_all_score})

        # Step5. 變數重要性
        if feature_importances == "PermutationImportance":
            print(f"Get {one_model_name}, {set_name} feature importances")
            feature_importances_information = {
                "Model": one_model_name,
                "Set": set_name
            }

            feature_importance_result = permutation_importance(predictor[one_model_name]["Model"],
                                                               X = set[predictor[one_model_name]["Features"]], y = set[target_label])
            totalFeatureImportanceResult += [{**feature_importances_information, 
                                              **{"Feature": one_feature, "Importance_Mean": mean, "Importance_Std": std, "Importances": original}}\
                                                  for one_feature, mean, std, original in zip(predictor[one_model_name]["Features"], *[feature_importance_result[i].tolist() for i in feature_importance_result.keys()])]

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