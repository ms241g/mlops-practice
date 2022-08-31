# load train and test
# train your algorithm
# save metrics and params

import os
import pandas as pd
import numpy as np
from hyper_tuning import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from make_dataset import read_params
from metrics_visualize import *
import argparse
import joblib
import json
##import mlflow

def eval_metrics(actual, pred):
    rmse= np.sqrt(mean_squared_error(actual, pred))
    mae= mean_absolute_error(actual, pred)
    r2= r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config= read_params(config_path)
    test_data_path= config["split_data"]["test_path"]
    train_data_path= config["split_data"]["train_path"]
    random_state= config["base"]["random_state"]
    target= config["base"]["target_col"]
    model_dir= config["model_dir"]

    #print(target)

    train= pd.read_csv(train_data_path, sep= ",")
    test= pd.read_csv(test_data_path, sep= ",")

    train_y= train[target]
    test_y= test[target]    

    test_X= test.drop(target, axis= 1)
    train_X= train.drop(target, axis= 1)

    best_n_estimators, best_lr, best_algo = load_and_tune(config_path)

    tuned_adaBoost = AdaBoostClassifier(n_estimators=best_n_estimators, 
                                        algorithm = best_algo, 
                                        learning_rate = best_lr, 
                                        random_state=random_state)

    ml_model(tuned_adaBoost, 'ADABoost', 
                    x_train = train_X, 
                    x_valid = test_X, 
                    y_train =  train_y.values.ravel(), 
                    y_valid = test_y.values.ravel(),
                    x_test = None)

    generate_learning_curves(
        model = tuned_adaBoost, 
        model_name = "ADABoost", 
        X = train_X,
        y = train_y.values.ravel(),
        epochs=4)

    model_path= config['webapp_model_dir']
    dump(tuned_adaBoost, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_arg = args.parse_args()
    train_and_evaluate(config_path=parsed_arg.config)
