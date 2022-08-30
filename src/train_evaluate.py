# load train and test
# train your algorithm
# save metrics and params

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from make_dataset import read_params
import argparse
import joblib
import json
import mlflow

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
    model_dir= config["model_dir"]
    alpha= config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio= config["estimators"]["ElasticNet"]["params"]["l1_ratio"]
    target= [config["base"]["target_col"]]
    #print(target)

    train= pd.read_csv(train_data_path, sep= ",")
    test= pd.read_csv(test_data_path, sep= ",")

    train_y= train[target]
    test_y= test[target]

    test_X= test.drop(target, axis= 1)
    train_X= train.drop(target, axis= 1)

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

#    experiment_id = mlflow.create_experiment(mlflow_config["experiment_name"])
#    print('experiment id: ', experiment_id)

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        lr= ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        lr.fit(train_X, train_y)

        predicted_qualities= lr.predict(test_X)
        (rmse, mae, r2)= eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

#        scores_file = config["reports"]["scores"]
#        params_file = config["reports"]["params"]

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store= urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr,
                "model",
                registered_model_name=mlflow_config["registered_model_name"])

        else:
            mlflow.sklearn.load_model(lr, "model")


#        with open(scores_file, "w") as f:
#            scores = {
##                "rmse": rmse,
#                "mae": mae,
#                "r2": r2
#            }
#            json.dump(scores, f, indent=4)

#        with open(params_file, "w") as f:
#            params = {
#                "alpha": alpha,
#                "l1_ratio": l1_ratio,
#            }
#            json.dump(params, f, indent=4)

#        os.makedirs(model_dir, exist_ok=True)
#        model_path = os.path.join(model_dir, "model.joblib")

#        joblib.dump(lr, model_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_arg = args.parse_args()
    train_and_evaluate(config_path=parsed_arg.config)
