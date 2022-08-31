from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from make_dataset import read_params
import argparse


def ABGridSearch(X, y, param_limits):
    param_grid = {
                  'n_estimators': param_limits[0],
                  'learning_rate': param_limits[1],
                  'algorithm' : param_limits[2]
                 }

    best_adb = GridSearchCV(
                           estimator = AdaBoostClassifier(random_state=7),
                           param_grid=param_grid, 
                           cv=10
                          )
    best_adb.fit(X, y)
    
    print("Best Decision Tree Hyper-Parameters are:")
    print(best_adb.best_params_)
    
    return best_adb.best_params_['n_estimators'], best_adb.best_params_['learning_rate'], best_adb.best_params_['algorithm']


def load_and_tune(config_path):
     config= read_params(config_path)
     #df= get_data(config_path)
     #new_cols= [col for col in df.columns]
     #print(config)
     estimators= config["estimators"]["Adaboost"]["params"]["n_estimators"]
     learning_rate= config["estimators"]["Adaboost"]["params"]["learning_rate"]
     algorithm= config["estimators"]["Adaboost"]["params"]["algorithm"]
     train= pd.read_csv(config["split_data"]["train_path"])
     x_train= train.drop(['target'], axis= 1)
     y_train= train['target']
     best_n_estimators, best_lr, best_algo = ABGridSearch(x_train,  y_train.values.ravel(), 
                                            (estimators, learning_rate, algorithm))

     return best_n_estimators, best_lr, best_algo
     #df.to_csv(raw_data_path, sep=",", header=new_cols, index=False)
     #print(df.head())


