#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Classification
# In this script, we will try to look at
# the inference part of the heart disease classification solution

# ### Import Modules
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yaml
import os
import json
import joblib
import numpy as np

params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")

print('schema_path', schema_path)


class NotInRange(Exception):
    def __init__(self, message="values entered are not in range"):
        self.message = message
        super().__init__(self.message)


class NotInCols(Exception):
    def __init__(self, message="Cols not present"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


# ### Apply Same Pre-processing

# apply same pre-processing and feature engineering techniques
# as applied during the training process


def encode_features(df, features):
    '''
    Method for one-hot encoding all selected categorical fields
    Input: The method takes pandas dataframe and
    list of the feature names as input
    Output: Returns a dataframe with one-hot encoded features
    Example usage:
    one_hot_encoded_df = encode_features(dataframe, list_features_to_encode)
    '''
    # Implement these steps to prevent dimension mismatch during inference
    encoded_df = pd.DataFrame(columns=['age', 'sex', 'resting_bp',
                                       'cholestoral', 'fasting_blood_sugar',
                                       'max_hr', 'exang', 'oldpeak',
                                       'num_major_vessels', 'thal_0', 'thal_1',
                                       'thal_2', 'thal_3', 'slope_0',
                                       'slope_1', 'slope_2',
                                       'chest_pain_type_0',
                                       'chest_pain_type_1',
                                       'chest_pain_type_2',
                                       'chest_pain_type_3', 'restecg_0',
                                       'restecg_1', 'restecg_2'])
    placeholder_df = pd.DataFrame()
    
    #df= df.tolist()
    #print(df)
    original_df = pd.DataFrame(data=df, columns=['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholestoral',
                'fasting_blood_sugar', 'restecg', 'max_hr', 'exang', 'oldpeak', 'slope',
                'num_major_vessels', 'thal'])

    #print(original_df)
    # One-Hot Encoding using get_dummies for the specified categorical features
    for f in features:
        if(f in original_df.columns):
            encoded = pd.get_dummies(original_df[f])
            encoded = encoded.add_prefix(f + '_')
            placeholder_df = pd.concat([placeholder_df, encoded], axis=1)
        else:
            print('Feature not found')
            return original_df

    # Implement these steps to prevent dimension mismatch during inference
    for feature in encoded_df.columns:
        if feature in original_df.columns:
            encoded_df[feature] = original_df[feature]
        if feature in placeholder_df.columns:
            encoded_df[feature] = placeholder_df[feature]
    # fill all null values
    encoded_df.fillna(0, inplace=True)
    #print('encoded_df', encoded_df)

    return encoded_df


def normalize_data(df):
    '''
    Normalize data using Min-Max Scaler
    Input: The method takes pandas dataframe as input
    Output: Returns a dataframe with normalized features
    Example usage:
    normalized_df = normalize_data(df)
    '''
    val = df.values
    #val = 
    print(val)
    #min_max_normalizer = MinMaxScaler()
    #norm_val = min_max_normalizer.fit_transform(val)
    #df2 = pd.DataFrame(norm_val)
    df2 = pd.DataFrame(val)

    #print('df2', df2)
    return df2


def apply_pre_processing(data):
    '''
    Normalize data using Min-Max Scaler
    Input: The method takes pandas dataframe as input
    Output: Returns a dataframe with normalized features
    Example usage:
    normalized_df = normalize_data(df)
    '''
    features_to_encode = ['thal', 'slope', 'chest_pain_type', 'restecg']
    print('inside pre process')
    encoded = encode_features(data, features_to_encode)
    #print(encoded)
    processed_data = normalize_data(encoded)
    #print(processed_data)
    # Please note this is fabricated inference data,
    # so just taking a small sample size
    return processed_data



def predict(processed_inference_data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    scaler_dir_path = config["scaler_dir"]
    model= joblib.load(model_dir_path)
    scaler= joblib.load(scaler_dir_path)
    processed_inference_data= scaler.transform(processed_inference_data)
    print(processed_inference_data)
    #processed_inference_data= scaler.min_max_normalizer(processed_inference_data)
    prediction= model.predict(processed_inference_data)
    print(prediction[0])

    try:
        if prediction[0] == 1:
            print('High chance of heart disease')
            result= 'High chance of heart disease'
            return result
        elif prediction[0] == 0:
            result = 'You have a healthy heart'
            return result
        else:
            raise NotInRange
    except NotInRange:
        return "Unexpected result"


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
        #print(schema)
    return schema


def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        #print('actual_cols', actual_cols)
        if col not in actual_cols:
            raise NotInCols

    def _validate_values(col, val):
        schema = get_schema()
        #print('schema', schema)

        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]) :
            #print('values not in range')
            raise NotInRange

    for col, val in dict_request.items():
        #print(col)
        _validate_cols(col)
        _validate_values(col, val)

    return True


def form_response(dict_request):
    if validate_input(dict_request):
        data = dict_request.values()
        data = [list(map(float, data))]
        #print(data)
        processed_inference_data = apply_pre_processing(data)
        response = predict(processed_inference_data)
        return response


def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = np.array([list(dict_request.values())])
            processed_inference_data = apply_pre_processing(data)
            print(processed_inference_data)
            response = predict(processed_inference_data)
            response = {"response": response}
            return response

    except NotInRange as e:
        response = {"the_expected_range": get_schema(), "response": str(e)}
        return response

    except NotInCols as e:
        response = {"the_expected_cols": get_schema().keys(), "response": str(e)}
        return response


    except Exception as e:
        response = {"response": str(e)}
        return response
