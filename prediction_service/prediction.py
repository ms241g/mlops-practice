#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease Classification
# In this script, we will try to look at
# the inference part of the heart disease classification solution

# ### Import Modules
import yaml
import os
import json
import joblib
import numpy as np

params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")

#print('schema_path', schema_path)


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


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model= joblib.load(model_dir_path)
    prediction= model.predict(data).tolist()[0]

    try:
        if prediction == 1:
            result= 'High chance of heart disease'
        elif prediction == 0:
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
        _validate_cols(col)
        _validate_values(col, val)

    return True


def form_response(dict_request):
    if validate_input(dict_request):
        data = dict_request.values()
        data = [list(map(float, data))]
        response = predict(data)
        return response


def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = np.array([list(dict_request.values())])
            response = predict(data)
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
