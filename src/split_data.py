## Split the raw data and store it in processed folder

import os
import argparse
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from make_dataset import get_data, read_params
from constants import * 
import joblib


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
    # from constants.py
    encoded_df = pd.DataFrame(columns=ONE_HOT_ENCODED_FEATURES)
    placeholder_df = pd.DataFrame()
    # One-Hot Encoding using get_dummies for the specified categorical features
    for f in features:
        if(f in df.columns):
            encoded = pd.get_dummies(df[f])
            encoded = encoded.add_prefix(f + '_')
            placeholder_df = pd.concat([placeholder_df, encoded], axis=1)
        else:
            print('Feature not found')
            return df
    # Implement these steps to prevent dimension mismatch during inference
    for feature in encoded_df.columns:
        if feature in df.columns:
            encoded_df[feature] = df[feature]
        if feature in placeholder_df.columns:
            encoded_df[feature] = placeholder_df[feature]
    # fill all null values
    encoded_df.fillna(0, inplace=True)
    return encoded_df


def normalize_data(df):
    '''
    Normalize data using Min-Max Scaler
    Input: The method takes pandas dataframe as input
    Output: Returns a dataframe with normalized features
    Example usage:
    normalized_df = normalize_data(df)
    '''
    target_col = df['target']
    df_normalize= df.drop(['target'], axis= 1)
    values = df_normalize.values
    min_max_normalizer = preprocessing.MinMaxScaler()
    #norm_val = min_max_normalizer.fit_transform(values)
    min_max_normalizer.fit(values)
    joblib.dump(min_max_normalizer, 'scaler.save')
    norm_val= min_max_normalizer.transform(values)
    norm_df = pd.DataFrame(norm_val)
    final_df= pd.concat([norm_df, target_col], axis=1)
    return final_df


def apply_pre_processing(data):
    '''
    Apply all pre-processing methods together
    Input: The method takes the inference data as pandas dataframe as input
    Output: Returns a dataframe after applying all preprocessing steps
    Example usage:
    processed_data = apply_pre_processing(df)
    '''
    features_to_encode = FEATURES_TO_ENCODE  # from constants.py
    # applying encoded features function
    encoded = encode_features(data, features_to_encode)
    processed_data = normalize_data(encoded)  # applying normalization function
    return processed_data

def split_and_saved_data(config_path):
    config= read_params(config_path)
    test_data_path= config["split_data"]["test_path"]
    train_data_path= config["split_data"]["train_path"]
    raw_data_path= config["load_data"]["raw_dataset_csv"]
    split_ratio= config["split_data"]["test_size"]
    random_state=config["base"]["random_state"]

    df= pd.read_csv(raw_data_path, sep= ',')
    processed_data= apply_pre_processing(df)
    train, test = train_test_split(processed_data, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_data_path, sep= ',', encoding='utf-8', index= False)
    test.to_csv(test_data_path, sep= ',', encoding='utf-8', index= False)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_arg = args.parse_args()
    split_and_saved_data(config_path=parsed_arg.config)
