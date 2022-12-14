# read data from data source
# save it in the data/raw for further processing

from make_dataset import read_params, get_data
import argparse

def load_and_save(config_path):
     config= read_params(config_path)
     df= get_data(config_path)
     new_cols= [col for col in df.columns]
     raw_data_path= config["load_data"]["raw_dataset_csv"]
     df.to_csv(raw_data_path, sep=",", header=new_cols, index=False)
     #print(df.head())

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_arg = args.parse_args()
    load_and_save(config_path=parsed_arg.config)

