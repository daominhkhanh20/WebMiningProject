import os
import glob
import json


def get_config_architecture(model_path):
    model_path = os.path.abspath(model_path)
    architecture_files = glob.glob(model_path + "/*architecture.json")
    if len(architecture_files) == 0:
        raise Exception(f"File config architecture in {model_path} not found")
    else:
        file_config = architecture_files[0]
        with open(f"{file_config}", "r") as file:
            config_architecture = json.load(file)

        return config_architecture

def save_json(data, file_name):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)
