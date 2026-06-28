import os
import sys
import pickle
import yaml
import numpy as np

from src.exception import CustomerException
from src.config import SCHEMA_FILE_PATH


def load_numpy_array_data(file_path: str):
    try:
        with open(file_path, "rb") as file:
            return np.load(file)
    except Exception as e:
        raise CustomerException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            np.save(file, array)

    except Exception as e:
        raise CustomerException(e, sys)


def write_yaml_file(file_path: str, content: dict):
    """
    Write dictionary to yaml file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as yaml_file:
            yaml.dump(content, yaml_file)

    except Exception as e:
        raise CustomerException(e, sys)


class MainUtils:

    @staticmethod
    def save_object(file_path: str, obj: object):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as file:
                pickle.dump(obj, file)

        except Exception as e:
            raise CustomerException(e, sys)

    @staticmethod
    def load_object(file_path: str):
        try:
            with open(file_path, "rb") as file:
                return pickle.load(file)

        except Exception as e:
            raise CustomerException(e, sys)

    @staticmethod
    def read_yaml_file(file_path: str):
        """
        Read any yaml file.
        """
        try:
            with open(file_path, "r") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomerException(e, sys)

    @staticmethod
    def read_schema_config_file():
        """
        Read schema.yaml.
        """
        try:
            return MainUtils.read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomerException(e, sys)

    @staticmethod    
    def save_numpy_array_data(file_path: str, array: np.ndarray):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "wb") as file:
                np.save(file, array)

        except Exception as e:
            raise CustomerException(e, sys)