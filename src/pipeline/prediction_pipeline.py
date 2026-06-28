import sys
import pandas as pd

from src.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_COLUMNS,
    FEATURE_DTYPES,
)
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils


class PredictionPipeline:

    def __init__(self):
        try:
            # Load model and preprocessor only once
            self.preprocessor = MainUtils.load_object(PREPROCESSOR_PATH)
            self.model = MainUtils.load_object(MODEL_PATH)

            logging.info("Model and Preprocessor loaded successfully.")

        except Exception as e:
            raise CustomerException(e, sys)

    def prepare_input(self, input_data: list) -> pd.DataFrame:
        """
        Convert user input into a DataFrame with the correct
        column names and datatypes.
        """
        try:
            if len(input_data) != len(FEATURE_COLUMNS):
                raise ValueError(
                    f"Expected {len(FEATURE_COLUMNS)} input values "
                    f"but received {len(input_data)}."
                )

            df = pd.DataFrame(
                [input_data],
                columns=FEATURE_COLUMNS
            )

            # Convert columns to correct datatype
            for column, dtype in FEATURE_DTYPES.items():
                df[column] = df[column].astype(dtype)

            logging.info("Input dataframe created successfully.")

            return df

        except Exception as e:
            raise CustomerException(e, sys)

    def predict(self, input_data: list):
        """
        Predict customer cluster.
        """
        try:
            # Prepare input
            input_df = self.prepare_input(input_data)

            # Apply preprocessing
            transformed_data = self.preprocessor.transform(input_df)

            # Predict
            prediction = self.model.predict(transformed_data)

            logging.info(f"Predicted Cluster: {prediction}")

            return prediction

        except Exception as e:
            raise CustomerException(e, sys)