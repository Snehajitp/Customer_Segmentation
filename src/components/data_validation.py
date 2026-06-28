import sys
import pandas as pd
from pandas import DataFrame

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataValidationConfig

from src.exception import CustomerException
from src.logger import logging


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):

        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise CustomerException(e, sys)

    def validate_schema_columns(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
    ) -> bool:
        """
        Validate that train and test have identical columns.
        """

        try:

            train_columns = train_df.columns.tolist()
            test_columns = test_df.columns.tolist()

            if train_columns != test_columns:
                logging.error("Train and Test columns do not match.")
                logging.error(f"Train Columns: {train_columns}")
                logging.error(f"Test Columns : {test_columns}")
                return False

            if train_df.empty or test_df.empty:
                logging.error("Train/Test dataset is empty.")
                return False

            logging.info("Data Validation Passed.")

            return True

        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:

        try:

            logging.info("Starting Data Validation")

            train_df = self.read_data(
                self.data_ingestion_artifact.trained_file_path
            )

            test_df = self.read_data(
                self.data_ingestion_artifact.test_file_path
            )

            validation_status = self.validate_schema_columns(
                train_df,
                test_df,
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path="",
                invalid_test_file_path="",
                drift_report_file_path="",
            )

            logging.info(
                f"Validation Status : {validation_status}"
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomerException(e, sys)