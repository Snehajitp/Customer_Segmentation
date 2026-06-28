import os
import sys
from typing import Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.data_access.customer_data import CustomerData
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils


class DataIngestion:

    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        self.data_ingestion_config = data_ingestion_config
        self.utils = MainUtils()

    ##############################################################
    # Split Train/Test
    ##############################################################

    def split_data_as_train_test(
        self,
        dataframe: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:

        logging.info("Splitting dataset into train and test.")

        try:

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            os.makedirs(
                self.data_ingestion_config.ingested_data_dir,
                exist_ok=True
            )

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )

            logging.info("Train dataset saved successfully.")
            logging.info("Test dataset saved successfully.")

        except Exception as e:
            raise CustomerException(e, sys)

    ##############################################################
    # Export MongoDB Data
    ##############################################################

    def export_data_into_feature_store(self) -> DataFrame:

        try:

            logging.info("Exporting data from MongoDB.")

            customer_data = CustomerData()

            customer_dataframe = customer_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            logging.info(
                f"Dataset shape : {customer_dataframe.shape}"
            )

            feature_store_path = (
                self.data_ingestion_config.feature_store_file_path
            )

            os.makedirs(
                os.path.dirname(feature_store_path),
                exist_ok=True
            )

            customer_dataframe.to_csv(
                feature_store_path,
                index=False,
                header=True
            )

            logging.info(
                "Feature Store created successfully."
            )

            return customer_dataframe

        except Exception as e:
            raise CustomerException(e, sys)

    ##############################################################
    # Data Ingestion Pipeline
    ##############################################################

    def initiate_data_ingestion(
        self
    ) -> DataIngestionArtifact:

        logging.info("Starting Data Ingestion.")

        try:

            dataframe = self.export_data_into_feature_store()

            # schema = self.utils.read_schema_config_file()

            # dataframe = dataframe.drop(
            #     schema["drop_columns"],
            #     axis=1
            # )

            # Drop MongoDB _id if present
            if "_id" in dataframe.columns:
                dataframe = dataframe.drop(columns=["_id"])
            logging.info(
                "Dropped unnecessary columns."
            )

            self.split_data_as_train_test(
                dataframe
            )

            logging.info(
                "Data Ingestion completed successfully."
            )

            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

        except Exception as e:
            raise CustomerException(e, sys)