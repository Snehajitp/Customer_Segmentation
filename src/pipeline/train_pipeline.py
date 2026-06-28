import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomerException
from src.logger import logging

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)


class TrainPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:

        try:
            logging.info("Starting Data Ingestion...")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Data Ingestion Completed.")

            return artifact

        except Exception as e:
            raise CustomerException(e, sys)

    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:

        try:
            logging.info("Starting Data Validation...")

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )

            artifact = data_validation.initiate_data_validation()

            logging.info("Data Validation Completed.")

            return artifact

        except Exception as e:
            raise CustomerException(e, sys)

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:

        try:
            logging.info("Starting Data Transformation...")

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_tranasformation_config=self.data_transformation_config
            )

            artifact = data_transformation.initiate_data_transformation()

            logging.info("Data Transformation Completed.")

            return artifact

        except Exception as e:
            raise CustomerException(e, sys)

    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:

        try:
            logging.info("Starting Model Training...")

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )

            artifact = model_trainer.initiate_model_trainer()

            logging.info("Model Training Completed.")

            return artifact

        except Exception as e:
            raise CustomerException(e, sys)

    def run_pipeline(self):

        try:

            logging.info("========== TRAINING PIPELINE STARTED ==========")

            # -----------------------------
            # Data Ingestion
            # -----------------------------
            data_ingestion_artifact = self.start_data_ingestion()

            # -----------------------------
            # Data Validation
            # -----------------------------
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact
            )

            # -----------------------------
            # Data Transformation
            # -----------------------------
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact,
                data_validation_artifact
            )

            # -----------------------------
            # Model Training
            # -----------------------------
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact
            )

            logging.info("========== TRAINING PIPELINE COMPLETED ==========")

            print("\n======================================")
            print("Training completed successfully!")
            print("Model saved at: artifacts/model.pkl")
            print("Preprocessor saved at: artifacts/preprocessing.pkl")
            print("======================================\n")

            return model_trainer_artifact

        except Exception as e:
            logging.exception(e)
            raise CustomerException(e, sys)