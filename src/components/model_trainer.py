import os
import sys

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)

from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, load_numpy_array_data

from neuro_mf import ModelFactory


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        logging.info("Starting Model Training")

        try:

            # -------------------------------------
            # Load transformed train/test arrays
            # -------------------------------------
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            logging.info("Training and testing data loaded successfully.")

            # -------------------------------------
            # Train Model
            # -------------------------------------
            model_factory = ModelFactory(
                model_config_path=self.model_trainer_config.model_config_file_path
            )

            best_model_detail = model_factory.get_best_model(
                X=x_train,
                y=y_train,
                base_accuracy=self.model_trainer_config.expected_accuracy,
            )

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    "No model achieved the expected accuracy."
                )

            logging.info(
                f"Best Model : {best_model_detail.best_model}"
            )

            logging.info(
                f"Best CV Score : {best_model_detail.best_score}"
            )

            # -------------------------------------
            # Load Preprocessor
            # -------------------------------------
            preprocessing_obj = self.utils.load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # -------------------------------------
            # Save Model & Preprocessor
            # -------------------------------------
            artifacts_dir = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            os.makedirs(artifacts_dir, exist_ok=True)

            preprocessor_path = os.path.join(
                artifacts_dir,
                "preprocessing.pkl",
            )

            model_path = os.path.join(
                artifacts_dir,
                "model.pkl",
            )

            self.utils.save_object(
                preprocessor_path,
                preprocessing_obj,
            )

            self.utils.save_object(
                model_path,
                best_model_detail.best_model,
            )

            logging.info("Model and Preprocessor saved successfully.")

            # -------------------------------------
            # Evaluate Model
            # -------------------------------------
            y_pred = best_model_detail.best_model.predict(x_test)

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                ),
                precision_score=precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                ),
                recall_score=recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                ),
            )

            logging.info(f"F1 Score        : {metric_artifact.f1_score:.4f}")
            logging.info(f"Precision Score : {metric_artifact.precision_score:.4f}")
            logging.info(f"Recall Score    : {metric_artifact.recall_score:.4f}")

            # -------------------------------------
            # Return Artifact
            # -------------------------------------
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=model_path,
                metric_artifact=metric_artifact,
            )

            logging.info("Model Training Completed Successfully.")

            return model_trainer_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e