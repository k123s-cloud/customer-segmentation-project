"""import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.entity.artifact_entity import (ModelPusherArtifact,
                                           ModelTrainerArtifact)
from src.entity.config_entity import ModelPusherConfig
from src.exception import CustomerException
from src.logger import logging
from src.ml.model.s3_estimator import CustomerClusterEstimator


class ModelPusher:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.s3 = SimpleStorageService()
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config
        self.src_estimator = CustomerClusterEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path,
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            logging.info("Saving model locally instead of uploading to s3 bucket")
            self.local_estimator.save_model(
                from_file="logistic_regression_model.pkl"
            )
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,
            )
            logging.info("Saved model locally")
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e"""
            
import os
import shutil
import sys

from src.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelPusherConfig
from src.exception import CustomerException
from src.logger import logging


class ModelPusher:
    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Saving trained model locally")

            source_model_path = self.model_trainer_artifact.trained_model_file_path

            os.makedirs("saved_models", exist_ok=True)

            destination_path = os.path.join("saved_models", "model.pkl")

            shutil.copy(source_model_path, destination_path)

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=destination_path,
            )

            logging.info(f"Model saved at: {destination_path}")

            return model_pusher_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e
