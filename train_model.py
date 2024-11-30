from google.cloud import storage
import pandas as pd
import io
from predictive_maintenance import PredictiveMaintenance
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_from_gcs(bucket_name, blob_name):
    """Load data from Google Cloud Storage"""
    try:
        # Get credentials from the environment
        credentials_path = 'service-account.json'
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

        storage_client = storage.Client(project='log-project-442913')
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        logger.info(f"Attempting to load data from gs://{bucket_name}/{blob_name}")
        content = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(content))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None


def train_models():
    """Train LSTM models for each machine"""
    try:
        df = load_data_from_gcs('manufacturing-sensor-data', 'manufacturing_sensor_data.csv')
        if df is None:
            raise ValueError("Could not load training data")

        machines = df['machine_id'].unique()
        for machine_id in machines:
            logger.info(f"Training model for {machine_id}")

            machine_data = df[df['machine_id'] == machine_id].copy()

            model = PredictiveMaintenance(sequence_length=100)
            history = model.train(machine_data, epochs=10, batch_size=32)

            model_dir = f'models/{machine_id}'
            model.save_model(model_dir)

            logger.info(f"Model for {machine_id} trained and saved successfully")

            final_loss = history.history['loss'][-1]
            final_accuracy = history.history['accuracy'][-1]
            logger.info(f"Final training metrics for {machine_id}:")
            logger.info(f"Loss: {final_loss:.4f}")
            logger.info(f"Accuracy: {final_accuracy:.4f}")

    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        raise


if __name__ == "__main__":
    train_models()