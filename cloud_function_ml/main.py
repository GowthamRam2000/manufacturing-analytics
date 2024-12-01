import functions_framework
from flask import jsonify
from google.cloud import storage
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import io
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_lstm_model(data, sequence_length=100):
    """Train LSTM model for a single machine"""
    try:
        features = ['temperature_reading', 'vibration_reading',
                    'pressure_reading', 'rpm_reading']

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[features])

        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            future_state = 1 if 'failure' in data['state'].iloc[i:i + 288].values else 0
            y.append(future_state)

        X = np.array(X)
        y = np.array(y)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(sequence_length, len(features)), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

        return model, scaler, history.history

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None, None, None
def save_model_to_gcs(model, scaler, machine_id, bucket_name):
    """Save trained model to Google Cloud Storage"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        model_path = f'/tmp/model_{machine_id}'
        model.save(model_path)

        for root, dirs, files in os.walk(model_path):
            for file in files:
                local_path = os.path.join(root, file)
                blob_path = f'models/{machine_id}/{file}'
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)

        scaler_blob = bucket.blob(f'models/{machine_id}/scaler.pkl')
        with open('/tmp/scaler.pkl', 'wb') as f:
            import pickle
            pickle.dump(scaler, f)
        scaler_blob.upload_from_filename('/tmp/scaler.pkl')

        logger.info(f"Saved model for {machine_id} to GCS")
        return True

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False

@functions_framework.http
def retrain_models(request):
    """HTTP Cloud Function for retraining ML models."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket('manufacturing-sensor-data')
        blob = bucket.blob('manufacturing_sensor_data.csv')

        content = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(content))
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        machines = df['machine_id'].unique()
        results = []

        for machine_id in machines:
            logger.info(f"Training model for {machine_id}")

            machine_data = df[df['machine_id'] == machine_id].copy()

            model, scaler, history = train_lstm_model(machine_data)

            if model and scaler:
                success = save_model_to_gcs(model, scaler, machine_id, 'manufacturing-sensor-data')
                results.append({
                    'machine_id': machine_id,
                    'status': 'success' if success else 'failed'
                })
            else:
                results.append({
                    'machine_id': machine_id,
                    'status': 'failed'
                })

        return jsonify({
            'status': 'success',
            'results': results
        })

    except Exception as e:
        logger.error(f"Error in retraining function: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
