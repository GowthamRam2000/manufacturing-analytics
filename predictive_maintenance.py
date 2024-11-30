import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import joblib
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class PredictiveMaintenance:
    def __init__(self, sequence_length=100):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def prepare_data(self, df):
        features = ['temperature_reading', 'vibration_reading',
                    'pressure_reading', 'rpm_reading']

        X = self.scaler.fit_transform(df[features])
        sequences = []
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:(i + self.sequence_length)])

        return np.array(sequences)

    def train(self, data):
        X = self.prepare_data(data)
        self.model = self.build_model((self.sequence_length, X.shape[2]))
        return self.model.fit(X, epochs=10, batch_size=32)

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model not trained")

        X = self.prepare_data(data)
        if len(X) < 1:
            return None

        return self.model.predict(X[-1:])

    def save_model(self, bucket_name, machine_id):
        try:
            from google.cloud import storage

            local_path = f'/tmp/model_{machine_id}'
            os.makedirs(local_path, exist_ok=True)
            self.model.save(f'{local_path}/model.keras')
            joblib.dump(self.scaler, f'{local_path}/scaler.pkl')

            client = storage.Client()
            bucket = client.bucket(bucket_name)

            model_blob = bucket.blob(f'models/{machine_id}/model.keras')
            model_blob.upload_from_filename(f'{local_path}/model.keras')

            scaler_blob = bucket.blob(f'models/{machine_id}/scaler.pkl')
            scaler_blob.upload_from_filename(f'{local_path}/scaler.pkl')

            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    @classmethod
    def load_model(cls, bucket_name, machine_id):
        try:
            from google.cloud import storage

            instance = cls()
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            local_path = f'/tmp/model_{machine_id}'
            os.makedirs(local_path, exist_ok=True)

            model_blob = bucket.blob(f'models/{machine_id}/model.keras')
            model_blob.download_to_filename(f'{local_path}/model.keras')

            scaler_blob = bucket.blob(f'models/{machine_id}/scaler.pkl')
            scaler_blob.download_to_filename(f'{local_path}/scaler.pkl')

            instance.model = tf.keras.models.load_model(f'{local_path}/model.keras')
            instance.scaler = joblib.load(f'{local_path}/scaler.pkl')

            return instance
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return None