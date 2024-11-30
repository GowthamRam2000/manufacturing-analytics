import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from google.cloud import storage
import io


class ManufacturingSensorDataGenerator:
    def __init__(self, num_machines=10, start_date=None):
        self.num_machines = num_machines
        self.start_date = start_date or datetime.now() - timedelta(days=180)

        self.sensor_params = {
            'temperature': {'mean': 65, 'std': 5, 'failure_threshold': 85},
            'vibration': {'mean': 0.5, 'std': 0.1, 'failure_threshold': 0.8},
            'pressure': {'mean': 100, 'std': 10, 'failure_threshold': 130},
            'rpm': {'mean': 1000, 'std': 50, 'failure_threshold': 1200}
        }

        self.machine_states = ['running', 'maintenance', 'failure']
        self.state_probabilities = [0.85, 0.10, 0.05]

    def generate_sensor_reading(self, sensor_type, machine_state):
        params = self.sensor_params[sensor_type]
        base_value = np.random.normal(params['mean'], params['std'])

        if machine_state == 'failure':
            base_value *= 1.5
        elif machine_state == 'maintenance':
            base_value *= 0.8

        return round(base_value, 2)

    def generate_data(self, num_records=100000):
        data = []
        records_per_machine = num_records // self.num_machines

        for machine_id in [f'machine_{i}' for i in range(1, self.num_machines + 1)]:
            current_date = self.start_date
            machine_state = 'running'
            maintenance_counter = 0

            for _ in range(records_per_machine):
                if maintenance_counter <= 0:
                    machine_state = np.random.choice(
                        self.machine_states,
                        p=self.state_probabilities
                    )
                    maintenance_counter = random.randint(10, 30) if machine_state == 'maintenance' else 0
                else:
                    maintenance_counter -= 1

                record = {
                    'timestamp': current_date,
                    'machine_id': machine_id,
                    'state': machine_state,
                }

                for sensor_type in self.sensor_params.keys():
                    record[f'{sensor_type}_reading'] = self.generate_sensor_reading(
                        sensor_type,
                        machine_state
                    )

                data.append(record)
                current_date += timedelta(minutes=5)

        return pd.DataFrame(data)

    def save_to_gcs(self, bucket_name='manufacturing-sensor-data', blob_name='manufacturing_sensor_data.csv',
                    num_records=100000):
        df = self.generate_data(num_records)

        csv_data = df.to_csv(index=False)

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_string(csv_data, content_type='text/csv')

        print(f"Generated {len(df)} records and uploaded to gs://{bucket_name}/{blob_name}")
        return df


if __name__ == "__main__":
    generator = ManufacturingSensorDataGenerator(num_machines=10)
    data = generator.save_to_gcs(num_records=100000)

    print("\nData Statistics:")
    print(f"Total Records: {len(data)}")
    print("\nMachine State Distribution:")
    print(data['state'].value_counts(normalize=True))