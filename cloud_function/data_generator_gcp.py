import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from google.cloud import storage
import io
import os


class ManufacturingSensorDataGenerator:
    def __init__(self, num_machines=10, start_date=None):
        self.num_machines = num_machines
        self.start_date = start_date or datetime.now() - timedelta(days=180)

        self.sensor_params = {
            'temperature': {
                'mean': 65,
                'std': 5,
                'failure_threshold': 85,
                'unit': '°C',
                'normal_range': (55, 75)
            },
            'vibration': {
                'mean': 0.5,
                'std': 0.1,
                'failure_threshold': 0.8,
                'unit': 'mm/s',
                'normal_range': (0.3, 0.7)
            },
            'pressure': {
                'mean': 100,
                'std': 10,
                'failure_threshold': 130,
                'unit': 'PSI',
                'normal_range': (85, 115)
            },
            'rpm': {
                'mean': 1000,
                'std': 50,
                'failure_threshold': 1200,
                'unit': 'RPM',
                'normal_range': (900, 1100)
            }
        }

        self.machine_states = ['running', 'maintenance', 'failure']
        self.state_probabilities = [0.85, 0.10, 0.05]

        self.maintenance_intervals = {
            'scheduled': 30,  
            'preventive': 15  
        }

    def generate_sensor_reading(self, sensor_type, machine_state, time_until_failure=None):
        params = self.sensor_params[sensor_type]
        base_value = np.random.normal(params['mean'], params['std'])

        if machine_state == 'failure':
            base_value *= 1.5
        elif machine_state == 'maintenance':
            base_value *= 0.8
        elif time_until_failure is not None and time_until_failure < 48:
            failure_factor = 1 + (48 - time_until_failure) / 48 * 0.5
            base_value *= failure_factor

        return round(base_value, 2)

    def generate_data(self, num_records=100000):
        """Generate manufacturing sensor data."""
        data = []
        records_per_machine = num_records // self.num_machines

        for machine_id in [f'machine_{i}' for i in range(1, self.num_machines + 1)]:
            current_date = self.start_date
            machine_state = 'running'
            maintenance_counter = 0
            time_until_failure = random.randint(100, 200)  
            last_maintenance = current_date - timedelta(days=random.randint(0, 30))

            for _ in range(records_per_machine):
                if maintenance_counter <= 0:
                    if time_until_failure <= 0:
                        machine_state = 'failure'
                        time_until_failure = random.randint(100, 200)
                    elif (current_date - last_maintenance).days >= self.maintenance_intervals['scheduled']:
                        machine_state = 'maintenance'
                        last_maintenance = current_date
                        maintenance_counter = random.randint(5, 10)  
                    else:
                        machine_state = np.random.choice(
                            self.machine_states,
                            p=self.state_probabilities
                        )
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
                        machine_state,
                        time_until_failure
                    )

                data.append(record)

                current_date += timedelta(minutes=5)
                if machine_state != 'maintenance':
                    time_until_failure -= 1 / 12  

        df = pd.DataFrame(data)
        return df

    def save_to_gcs(self, bucket_name='manufacturing-sensor-data', blob_name='manufacturing_sensor_data.csv',
                    num_records=100000):
        """Generate data and save to Google Cloud Storage."""
        try:
            print(f"Generating {num_records} records...")
            df = self.generate_data(num_records)

            print("Converting to CSV...")
            csv_data = df.to_csv(index=False)

            print("Initializing GCS client...")
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            print(f"Uploading to gs://{bucket_name}/{blob_name}...")
            blob.upload_from_string(csv_data, content_type='text/csv')

            print(f"Successfully generated {len(df)} records and uploaded to GCS")

            print("\nData Statistics:")
            print(f"Total Records: {len(df)}")
            print("\nMachine State Distribution:")
            print(df['state'].value_counts(normalize=True))
            print("\nSensor Reading Statistics:")
            for sensor in ['temperature', 'vibration', 'pressure', 'rpm']:
                print(f"\n{sensor.title()} Readings:")
                print(df[f'{sensor}_reading'].describe())

            return df

        except Exception as e:
            print(f"Error in save_to_gcs: {str(e)}")
            raise

    def save_local(self, filename='manufacturing_sensor_data.csv', num_records=100000):
        """Generate data and save locally (for testing)."""
        df = self.generate_data(num_records)
        df.to_csv(filename, index=False)
        print(f"Generated {len(df)} records and saved to {filename}")
        return df


if __name__ == "__main__":
    generator = ManufacturingSensorDataGenerator(num_machines=10)

    if os.getenv('ENV') == 'local':
        data = generator.save_local(num_records=100000)
    else:
        data = generator.save_to_gcs(num_records=100000)

    print("\nData Generation Complete")
