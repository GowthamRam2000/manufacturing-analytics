from cloud_function.data_generator_gcp import ManufacturingSensorDataGenerator
from google.cloud import storage
from google.oauth2 import service_account
import pandas as pd
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            'service-account.json',
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

        storage_client = storage.Client(
            project='log-analysis-4a093',
            credentials=credentials
        )

        logger.info("Generating data...")
        generator = ManufacturingSensorDataGenerator(num_machines=10)
        df = generator.generate_data(num_records=100000)

        logger.info("Converting to CSV...")
        csv_data = df.to_csv(index=False)

        bucket_name = 'manufacturing-sensor-data'
        blob_name = 'manufacturing_sensor_data.csv'

        logger.info(f"Uploading to gs://{bucket_name}/{blob_name}")
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.upload_from_string(csv_data, content_type='text/csv')

        logger.info("Upload successful!")
        logger.info(f"Generated {len(df)} records")
        logger.info("\nData Statistics:")
        logger.info("State Distribution:")
        logger.info(df['state'].value_counts().to_string())
        logger.info("\nSensor Statistics:")
        for col in ['temperature_reading', 'vibration_reading', 'pressure_reading', 'rpm_reading']:
            logger.info(f"\n{col}:")
            logger.info(df[col].describe().to_string())

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()