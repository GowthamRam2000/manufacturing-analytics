import functions_framework
import logging
import json
import base64
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    client = bigquery.Client()
except Exception as e:
    logger.error(f"Error initializing BigQuery client: {str(e)}")
    client = None


@functions_framework.http
def process_sensor_data(request):
    """HTTP Cloud Function for processing sensor data."""
    try:
        if not client:
            raise Exception("BigQuery client not initialized")

        # Get data from Pub/Sub message
        envelope = request.get_json()
        if not envelope:
            raise ValueError("No Pub/Sub message received")

        if not isinstance(envelope, dict) or 'message' not in envelope:
            raise ValueError("Invalid Pub/Sub message format")

        pubsub_message = envelope['message']
        if isinstance(pubsub_message, dict) and 'data' in pubsub_message:
            data = base64.b64decode(pubsub_message['data']).decode('utf-8')
        else:
            raise ValueError("No data found in Pub/Sub message")

        sensor_data = json.loads(data)

        rows_to_insert = [{
            'timestamp': sensor_data['timestamp'],
            'machine_id': sensor_data['machine_id'],
            'temperature': sensor_data['temperature_reading'],
            'vibration': sensor_data['vibration_reading'],
            'pressure': sensor_data['pressure_reading'],
            'rpm': sensor_data['rpm_reading'],
            'state': sensor_data['state']
        }]

        table_id = "log-project-442913.manufacturing_data.real_time_sensor_data"
        errors = client.insert_rows_json(table_id, rows_to_insert)

        if errors:
            raise Exception(f"BigQuery insert errors: {errors}")

        logger.info("Data successfully processed and stored")
        return 'Success', 200

    except Exception as e:
        error_message = f"Error processing message: {str(e)}"
        logger.error(error_message)
        return error_message, 500