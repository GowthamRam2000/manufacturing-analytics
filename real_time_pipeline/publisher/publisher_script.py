# publisher_script.py
from sensor_publisher import RealTimeDataPublisher
import time
import logging
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = 'log-project-442913'
TOPIC_NAME = 'sensor-data-stream'
SERVICE_ACCOUNT_PATH = '/Users/gowthamram/PycharmProjects/manufacturing_analytics/service-account.json'


def main():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        scopes=['https://www.googleapis.com/auth/pubsub']
    )

    publisher = RealTimeDataPublisher(PROJECT_ID, TOPIC_NAME, credentials)

    try:
        logger.info("Starting real-time data publication...")
        while True:
            publisher.publish_sensor_data()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping data publication...")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()