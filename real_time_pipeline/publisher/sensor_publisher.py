from google.cloud import pubsub_v1
import json
import datetime
import numpy as np
import pandas as pd
import logging


class RealTimeDataPublisher:
    def __init__(self, project_id, topic_name, credentials=None):
        self.publisher = pubsub_v1.PublisherClient(credentials=credentials)
        self.topic_path = self.publisher.topic_path(project_id, topic_name)
        self.logger = logging.getLogger(__name__)

    def generate_sensor_data(self):
        """Generate single sensor reading"""
        machine_id = f"machine_{np.random.randint(1, 11)}"
        now = datetime.datetime.now()

        temp = np.random.normal(65.0, 5.0)
        vibration = np.random.normal(0.5, 0.1)
        pressure = np.random.normal(100.0, 10.0)
        rpm = np.random.normal(1000.0, 50.0)

        if temp > 85 or vibration > 0.8:
            state = 'failure'
        elif np.random.random() < 0.05:
            state = 'maintenance'
        else:
            state = 'running'

        return {
            'timestamp': now.isoformat(),
            'machine_id': machine_id,
            'temperature_reading': float(temp),
            'vibration_reading': float(vibration),
            'pressure_reading': float(pressure),
            'rpm_reading': float(rpm),
            'state': state
        }

    def publish_sensor_data(self):
        try:
            data = self.generate_sensor_data()
            message = json.dumps(data)

            future = self.publisher.publish(
                self.topic_path,
                message.encode('utf-8'),
                timestamp=datetime.datetime.now().isoformat()
            )

            self.logger.info(f"Published message: {message}")
            return future.result()

        except Exception as e:
            self.logger.error(f"Error publishing message: {str(e)}")
            raise