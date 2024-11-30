from cloud_function.data_generator_gcp import ManufacturingSensorDataGenerator

def main():
    generator = ManufacturingSensorDataGenerator(num_machines=10)
    data = generator.save_to_gcs(
        bucket_name='manufacturing-sensor-data',
        blob_name='manufacturing_sensor_data.csv',
        num_records=100000
    )

if __name__ == "__main__":
    main()