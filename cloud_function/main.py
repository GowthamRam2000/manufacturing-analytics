import functions_framework
from data_generator_gcp import ManufacturingSensorDataGenerator

@functions_framework.http
def generate_data(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
    """
    try:
        generator = ManufacturingSensorDataGenerator(num_machines=10)
        data = generator.save_to_gcs(num_records=100000)
        return 'Data generation and upload successful'
    except Exception as e:
        return f'Error: {str(e)}', 500