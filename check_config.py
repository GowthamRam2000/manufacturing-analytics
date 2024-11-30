import firebase_admin
from firebase_admin import credentials, auth
from google.cloud import storage
from google.oauth2 import service_account
import os
import json


def check_firebase():
    print("\nChecking Firebase configuration...")
    try:
        if not os.path.exists('service-account.json'):
            print("❌ service-account.json not found!")
            return False

        with open('service-account.json', 'r') as f:
            service_account = json.load(f)
            print("✅ service-account.json is valid JSON")
            print(f"Project ID: {service_account.get('project_id')}")

        if not firebase_admin._apps:
            cred = credentials.Certificate('service-account.json')
            firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin SDK initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Firebase error: {str(e)}")
        return False


def check_gcs():
    print("\nChecking Google Cloud Storage configuration...")
    try:
        credentials = service_account.Credentials.from_service_account_file(
            'service-account.json'
        )

        storage_client = storage.Client(
            project='log-analysis-4a093',
            credentials=credentials
        )

        bucket = storage_client.bucket('manufacturing-sensor-data')

        test_content = "test,data\n1,2"
        blob = bucket.blob('test_upload.csv')

        print("Testing file upload...")
        blob.upload_from_string(test_content, content_type='text/csv')
        print("✅ Successfully uploaded test file")

        print("Testing file download...")
        content = blob.download_as_string()
        print("✅ Successfully downloaded test file")

        print("Testing file deletion...")
        blob.delete()
        print("✅ Successfully deleted test file")

        return True
    except Exception as e:
        print(f"❌ GCS error: {str(e)}")
        return False


def main():
    print("Starting configuration check...")

    firebase_ok = check_firebase()
    gcs_ok = check_gcs()

    if firebase_ok and gcs_ok:
        print("\n✅ All configurations are valid!")
        print("Note: Basic file operations are working, which is sufficient for our application.")
        return True
    else:
        print("\n❌ Some configurations failed!")
        if not firebase_ok:
            print("- Firebase configuration needs attention")
        if not gcs_ok:
            print("- GCS access needs attention")
        return False


if __name__ == "__main__":
    main()