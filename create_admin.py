import firebase_admin
from firebase_admin import credentials, auth

cred = credentials.Certificate('service-account.json')
firebase_admin.initialize_app(cred)

user = auth.create_user(
    email='admin@manufacturing.com',
    password='Admin@123456',
    display_name='Admin User'
)

auth.set_custom_user_claims(user.uid, {'admin': True})

print(f"Admin user created successfully!")
print(f"Email: admin@manufacturing.com")
print(f"Password: Admin@123456")