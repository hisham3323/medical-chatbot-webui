import os
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Load Environment and Connect to Supabase ---
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise RuntimeError("Supabase URL or Key not found in .env file.")

supabase: Client = create_client(url, key)
print("✅ Connected to Supabase.")

# --- Define the Test Patient Data ---
# Note: The 'conditions' array is a list of strings in Python.
patient_data = {
    "id": "test-user-123",
    "name": "John Doe",
    "age": 35,
    "gender": "Male",
    "blood_type": "O+",
    "weight": 80,
    "height": 180,
    "diabetes": False,
    "hypertension": True,
    "conditions": ["headache", "sore throat"]
}

# --- Insert the Data into the 'patients' Table ---
try:
    print(f"Attempting to insert patient with id: {patient_data['id']}...")
    
    # The insert function takes a dictionary.
    # The .execute() command runs the operation.
    response = supabase.table("patients").insert(patient_data).execute()
    
    # Check if the insert was successful
    if response.data:
        print("✅ Patient data inserted successfully!")
    else:
        # Supabase V2 might return an empty list on success, check status code
        if 200 <= response.status_code < 300:
            print("✅ Patient data inserted successfully!")
        else:
            print("❌ Insertion failed. Response:", response)

except Exception as e:
    # This will catch errors like "unique constraint violated" if the user already exists.
    print(f"❌ An error occurred: {e}") 