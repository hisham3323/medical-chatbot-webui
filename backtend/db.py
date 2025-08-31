"""
Supabase helper for the medical-chat project.

- Uses SUPABASE_URL and SUPABASE_KEY from the .env file.
- Reads from the 'patients' table.
"""

import os
from supabase import create_client, Client

# --- Supabase Connection ---
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise RuntimeError("Supabase URL or Key not found in environment variables.")

supabase: Client = create_client(url, key)
print("✅ Supabase client initialised.")

# --- Collection Name Constant ---
PATIENTS_TABLE = "patients"

def get_user_data(user_id: str) -> dict:
    """
    Return the patient profile from the Supabase 'patients' table as a dict.
    If nothing is found, returns {}.
    """
    try:
        response = supabase.table(PATIENTS_TABLE).select("*").eq("id", user_id).execute()
        
        # The Supabase client returns a response object with a 'data' attribute,
        # which is a list of matching rows.
        if response.data:
            # Return the first matching patient record
            return response.data[0]
        else:
            print(f"⚠️ No patient data found for '{user_id}' in '{PATIENTS_TABLE}' table.")
            return {}
            
    except Exception as e:
        print(f"❌ Error fetching data from Supabase: {e}")
        return {}

# --- Quick Manual Test (Read-Only) ---
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python db.py <patient_uid>")
        sys.exit(1)

    uid = sys.argv[1]
    profile = get_user_data(uid)
    
    if profile:
        print("\nFetched profile:")
        for k, v in profile.items():
            print(f"{k}: {v}")