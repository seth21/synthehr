import sqlite3
import json
from classes import LatentSubclinicalProgress
from typing import List
from openai import OpenAI
import instructor

# --- OLLAMA CLIENT SETUP ---
client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)
MODEL = "gemma4:e4b"

# --- DATABASE FETCH LOGIC ---
def fetch_patient_baseline(patient_id: str, db_name="synthetic_ehr.db"):
    """Pulls the patient's seed data from SQLite and parses the JSON strings."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT first_name, last_name, demographics, behavioral_profile, genetic_risks, current_sdoh 
        FROM patients WHERE patient_id = ?
    """, (patient_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise ValueError(f"Patient {patient_id} not found.")

    return {
        "first_name": row[0],
        "last_name": row[1],
        "demographics": row[2],
        "behavior": row[3],
        "genetics": json.loads(row[4]),
        "sdoh": json.loads(row[5])
    }


# --- THE DRIFT ENGINE ---
def simulate_latent_gap(patient_data: dict, current_age: float,
                        active_conditions: dict) -> LatentSubclinicalProgress:
    """Calculates biological and social drift over unobserved time."""

    prompt = f"""
    You are a clinical pathology and mortality simulator modeling a human life.
    
    PATIENT DEMOGRAPHICS: {patient_data['demographics']}
    CURRENT AGE: {current_age}
    ACTIVE CONDITIONS: {active_conditions}

    # GENETICS & SDOH
    BEHAVIOR: {patient_data['behavior']}
    GENETIC RISKS: {patient_data['genetics'].get('genetic_risk_factors', [])}
    SDOH STRESSORS: {patient_data['sdoh'].get('current_life_stressors', [])}
    FINANCIAL STRAIN: {patient_data['sdoh'].get('financial_strain', 'Unknown')}

    Based on their biological anchors and social reality:
    1. Determine how many years pass before they interact with the healthcare system again.
    2. Document the silent pathology accumulating in the dark.
    3. Determine the triggering event that ends the gap.
    4. Decide if they SURVIVED or if this was a FATAL_EVENT.
    """

    print(f"Simulating time drift for patient at age {current_age}...")

    drift_state = client.chat.completions.create(
        model=MODEL,
        response_model=LatentSubclinicalProgress,
        messages=[
            {"role": "system", "content": "You are a deterministically grounded medical AI. Output valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    return drift_state


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Quick hack to grab the most recently generated patient for testing
    conn = sqlite3.connect("synthetic_ehr.db")
    cursor = conn.cursor()
    cursor.execute("SELECT patient_id FROM patients ORDER BY rowid DESC LIMIT 1")
    latest_patient_id = cursor.fetchone()[0]
    conn.close()

    # Fetch the data
    patient_data = fetch_patient_baseline(latest_patient_id)

    # We simulate their life starting at age 30 with no active conditions yet
    current_age = 30.0
    active_conditions = []

    # Run the engine
    drift_result = simulate_latent_gap(patient_data, current_age, active_conditions)

    # Output the result
    print("\n--- LATENT DRIFT RESULT ---")
    print(f"Time Passed: {drift_result.unobserved_years_passed} years")
    print(f"New Age: {drift_result.patient_age_at_end_of_gap}")
    print(f"Silent Pathology: {drift_result.silent_pathology_accumulated}")
    print(f"Triggering Event: {drift_result.triggering_event}")
    print(f"Survival Status: {drift_result.survival_status}")