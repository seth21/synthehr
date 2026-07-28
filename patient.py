from datetime import datetime
from openai import OpenAI
import instructor
import sqlite3
import random

from classes import PatientBaseState

# Bind instructor to the local Ollama instance
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # The API key is required by the SDK but ignored by Ollama
    ),
    mode=instructor.Mode.JSON, # Crucial for local models to force JSON output
)

MODEL = "gemma4:e4b"

def generate_patient_base(birth_year: int, gender: str, ethnicity: str) -> PatientBaseState:
    """Generates the starting genetic and social state of the patient."""
    # Force clinical diversity by injecting a random archetype
    archetypes = [
        "Autoimmune (e.g., Rheumatoid Arthritis, Lupus, Psoriasis)",
        "Neurological (e.g., Migraines, Early-onset Parkinson's, Epilepsy)",
        "Musculoskeletal/Orthopedic (e.g., Osteoarthritis, Chronic back pain, Gout)",
        "Pulmonary (e.g., Asthma, COPD, Interstitial lung disease)",
        "Gastrointestinal (e.g., Crohn's, Ulcerative Colitis, Severe GERD)",
        "Psychiatric (e.g., Bipolar disorder, Major Depressive Disorder, Severe Anxiety)",
        "Cardio-Metabolic (e.g., Diabetes, Hypertension, Heart Failure)",
        "Generally Healthy"
    ]
    assigned_archetype = random.choices(archetypes, weights=[5, 10, 20, 20, 15, 10, 30, 20], k=1)[0]
    compliance_types = [
        "Highly Compliant", "Needs Reminders", "Care Avoider"
    ]
    assigned_compliance = random.choices(compliance_types, weights=[60, 30, 10], k=1)[0]

    prompt = f"""
    You are a clinical data generator. Generate a realistic medical backstory for a patient.
    
    Demographics: {gender}, {ethnicity}, born {birth_year}
    
    CLINICAL DESTINY ARCHETYPE: {assigned_archetype}
    COMPLIANCE ARCHETYPE: {assigned_compliance}

    INSTRUCTIONS:
    1. Generate the patient's genetic risks and family history strictly focused on the CLINICAL DESTINY ARCHETYPE provided above. 
    2. Give them a realistic family history with at least one notable genetic risk factor.
    3. Do NOT default to common diseases like Diabetes or Hypertension unless it is the assigned archetype. 
    4. Generate a realistic behavioral and SDOH profile and make the profile cohesive.
    """

    print(f"Generating base state for {gender} born in {birth_year}...")

    patient_state = client.chat.completions.create(
        model=MODEL,
        response_model=PatientBaseState,
        messages=[
            {"role": "system", "content": "You output strict, valid JSON matching the schema."},
            {"role": "user", "content": prompt}
        ],
        temperature = 0.7
    )
    return patient_state


def save_patient_to_db(patient_id: str, dob: datetime, state: PatientBaseState, db_name="synthetic_ehr.db"):
    """Saves the generated patient baseline into SQLite."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO patients (
            patient_id, first_name, last_name, dob, demographics, behavioral_profile, 
            genetic_risks, current_sdoh, is_alive, death_date, cause_of_death
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_id,
        state.first_name,
        state.last_name,
        dob,
        state.demographics,
        state.behavioral_profile,
        state.family_history.model_dump_json(),
        state.sdoh_profile.model_dump_json(),
        True,  # is_alive
        None,  # age_at_death
        None  # cause_of_death
    ))

    conn.commit()
    conn.close()
    print(f"Patient {patient_id} successfully saved to database.")
