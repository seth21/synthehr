from datetime import datetime
from openai import OpenAI
import instructor
import sqlite3
import faker

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

    prompt = f"""
    You are a clinical data generator. Generate a realistic medical backstory for a patient.
    - Born: {birth_year}
    - Gender: {gender}
    - Ethnicity: {ethnicity}

    Make the profile cohesive. For example, if they are 'FINANCIALLY_CONSTRAINED', their SDOH profile should reflect poverty.
    Give them a realistic family history with at least one or two notable genetic risk factors.
    """

    print(f"Generating base state for {gender} born in {birth_year}...")

    patient_state = client.chat.completions.create(
        model=MODEL,
        response_model=PatientBaseState,
        messages=[
            {"role": "system", "content": "You output strict, valid JSON matching the schema."},
            {"role": "user", "content": prompt}
        ]
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
