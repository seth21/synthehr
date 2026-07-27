import sqlite3
import json
import uuid
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
import instructor
from classes import ClinicalEncounterResult
from datetime import datetime, timedelta

# Import Step 3 functions to pull the latent drift data
from drift import fetch_patient_baseline, simulate_latent_gap, LatentSubclinicalProgress

# --- OLLAMA CLIENT SETUP ---
client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)
MODEL = "gemma4:e4b"


# --- THE CLINICAL ENCOUNTER ENGINE ---
def generate_clinical_encounter(
        patient_data: dict,
        latent_drift: LatentSubclinicalProgress,
        patient_age_at_end_of_gap: int,
        active_conditions: List[str],
        active_medications: List[str]
) -> ClinicalEncounterResult:
    """Generates the structured clinical visit output from the drift event."""

    prompt = f"""
    You are an attending physician documenting a clinical encounter.

    FULL NAME: {patient_data['first_name']} {patient_data['last_name']}
    PATIENT DEMOGRAPHICS: {patient_data['demographics']}
    AGE TODAY: {patient_age_at_end_of_gap}
    SURVIVAL STATUS: {latent_drift.survival_status}

    ACTIVE CONDITIONS PRIOR TO VISIT: {active_conditions}
    ACTIVE MEDICATIONS PRIOR TO VISIT: {active_medications}

    REASON FOR PRESENTATION / TRIGGERING EVENT: {latent_drift.triggering_event}
    SILENT PATHOLOGY THAT ACCUMULATED PRIOR TO VISIT: {latent_drift.silent_pathology_accumulated}

    SOCIAL DETERMINANTS OF HEALTH (SDOH):
    - Financial Strain: {patient_data['sdoh'].get('financial_strain')}
    - Stressors: {patient_data['sdoh'].get('current_life_stressors')}

    INSTRUCTIONS:
    1. Write a detailed clinical note with 100-300 words. If SDOH/finances contributed to skipped meds, document it.
    2. Record realistic observations associated with the encounter (e.g. vitals, lab tests, imaging tests).
    3. MANAGE THE PROBLEM LIST (ICD-10) taking into account the silent pathology:
       - If a new disease is diagnosed, START it with a valid ICD-10 code.
       - If an acute condition from their active list (e.g., Acute bronchitis, fractures, acute infections) has healed, RESOLVE it using its exact ICD-10 code.
       - Do NOT resolve chronic, incurable diseases (e.g., Type 2 Diabetes, Amputations).
    4. Adjust or start medications as needed. Use numeric RxNorm RxCUIs for all medication changes.
    5. If SURVIVAL_STATUS is FATAL_EVENT, document the resuscitation/fatal outcome and populate 'primary_cause_of_death'.
    """

    print(f"Generating clinical encounter note for age {patient_age_at_end_of_gap}...")

    encounter = client.chat.completions.create(
        model=MODEL,
        response_model=ClinicalEncounterResult,
        messages=[
            {"role": "system", "content": "You are a professional clinical AI. Output valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    return encounter


# --- DATABASE PERSISTENCE LOGIC ---
def save_encounter_to_db(
        patient_id: str,
        encounter_start: datetime,
        encounter: ClinicalEncounterResult,
        db_name="synthetic_ehr.db"):

    """Appends the generated encounter to the encounters ledger table in SQLite."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    encounter_id = str(uuid.uuid4())
    encounter_end = encounter_start + timedelta(minutes=encounter.total_encounter_duration_minutes)

    # 1. Insert Encounter
    cursor.execute('''
            INSERT INTO encounters (encounter_id, patient_id, encounter_type, start_time, end_time, reason_for_visit, clinical_note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
        encounter_id, patient_id, encounter.encounter_type,
        encounter_start.isoformat(), encounter_end.isoformat(),
        encounter.reason_for_visit, encounter.clinical_note
    ))

    # 2. Insert Observations (Vitals/Labs)
    for obs in encounter.observations:
        obs_time = encounter_start + timedelta(minutes=obs.offset_minutes)
        cursor.execute('''
                INSERT INTO observations (observation_id, encounter_id, patient_id, timestamp, name, value, string_value, unit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), encounter_id, patient_id, obs_time.isoformat(), obs.name, obs.value, obs.string_value, obs.unit))

    # 3. Insert Condition Changes
    for cond in encounter.condition_changes:
        cond_time = encounter_start + timedelta(minutes=cond.offset_minutes)
        cursor.execute('''
                INSERT INTO conditions (condition_id, encounter_id, patient_id, timestamp, action, icd10_code, condition_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), encounter_id, patient_id, cond_time.isoformat(), cond.action, cond.icd10_code,
                  cond.condition_name))

    # 4. Insert Medication Changes
    for med in encounter.medication_changes:
        med_time = encounter_start + timedelta(minutes=med.offset_minutes)
        cursor.execute('''
                INSERT INTO medications (medication_id, encounter_id, patient_id, timestamp, action, rxcui, medication_name, dosage, unit, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), encounter_id, patient_id, med_time.isoformat(), med.action, med.rxcui,
                  med.medication_name, med.dosage, med.unit, med.reason))

    # If the encounter was fatal, update the patient record
    if encounter.primary_cause_of_death:
        death_time = encounter_end.isoformat()
        cursor.execute('''
            UPDATE patients 
            SET is_alive = False, death_date = ?, cause_of_death = ?
            WHERE patient_id = ?
        ''', (death_time, encounter.primary_cause_of_death, patient_id))
        print(f"⚠️ Patient record updated: DECEASED at {death_time}. Cause: {encounter.primary_cause_of_death}")

    conn.commit()
    conn.close()
    print(f"Encounter {encounter_id} saved to SQLite database successfully.")


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Fetch the latest patient from SQLite
    conn = sqlite3.connect("synthetic_ehr.db")
    cursor = conn.cursor()
    cursor.execute("SELECT patient_id FROM patients ORDER BY rowid DESC LIMIT 1")
    patient_id = cursor.fetchone()[0]
    conn.close()

    patient_data = fetch_patient_baseline(patient_id)
    current_age = 35.0
    active_conditions = ["Essential Hypertension"]
    active_medications = ["Lisinopril 10mg Daily"]

    # 2. Simulate Latent Drift (Step 3)
    drift = simulate_latent_gap(patient_data, current_age, active_conditions)

    # 3. Generate Clinical Encounter (Step 4)
    encounter = generate_clinical_encounter(patient_data, drift, active_conditions, active_medications)

    # 4. Save to Database
    save_encounter_to_db(patient_id, drift.patient_age_at_end_of_gap, encounter)

    print("\n--- CLINICAL ENCOUNTER SUMMARY ---")
    print(f"Type: {encounter.encounter_type}")
    print(f"New Diagnoses: {encounter.new_diagnoses}")
    print(f"Observations Recorded: {len(encounter.observations)}")
    print(f"Medication Changes: {len(encounter.medication_changes)}")
    print(f"\nClinical Note Excerpt:\n{encounter.clinical_note[:300]}...")