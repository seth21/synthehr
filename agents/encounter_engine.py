import sqlite3
import uuid
from openai import OpenAI
import instructor
from core.classes import ClinicalEncounterResult
from datetime import datetime, timedelta
import json
from db.database import db

# Import Step 3 functions to pull the latent drift data
from agents.drift_engine import LatentSubclinicalProgress

# --- OLLAMA CLIENT SETUP ---
client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)
MODEL = "gemma4:e4b"


def get_latest_observations(patient_id, num_observations):
    recent_obs_data = db.fetch_all('''
            SELECT name, value, unit, timestamp 
            FROM observations 
            WHERE patient_id = ? 
            ORDER BY timestamp DESC LIMIT ?
        ''', (patient_id, num_observations)
    )

    recent_obs_list = []
    for obs in recent_obs_data:
        obs_date = datetime.fromisoformat(obs[3]).strftime("%Y-%m-%d")
        unit = obs[2] if obs[2] else ""
        recent_obs_list.append(f"{obs[0]}: {obs[1]} {unit} ({obs_date})")

    recent_obs_text = ", ".join(recent_obs_list) if recent_obs_list else "No prior observations."
    return recent_obs_text

# --- THE CLINICAL ENCOUNTER ENGINE ---
def generate_clinical_encounter(
        patient_data: dict,
        latent_drift: LatentSubclinicalProgress,
        patient_age_at_end_of_gap: int,
        active_conditions: dict,
        active_medications: dict,
        patient_id: str,
        still_pending_orders_txt: str
) -> ClinicalEncounterResult:
    """Generates the structured clinical visit output from the drift event."""
    active_meds_display = [data["display"] for data in active_medications.values()]
    active_conds_display = list(active_conditions.values())

    # Safely parse the significant_diagnostics JSON
    sig_diag = patient_data.get('significant_diagnostics')
    sig_diag_list = json.loads(sig_diag) if sig_diag else []
    # Get past medical history
    recent_obs_text = get_latest_observations(patient_id, 5)

    prompt = f"""
    You are an attending physician documenting a clinical encounter.

    FULL NAME: {patient_data['first_name']} {patient_data['last_name']}
    PATIENT DEMOGRAPHICS: {patient_data['demographics']}
    AGE TODAY: {patient_age_at_end_of_gap}
    SURVIVAL STATUS: {latent_drift.survival_status}

    ACTIVE CONDITIONS PRIOR TO VISIT: {active_conds_display}
    ACTIVE MEDICATIONS PRIOR TO VISIT: {active_meds_display}
    
    PAST MEDICAL & SURGICAL HISTORY: {patient_data.get('pmh_summary', 'No significant history.')}
    MAJOR PAST DIAGNOSTIC BASELINES: {sig_diag_list if sig_diag_list else 'No significant baseline diagnostics on file.'}
    RECENT FLOWSHEET (Last 5 Labs/Vitals): {recent_obs_text}
    FUTURE SCHEDULED ORDERS (DO NOT DUPLICATE THESE): {still_pending_orders_txt}
    REASON FOR VISIT / TRIGGERING EVENT: {latent_drift.triggering_event}
    SILENT PATHOLOGY THAT ACCUMULATED PRIOR TO VISIT: {latent_drift.silent_pathology_accumulated}

    SOCIAL DETERMINANTS OF HEALTH (SDOH):
    - Financial Strain: {patient_data['sdoh'].get('financial_strain')}
    - Stressors: {patient_data['sdoh'].get('current_life_stressors')}

    INSTRUCTIONS:
    1. Write a detailed clinical note with 100-300 words. If SDOH/finances contributed to skipped meds, document it.
    2. Record realistic observations associated with the encounter (e.g. vitals, lab tests, imaging tests).
    3. SPECIALTY ALIGNMENT: Read the REASON FOR VISIT carefully. If this is a specific scheduled order (e.g., "Cardiology Consultation", "Renal Ultrasound", "Colonoscopy"), you MUST act as that specific specialist or department. 
    4. EXECUTE THE ORDER: Ensure your clinical note and observations directly reflect the results of the requested procedure or consultation.
    5. MANAGE THE PROBLEM LIST (ICD-10) taking into account the silent pathology:
       - If a new disease is diagnosed, START it with a valid ICD-10 code.
       - If an acute condition from their active list (e.g., Acute bronchitis, fractures, acute infections) has healed, RESOLVE it using its exact ICD-10 code.
       - Do NOT resolve chronic, incurable diseases (e.g., Type 2 Diabetes, Amputations).
       - CRITICAL: If this is a routine checkup for a young/healthy patient, it is PERFECTLY NORMAL to have NO new diagnoses and NO medication changes. Do not invent pathology if they are healthy.
       - Do NOT output a START action for a diagnosis that is already present in the ACTIVE CONDITIONS list.
    6. Adjust, start or stop medications as needed. Use numeric RxNorm RxCUIs for all medication changes. 
    You cannot STOP or perform a DOSE_CHANGE on a medication unless it is explicitly listed in the ACTIVE MEDICATIONS PRIOR TO VISIT. If starting a new drug, you must use START.
    DIAGNOSIS-MEDICATION LINKAGE: You MUST NOT prescribe a medication (like Metformin) unless the underlying disease (like Diabetes) is explicitly diagnosed and added to the Problem List today or is already active.
    THERAPEUTIC DUPLICATION: If you are switching a patient to a new medication within the same class (e.g., switching from Simvastatin to Atorvastatin), you MUST output a STOP action for the old medication.
    7. If SURVIVAL_STATUS is FATAL_EVENT, document the resuscitation/fatal outcome and populate 'primary_cause_of_death'.
    8. PLAN & NEXT STEPS: If the patient requires future follow-up, specialist evaluation, or imaging, generate orders_placed with realistic timelines (e.g. a 3-month routine follow-up).
    - FUTURE ORDERS ONLY: Use `orders_placed` strictly for appointments or labs happening TOMORROW or later (target_days_from_now >= 1). Do NOT re-order routine labs, consults, or follow-ups if they are already on the "Future Scheduled Orders" list
    - SAME-DAY DIAGNOSTICS: If you need a lab test or imaging study TODAY (e.g., STAT Troponin, CMP, Chest X-Ray), DO NOT put it in `orders_placed`. Instead, assume the test was already performed. Invent realistic results and place them directly into the `observations` array. Evaluate these results in your clinical note.
    - If the patient is young and healthy, the only order might be "Follow up in 1 year" (365 days) or NO orders at all.
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
                INSERT INTO observations (observation_id, encounter_id, patient_id, timestamp, name, value, unit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), encounter_id, patient_id, obs_time.isoformat(), obs.name, obs.value, obs.unit))

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
        new_medication_id = str(uuid.uuid4())
        #Create the new medication if it does NOT exist in the database
        cursor.execute("""
                INSERT OR IGNORE INTO medications (
                    medication_id,
                    rxcui,
                    medication_name,
                    dosage,
                    unit,
                    drug_class,
                    therapeutical_class,
                    form
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
            new_medication_id,
            med.rxcui,
            med.medication_name,
            med.dosage,
            med.unit,
            'N/A',
            'N/A',
            'N/A',
        ))
        #Get the medication_id of the drug
        row = cursor.execute("""
                SELECT medication_id
                FROM medications
                WHERE rxcui = ?
            """, (med.rxcui,)).fetchone()
        actual_medication_id = row[0]

        #Create the new medication change event
        cursor.execute('''
                INSERT INTO medication_events (medication_event_id, encounter_id, patient_id, medication_id, timestamp, action, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), encounter_id, patient_id, actual_medication_id, med_time.isoformat(), med.action,
                  med.reason))

        #Update the current patient medications
        cursor.execute("""
                INSERT INTO patient_medications (
                    patient_medication_id,
                    patient_id,
                    medication_id,
                    start_date,
                    updated_at,
                    status,
                    route,
                    frequency
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(patient_id, medication_id)
                DO UPDATE SET
                    updated_at = excluded.updated_at,
                    status = excluded.status,
                    route = excluded.route,
                    frequency = excluded.frequency,
                    end_date = CASE
                        WHEN excluded.status = 'STOP'
                        THEN excluded.updated_at
                        ELSE NULL
                    END
            """, (
            str(uuid.uuid4()),
            patient_id,
            actual_medication_id,
            med_time.isoformat(),
            med_time.isoformat(),
            med.action,
            'N/A',
            'N/A'
        ))

    # 5. Insert Pending Orders (The Care Pathway Queue)
    for order in encounter.orders_placed:
        # Ignore same-day orders so they don't clog the future queue
        if order.target_days_from_now <= 0:
            continue
        order_id = str(uuid.uuid4())

        # Calculate the future date this order should happen
        target_date = encounter_start + timedelta(days=order.target_days_from_now)

        cursor.execute('''
            INSERT INTO pending_orders (
                order_id, patient_id, encounter_id, date_ordered, 
                target_date, order_type, description, urgency, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            order_id,
            patient_id,
            encounter_id,
            encounter_start.isoformat(),  # When it was ordered
            target_date.isoformat(),  # When it should happen
            order.order_type,
            order.description,
            order.urgency,
            'PENDING'  # Always starts as PENDING
        ))

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
