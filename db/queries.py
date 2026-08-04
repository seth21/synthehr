import json
import uuid
from datetime import timedelta, datetime
from sqlite3 import Row
from typing import List

from core.classes import ClinicalEncounterResult, PatientBaseState
from db.database import Database

# ==========================================
# READ OPERATIONS (Fetching State)
# ==========================================

def get_patient_profile(db: Database, patient_id: str):
    """Pulls the patient's seed data from SQLite and parses the JSON strings."""
    row = db.execute_fetch_one("""
            SELECT first_name, last_name, dob, demographics, behavioral_profile, genetic_risks, current_sdoh 
            FROM patients WHERE patient_id = ?
        """, (patient_id,))
    if not row:
        raise ValueError(f"Patient {patient_id} not found.")

    return {
        "first_name": row[0],
        "last_name": row[1],
        "dob": row[2],
        "demographics": row[3],
        "behavior": row[4],
        "genetics": json.loads(row[5]),
        "sdoh": json.loads(row[6])
    }

def get_active_conditions(db: Database, patient_id: str) -> list:
    """Returns a list of actively managed ICD-10 conditions."""
    rows = db.execute_fetch_all('''
        SELECT icd10_code, condition_name 
        FROM conditions 
        WHERE patient_id = ? AND status = 'ACTIVE'
    ''', (patient_id,))
    return [f"[{row['icd10_code']}] {row['condition_name']}" for row in rows]

def get_active_medications(db: Database, patient_id: str) -> list[Row]:
    """Returns currently active medications and dosages."""
    meds = db.execute_fetch_all("""
        SELECT
            pm.medication_id,
            m.rxcui,
            m.medication_name,
            m.dosage,
            m.unit,
            pm.frequency
        FROM patient_medications pm
        JOIN medications m
            ON pm.medication_id = m.medication_id
        WHERE pm.patient_id = ?
          AND pm.status = 'ACTIVE'
        ORDER BY m.medication_name
    """, (patient_id,))

    return meds

def get_pending_orders(db: Database, patient_id: str) -> list[dict]:
    """Retrieves all active orders on the patient's calendar."""
    orders = db.execute_fetch_all_dict("""
        SELECT order_id, target_date, order_type, description, urgency 
        FROM pending_orders 
        WHERE patient_id = ? AND status = 'PENDING'
    """, (patient_id,))
    return orders

def get_latest_observations(db: Database, patient_id, num_observations):
    recent_obs_data = db.execute_fetch_all('''
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

def get_patient_current_time(db: Database, patient_id: str, baseline_time: datetime) -> datetime:
    """Helper to get the current simulated clock from the last encounter."""
    last_date = db.execute_fetch_one("SELECT MAX(start_time) FROM encounters WHERE patient_id = ?", (patient_id,))[0]
    return datetime.fromisoformat(last_date) if last_date else baseline_time

# ==========================================
# WRITE OPERATIONS (Saving State)
# ==========================================

def update_order_statuses(db: Database, order_ids: List[str], new_status: str):
    """Marks orders as 'FULFILLED' or 'MISSED' in the database."""
    if not order_ids:
        return
    placeholders = ','.join(['?'] * len(order_ids))
    # We pass new_status first, then unpack the list of order_ids
    db.execute(f"UPDATE pending_orders SET status = ? WHERE order_id IN ({placeholders})", [new_status] + order_ids)

def save_encounter_to_db(db: Database,
        patient_id: str,
        encounter_start: datetime,
        encounter: ClinicalEncounterResult):

    """Appends the generated encounter to the encounters ledger table in SQLite."""
    encounter_id = str(uuid.uuid4())
    encounter_end = encounter_start + timedelta(minutes=encounter.total_encounter_duration_minutes)

    # 1. Insert Encounter
    db.execute('''
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
        db.execute('''
                INSERT INTO observations (observation_id, encounter_id, patient_id, timestamp, name, value, unit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), encounter_id, patient_id, obs_time.isoformat(), obs.name, obs.value, obs.unit))

    # 3. Insert Condition Changes
    for cond in encounter.condition_changes:
        cond_time = encounter_start + timedelta(minutes=cond.offset_minutes)
        db.execute('''
                INSERT INTO conditions (condition_id, encounter_id, patient_id, timestamp, action, icd10_code, condition_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), encounter_id, patient_id, cond_time.isoformat(), cond.action, cond.icd10_code,
                  cond.condition_name))

    # 4. Insert Medication Changes
    for med in encounter.medication_changes:
        med_time = encounter_start + timedelta(minutes=med.offset_minutes)
        new_medication_id = str(uuid.uuid4())
        # Create the new medication if it does NOT exist in the database
        db.execute("""
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
        # Get the medication_id of the drug
        row = db.execute_fetch_one("""
                SELECT medication_id
                FROM medications
                WHERE rxcui = ?
            """, (med.rxcui,))
        actual_medication_id = row[0]

        # Create the new medication change event
        db.execute('''
                INSERT INTO medication_events (medication_event_id, encounter_id, patient_id, medication_id, timestamp, action, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), encounter_id, patient_id, actual_medication_id, med_time.isoformat(), med.action,
                  med.reason))

        # Update the current patient medications
        db.execute("""
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

        db.execute('''
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
        db.execute('''
            UPDATE patients 
            SET is_alive = False, death_date = ?, cause_of_death = ?
            WHERE patient_id = ?
        ''', (death_time, encounter.primary_cause_of_death, patient_id))
        print(f"⚠️ Patient record updated: DECEASED at {death_time}. Cause: {encounter.primary_cause_of_death}")

    print(f"Encounter {encounter_id} saved to SQLite database successfully.")



def save_patient_to_db(db: Database, patient_id: str, dob: datetime, state: PatientBaseState):
    """Saves the generated patient baseline into SQLite."""
    db.execute('''
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
    print(f"Patient {patient_id} successfully saved to database.")


def save_patient_memory(db: Database, patient_id: str, memory_data: dict):
    """Saves the rolling PMH and Diagnostic JSON arrays."""
    pmh = memory_data['pmh_summary']
    diag_json = json.dumps(memory_data['significant_diagnostics'])

    db.execute('''
        UPDATE patients 
        SET pmh_summary = ?, significant_diagnostics = ? 
        WHERE patient_id = ?
    ''', (pmh, diag_json, patient_id))