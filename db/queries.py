import json
import sqlite3
import uuid
from datetime import timedelta, datetime
from typing import List

from core.classes import ClinicalEncounterResult, PatientBaseState
from db.database import db

# ==========================================
# READ OPERATIONS (Fetching State)
# ==========================================

def get_patient_baseline(patient_id: str, db_name="synthetic_ehr.db"):
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

def get_active_conditions(cursor, patient_id: str) -> list:
    """Returns a list of actively managed ICD-10 conditions."""
    cursor.execute('''
        SELECT icd10_code, condition_name 
        FROM conditions 
        WHERE patient_id = ? AND status = 'ACTIVE'
    ''', (patient_id,))
    return [f"[{row[0]}] {row[1]}" for row in cursor.fetchall()]

def get_active_medications(cursor, patient_id: str) -> List[dict]:
    """Returns currently active medications and dosages."""
    cursor.execute("""
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

    cols = [desc[0] for desc in cursor.description]
    meds = [dict(zip(cols, row)) for row in cursor.fetchall()]

    return meds

def get_pending_orders(patient_id: str, db_name="synthetic_ehr.db") -> List[dict]:
    """Retrieves all active orders on the patient's calendar."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT order_id, target_date, order_type, description, urgency 
        FROM pending_orders 
        WHERE patient_id = ? AND status = 'PENDING'
    """, (patient_id,))
    cols = [desc[0] for desc in cursor.description]
    orders = [dict(zip(cols, row)) for row in cursor.fetchall()]
    conn.close()
    return orders

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

# ==========================================
# WRITE OPERATIONS (Saving State)
# ==========================================

def update_order_statuses(order_ids: List[str], new_status: str, db_name="synthetic_ehr.db"):
    """Marks orders as 'FULFILLED' or 'MISSED' in the database."""
    if not order_ids:
        return
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    placeholders = ','.join(['?'] * len(order_ids))
    # We pass new_status first, then unpack the list of order_ids
    cursor.execute(f"UPDATE pending_orders SET status = ? WHERE order_id IN ({placeholders})", [new_status] + order_ids)
    conn.commit()
    conn.close()

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
        # Create the new medication if it does NOT exist in the database
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
        # Get the medication_id of the drug
        row = cursor.execute("""
                SELECT medication_id
                FROM medications
                WHERE rxcui = ?
            """, (med.rxcui,)).fetchone()
        actual_medication_id = row[0]

        # Create the new medication change event
        cursor.execute('''
                INSERT INTO medication_events (medication_event_id, encounter_id, patient_id, medication_id, timestamp, action, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (str(uuid.uuid4()), encounter_id, patient_id, actual_medication_id, med_time.isoformat(), med.action,
                  med.reason))

        # Update the current patient medications
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


def update_patient_memory(cursor, patient_id: str, memory_data: dict):
    """Saves the rolling PMH and Diagnostic JSON arrays."""
    pmh = memory_data['pmh_summary']
    diag_json = json.dumps(memory_data['significant_diagnostics'])

    cursor.execute('''
        UPDATE patients 
        SET pmh_summary = ?, significant_diagnostics = ? 
        WHERE patient_id = ?
    ''', (pmh, diag_json, patient_id))