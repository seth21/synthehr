import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "synthetic_ehr.db"

def init_db(db_name="synthetic_ehr.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 1. PATIENTS (Now with full DOB and exact Death Date)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            dob TEXT,                     -- ISO-8601 Date: YYYY-MM-DD
            demographics TEXT,
            behavioral_profile TEXT,
            genetic_risks TEXT,           -- JSON string
            current_sdoh TEXT,            -- JSON string
            is_alive BOOLEAN,
            death_date TEXT,              -- ISO-8601 Datetime
            cause_of_death TEXT,
            pmh_summary TEXT, -- The narrative history
            significant_diagnostics TEXT -- JSON string of major baselines
        )
    ''')

    # 2. ENCOUNTERS (The parent event)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS encounters (
            encounter_id TEXT PRIMARY KEY,
            patient_id TEXT,
            encounter_type TEXT,
            start_time TEXT,              -- ISO-8601 Datetime
            end_time TEXT,                -- ISO-8601 Datetime
            reason_for_visit TEXT,
            clinical_note TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    ''')

    # 3. OBSERVATIONS (Vitals & Labs as time-series data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            observation_id TEXT PRIMARY KEY,
            encounter_id TEXT,
            patient_id TEXT,
            timestamp TEXT,               -- ISO-8601 Datetime
            name TEXT,
            value TEXT,
            unit TEXT,
            FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id)
        )
    ''')

    # 4. CONDITIONS (The Problem List Ledger)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conditions (
            condition_id TEXT PRIMARY KEY,
            encounter_id TEXT,
            patient_id TEXT,
            timestamp TEXT,               -- ISO-8601 Datetime
            action TEXT,                  -- 'START' or 'RESOLVE'
            icd10_code TEXT,
            condition_name TEXT,
            FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id)
        )
    ''')

    # 5. MEDICATIONS
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS medications (
                medication_id TEXT PRIMARY KEY,
                rxcui TEXT UNIQUE,
                medication_name TEXT,
                dosage TEXT,
                unit TEXT,
                drug_class TEXT,
                therapeutical_class TEXT,
                form TEXT
            )
        ''')

    # 6. MEDICATION EVENTS
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medication_events (
            medication_event_id TEXT PRIMARY KEY,
            encounter_id,
            patient_id TEXT,
            medication_id,
            timestamp TEXT,               -- ISO-8601 Datetime
            action TEXT,                  -- 'START', 'STOP', 'DOSE_CHANGE'
            reason TEXT,
            route TEXT,
            frequency TEXT,
            FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id),
            FOREIGN KEY(medication_id) REFERENCES medications(medication_id)
        )
    ''')

    # 7. CURRENT MEDICATIONS
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_medications (
                patient_medication_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                medication_id TEXT NOT NULL,
                start_date TEXT,               -- ISO-8601 Datetime
                end_date TEXT,               -- ISO-8601 Datetime
                updated_at TEXT,               -- ISO-8601 Datetime
                status TEXT,                 -- 'ACTIVE', 'STOPPED', 'ON_HOLD'
                route TEXT,
                frequency TEXT,
                FOREIGN KEY(patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY(medication_id) REFERENCES medications(medication_id),
                UNIQUE(patient_id, medication_id)
            )
        ''')

    # 8. PENDING ORDERS (The Care Pathway Queue)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pending_orders (
            order_id TEXT PRIMARY KEY,
            patient_id TEXT,
            encounter_id TEXT,            -- The visit where this was ordered
            date_ordered TEXT,            -- ISO-8601 Datetime
            target_date TEXT,             -- ISO-8601 Datetime
            order_type TEXT,              -- 'IMAGING', 'LAB_DRAW', 'REFERRAL', etc.
            description TEXT,
            urgency TEXT,                 -- 'ROUTINE', 'URGENT', 'STAT'
            status TEXT,                  -- 'PENDING', 'FULFILLED', 'MISSED'
            FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id),
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Time-series Database '{db_name}' initialized successfully.")

if __name__ == "__main__":
    init_db(DB_PATH)