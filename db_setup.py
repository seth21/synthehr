import sqlite3

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
            cause_of_death TEXT
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
            string_value TEXT,
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

    # 5. MEDICATIONS (The Pharmacy Ledger)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medications (
            medication_id TEXT PRIMARY KEY,
            encounter_id TEXT,
            patient_id TEXT,
            timestamp TEXT,               -- ISO-8601 Datetime
            action TEXT,                  -- 'START', 'STOP', 'DOSE_CHANGE'
            rxcui TEXT,
            medication_name TEXT,
            dosage TEXT,
            unit TEXT,
            reason TEXT,
            FOREIGN KEY(encounter_id) REFERENCES encounters(encounter_id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Time-series Database '{db_name}' initialized successfully.")

if __name__ == "__main__":
    init_db()