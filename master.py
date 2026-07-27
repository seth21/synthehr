from typing import List
import uuid
from patient import generate_patient_base, save_patient_to_db
from drift import simulate_latent_gap, fetch_patient_baseline
from clinical_encounter import generate_clinical_encounter, save_encounter_to_db
from datetime import datetime, timedelta
import random
from dateutil.relativedelta import relativedelta

def update_medication_list(active_meds: dict, med_changes: list) -> dict:
    """Helper function to apply START, STOP, or DOSE_CHANGE to the active medication list."""
    updated = dict(active_meds)
    for change in med_changes:
        if change.action in ["START", "DOSE_CHANGE"]:
            # START and DOSE_CHANGE both simply overwrite the value at that RxCUI key
            updated[change.rxcui] = f"{change.medication_name} {change.dosage}{change.unit}"
        elif change.action == "STOP":
            # Remove the medication if it exists
            if change.rxcui in updated:
                del updated[change.rxcui]
    return updated

def update_condition_list(active_conditions: dict, condition_changes: list) -> dict:
    """Helper function to apply START or RESOLVE to the active problem list."""
    updated = dict(active_conditions)
    for change in condition_changes:
        if change.action == "START":
            # Add or overwrite, preventing duplicates
            updated[change.icd10_code] = change.condition_name
        elif change.action == "RESOLVE":
            # Remove the acute condition if it exists
            if change.icd10_code in updated:
                del updated[change.icd10_code]
    return updated


def run_lifetime_simulation(
        birth_year: int = 1970,
        gender: str = "Male",
        ethnicity: str = "African American",
        start_age: float = 18.0,
        max_age: float = 100.0
):
    """Executes a full human life cycle simulation from start_age until death or max_age."""

    patient_id = str(uuid.uuid4())
    print(f"\n==================================================")
    print(f" STARTING LIFETIME SIMULATION FOR PATIENT {patient_id[:8]}")
    print(f"==================================================")

    # --- STEP 2: INITIALIZATION ---
    # Generate seed, genetics, and SDOH baseline
    # 1. Initialize the Macro Clock (Random DOB in the given year)
    dob = datetime(birth_year, random.randint(1, 12), random.randint(1, 28))
    base_state = generate_patient_base(birth_year, gender, ethnicity)
    save_patient_to_db(patient_id, dob, base_state)

    # Load initialized patient data from DB
    patient_data = fetch_patient_baseline(patient_id)
    print(f"Full Name: {patient_data['first_name']} {patient_data['last_name']}")
    print(f"Demographics: {patient_data['demographics']}")
    print(f"Behavior Persona: {patient_data['behavior']}")
    print(f"Genetic Risks: {patient_data['genetics'].get('genetic_risk_factors')}")
    print(f"Financial Strain: {patient_data['sdoh'].get('financial_strain')}")
    print("--------------------------------------------------\n")

    # --- LIFETIME LOOP STATE ---
    # Start the simulation at start_age
    current_time = dob + timedelta(days=start_age * 365.25 + random.randint(0, 364))
    #current_age = start_age
    current_age = current_time.year - dob.year
    is_alive = True
    active_conditions: dict = {}
    active_medications: dict = {}
    encounter_count = 0

    # --- STEPS 3 & 4: THE LIFE SIMULATION LOOP ---
    while is_alive and current_age < max_age:
        encounter_count += 1
        print(f"\n--- [Cycle #{encounter_count}] Patient Age: {current_age:.1f} ---")

        # 1. LATENT DRIFT (Step 3): Calculate unobserved time & silent pathology
        drift = simulate_latent_gap(patient_data, current_age, active_conditions)

        # Advance the absolute clock (Convert years to days)
        days_passed = int(drift.unobserved_years_passed * 365.25 + random.randint(-182, 182))
        current_time += timedelta(days=days_passed, hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))

        # Enforce minimum aging safety check (prevents infinite loops if model outputs 0 years)
        #print(drift)
        #if drift.unobserved_years_passed <= 0.1:
        #    drift.patient_age_at_end_of_gap = round(current_age + 0.5, 1)
        current_age = relativedelta(current_time, dob).years

        print(f"  └ Time Jump (approximately): +{drift.unobserved_years_passed:.1f} yrs -> Age {current_age:.1f}")
        print(f"  └ Triggering Event: {drift.triggering_event}")
        print(f"  └ Silent Pathology: {drift.silent_pathology_accumulated}")

        # 2. CLINICAL ENCOUNTER (Step 4): Generate visit, diagnoses, and notes
        encounter = generate_clinical_encounter(
            patient_data, drift, current_age, active_conditions, active_medications
        )

        # C. Calculate Micro-Timestamps dynamically
        encounter_start = current_time
        encounter_end = encounter_start + timedelta(minutes=encounter.total_encounter_duration_minutes)
        print(f"\n--- ENCOUNTER AT {encounter_start.strftime('%Y-%m-%d %H:%M')} ---")
        # Save encounter to SQLite database
        save_encounter_to_db(patient_id, encounter_start, encounter)

        # 3. UPDATE ACTIVE STATE
        # Add new diagnoses to ongoing problem list or remove resolved ones
        active_conditions = update_condition_list(active_conditions, encounter.condition_changes)

        # Update active medication regimen
        active_medications = update_medication_list(active_medications, encounter.medication_changes)

        for obs in encounter.observations:
            obs_time = encounter_start + timedelta(minutes=obs.offset_minutes)
            print(f"[{obs_time.strftime('%H:%M')}] Observation: {obs.name} = {obs.value} | {obs.string_value}")

        for cond in encounter.condition_changes:
            cond_time = encounter_start + timedelta(minutes=cond.offset_minutes)
            print(f"[{cond_time.strftime('%H:%M')}] Diagnosis: {cond.icd10_code} ({cond.action})")

        for med in encounter.medication_changes:
            med_time = encounter_start + timedelta(minutes=med.offset_minutes)
            print(f"[{med_time.strftime('%H:%M')}] Prescription: {med.medication_name} ({med.action})")
        formatted_conditions = [f"[{k}] {v}" for k, v in active_conditions.items()]
        print(f"  └ Updated Problem List: {formatted_conditions}")
        print(f"  └ Updated Medication List: {active_medications}")
        print(f"[{encounter_end.strftime('%H:%M')}] Encounter completed & note signed.")
        # 4. MORTALITY CHECK
        if drift.survival_status == "FATAL_EVENT" or encounter.primary_cause_of_death:
            is_alive = False
            cause = encounter.primary_cause_of_death or drift.triggering_event
            print(f"\n💀 TERMINAL EVENT OCCURRED AT AGE {current_age:.1f}")
            print(f"   Primary Cause of Death: {cause}")
            break

    print(f"\n==================================================")
    print(f"🏁 SIMULATION COMPLETE FOR PATIENT {patient_id[:8]}")
    print(f"   Final Age: {current_age:.1f} | Encounters Logged: {encounter_count}")
    print(f"==================================================\n")


if __name__ == "__main__":
    # Run a full life simulation loop
    run_lifetime_simulation(
        birth_year=random.randint(1945, 2010),
        gender=random.choice(["Male", "Female"]),
        ethnicity=random.choice(["Caucasian", "Hispanic", "African", "Asian"]),
        start_age=random.randint(15, 85),
    )