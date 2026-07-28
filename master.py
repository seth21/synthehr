from typing import List
import uuid
from patient import generate_patient_base, save_patient_to_db
from drift import simulate_latent_gap, fetch_patient_baseline
from clinical_encounter import generate_clinical_encounter, save_encounter_to_db
from datetime import datetime, timedelta
import random
from dateutil.relativedelta import relativedelta
import sqlite3

def update_medication_list(active: dict, changes: list, current_time: datetime) -> dict:
    """Updates active medications and calculates expiration dates for acute drugs."""
    updated = dict(active)

    for m in changes:
        key = m.medication_name.lower().strip()

        if m.action == "START":
            # Calculate the exact expiration date if the LLM provided a duration
            end_date = current_time + timedelta(days=m.duration_days) if m.duration_days else None
            updated[key] = {
                "display": f"{m.medication_name} {m.dosage}{m.unit}",
                "end_date": end_date
            }

        elif m.action == "DOSE_CHANGE":
            if key in updated:
                # Keep the old expiration date unless the LLM provided a new one
                end_date = current_time + timedelta(days=m.duration_days) if m.duration_days else updated[key][
                    "end_date"]
                updated[key] = {
                    "display": f"{m.medication_name} {m.dosage}{m.unit}",

                    "end_date": end_date
                }
            else:
                print(f"  ⚠️ REJECTED: LLM attempted to DOSE_CHANGE {m.medication_name} but it is not active.")

        elif m.action == "STOP" and key in updated:
            del updated[key]

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

    # Generate seed, genetics, and SDOH baseline
    # Initialize the Macro Clock (Random DOB in the given year)
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
    current_age = current_time.year - dob.year
    is_alive = True
    active_conditions: dict = {}
    active_medications: dict = {}
    encounter_count = 0

    # --- THE LIFE SIMULATION LOOP ---
    while is_alive and current_age < max_age:
        encounter_count += 1
        print(f"\n--- [Cycle #{encounter_count}] Patient Age: {current_age:.1f} ---")

        # Fetch the Calendar
        pending_orders = fetch_pending_orders(patient_id)
        if pending_orders:
            print(f"  Calendar: {len(pending_orders)} pending order(s) found.")

        # LATENT DRIFT: Calculate unobserved time & silent pathology
        drift = simulate_latent_gap(patient_data, current_age, active_conditions, pending_orders)

        # FIX for LLM hallucination: Force exact calendar math for fulfilled orders
        if drift.event_type == "FULFILLED_ORDER" and drift.fulfilled_order_id:
            # Find the target order in our local list
            target_order = next((o for o in pending_orders if o['order_id'] == drift.fulfilled_order_id), None)
            if target_order:
                # Override the generic trigger with the exact medical order
                drift.triggering_event = f"Scheduled Visit to fulfill order: {target_order['description']} ({target_order['order_type']})"
                # We also calculate the exact days passed based on the calendar here
                target_date = datetime.fromisoformat(target_order['target_date'])
                drift.days_passed = max(1, (target_date - current_time).days)

        # Advance the absolute clock (Convert years to days)
        days_passed = max(1, drift.days_passed)

        # Safety net to avoid too small gaps for non-compliant patients
        if drift.event_type == "MISSED_ORDERS_AND_DELAYED_VISIT" and days_passed < 30:
            # If the LLM tries to bring a non-compliant patient back in 3 days, override it.
            days_passed = random.randint(90, 365)
            print(f"  SYSTEM OVERRIDE: Forced realistic time gap of {days_passed} days for missed visit.")
        elif days_passed <= 0:
            days_passed = random.randint(30, 90)

        current_time += timedelta(days=days_passed, hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
        if current_time >= datetime.today(): break
        current_age = relativedelta(current_time, dob).years

        print(f"  └ Time Jump (approximately): +{days_passed} days -> Age {current_age:.1f}")
        print(f"  └ Triggering Event: {drift.triggering_event}")
        print(f"  └ Event Type: {drift.event_type}")
        print(f"  └ Silent Pathology: {drift.silent_pathology_accumulated}")

        # Update Queue Statuses based on LLM's compliance routing
        if drift.missed_order_ids:
            update_order_statuses(drift.missed_order_ids, 'MISSED')
            print(f"  ⚠️ MISSED APPOINTMENTS: {len(drift.missed_order_ids)} order(s) abandoned.")

        if drift.event_type == "FULFILLED_ORDER" and drift.fulfilled_order_id:
            update_order_statuses([drift.fulfilled_order_id], 'FULFILLED')

        # Temporal Garbage Collector for orders
        # Sweep the calendar: if we time-traveled past a pending order's target date, auto-miss it.
        auto_missed_orders_count = 0
        for order in pending_orders:
            target_date = datetime.fromisoformat(order['target_date'])
            if current_time > target_date:
                # If it wasn't explicitly handled by the LLM this cycle, force it to MISSED
                if order['order_id'] not in drift.missed_order_ids and drift.fulfilled_order_id != order['order_id']:
                    update_order_statuses([order['order_id']], 'MISSED')
                    auto_missed_orders_count += 1
        if auto_missed_orders_count > 0:
            print(f"  SYSTEM SWEEP: Auto-expired {auto_missed_orders_count} past-due order(s) as MISSED.")
        # Medication Garbage Collector
        expired_meds = []
        # Convert items to list so we can safely delete from the dictionary while iterating
        for key, data in list(active_medications.items()):
            if data["end_date"] and current_time > data["end_date"]:
                expired_meds.append(data["display"])
                del active_medications[key]
        if expired_meds:
            print(f"  SYSTEM SWEEP: Auto-stopped {len(expired_meds)} expired acute medication(s): {', '.join(expired_meds)}")


        # CLINICAL ENCOUNTER: Generate visit, diagnoses, and notes
        encounter = generate_clinical_encounter(
            patient_data, drift, current_age, active_conditions, active_medications
        )

        # Normalize business hours for non-emergencies
        if drift.event_type != "UNSCHEDULED_ACUTE_EVENT":
            # Set the clock to a random time between 8:00 AM and 4:00 PM
            business_hour = random.randint(8, 16)
            business_minute = random.randint(0, 59)
            current_time = current_time.replace(hour=business_hour, minute=business_minute)

        # Calculate Micro-Timestamps dynamically
        encounter_start = current_time
        encounter_end = encounter_start + timedelta(minutes=encounter.total_encounter_duration_minutes)
        print(f"\n--- ENCOUNTER AT {encounter_start.strftime('%Y-%m-%d %H:%M')} ---")

        # Save encounter to SQLite database
        save_encounter_to_db(patient_id, encounter_start, encounter)

        # UPDATE ACTIVE STATE
        # Add new diagnoses to ongoing problem list or remove resolved ones
        active_conditions = update_condition_list(active_conditions, encounter.condition_changes)

        # Update active medication regimen
        active_medications = update_medication_list(active_medications, encounter.medication_changes, encounter_end)

        for obs in encounter.observations:
            obs_time = encounter_start + timedelta(minutes=obs.offset_minutes)
            print(f"[{obs_time.strftime('%H:%M')}] Observation: {obs.name} = {obs.value} | {obs.unit}")

        for cond in encounter.condition_changes:
            cond_time = encounter_start + timedelta(minutes=cond.offset_minutes)
            print(f"[{cond_time.strftime('%H:%M')}] Diagnosis: {cond.icd10_code} ({cond.action})")

        for med in encounter.medication_changes:
            med_time = encounter_start + timedelta(minutes=med.offset_minutes)
            print(f"[{med_time.strftime('%H:%M')}] Prescription: {med.medication_name} ({med.action})")

        for order in encounter.orders_placed:
            print(f"[Target days: {order.target_days_from_now}] Order: {order.description}")
        formatted_conditions = [f"[{k}] {v}" for k, v in active_conditions.items()]
        print(f"  └ Updated Problem List: {formatted_conditions}")
        print(f"  └ Updated Medication List: {active_medications}")
        print(f"[{encounter_end.strftime('%H:%M')}] Encounter completed & note signed.")
        # MORTALITY CHECK
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

def fetch_pending_orders(patient_id: str, db_name="synthetic_ehr.db") -> List[dict]:
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

if __name__ == "__main__":
    # Run a full life simulation loop
    run_lifetime_simulation(
        birth_year=random.randint(1945, 2010),
        gender=random.choice(["Male", "Female"]),
        ethnicity=random.choice(["Caucasian", "Hispanic", "African", "Asian"]),
        start_age=random.randint(15, 85),
    )