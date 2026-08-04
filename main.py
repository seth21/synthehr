import random
from dateutil.relativedelta import relativedelta
from agents.drift_engine import simulate_latent_gap
from db.database import Database
from datetime import datetime, timedelta
import json

# Set up your Instructor/OpenAI client for the local model
from openai import OpenAI
import instructor

# Import your modular architecture
from db import queries
from agents import drift_engine, encounter_engine, memory_agent
from core import sweepers, utils
from LLM import LLMService
from patient import initialize_patient


def master_loop():
    # Initialize DB Connection and LLM
    db = Database()
    llm = LLMService()

    # Create a base patient profile
    patient_id = initialize_patient(llm, db, birth_year=1945,
        gender=random.choice(["Male", "Female"]),
        ethnicity=random.choice(["Caucasian", "Hispanic", "African", "Asian"]))

    # Run patient simulation
    run_patient_simulation(db, llm, patient_id, start_age=75)

    db.close()


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


def advance_time(current_time, drift, pending_orders):
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

    new_time = current_time + timedelta(days=days_passed, hours=random.randint(0, 23), minutes=random.randint(0, 59),
                              seconds=random.randint(0, 59))
    # Normalize business hours for non-emergencies
    if drift.event_type != "UNSCHEDULED_ACUTE_EVENT":
        # Set the clock to a random time between 8:00 AM and 4:00 PM
        business_hour = random.randint(8, 16)
        business_minute = random.randint(0, 59)
        new_time = new_time.replace(hour=business_hour, minute=business_minute)

    return new_time


def run_patient_simulation(db: Database, llm: LLMService, patient_id: str, max_cycles: int = 10, start_age: float = 18.0,
        max_age: float = 100.0):
    print(f"==================================================")
    print(f" STARTING LIFETIME SIMULATION FOR {patient_id[:8]}")
    print(f"==================================================")

    # Verify patient exists
    patient = queries.get_patient_profile(db, patient_id)
    if not patient:
        print(f"Error: Patient {patient_id} not found in database.")
        return

    # Set the starting clock
    birth_date = datetime.fromisoformat(patient['dob'])
    patient_start_time = birth_date + timedelta(days=start_age * 365.25 + random.randint(0, 364))
    current_time = queries.get_patient_current_time(db, patient_id, patient_start_time)

    current_age = current_time.year - birth_date.year
    active_conditions: dict = {}
    active_medications: dict = {}
    for cycle in range(max_cycles):
        print(f"\n--- [Cycle #{cycle + 1}] Patient Age: {current_age:.1f} ---")

        # ---------------------------------------------------------
        # A. READ STATE
        # ---------------------------------------------------------
        #active_conds = queries.get_active_conditions(cursor, patient_id)
        #active_meds = queries.get_active_medications(cursor, patient_id)
        pending_orders = queries.get_pending_orders(db, patient_id)
        recent_obs = queries.get_latest_observations(db, patient_id, 5)

        if pending_orders:
            print(f"  Calendar: {len(pending_orders)} pending order(s) found.")

        # ---------------------------------------------------------
        # B. THE DRIFT ENGINE (Simulate passage of time)
        # ---------------------------------------------------------
        print("  Simulating time drift...")
        drift = simulate_latent_gap(llm, patient, current_age, active_conditions, pending_orders)

        # Advance the clock!
        current_time = advance_time(current_time, drift, pending_orders)
        if current_time >= datetime.today(): break
        current_age = relativedelta(current_time, birth_date).years

        print(f"    └ Time Jump: +{drift.days_passed} days -> {current_time.strftime('%Y-%m-%d')}")
        print(f"    └ Age: {current_age:.1f} years old")
        print(f"    └ Event Type: {drift.event_type}")
        print(f"    └ Trigger: {drift.triggering_event}")
        print(f"    └ Silent Pathology: {drift.silent_pathology_accumulated}")

        # Update pending order statuses based on LLM's compliance routing
        if drift.missed_order_ids:
            queries.update_order_statuses(db, drift.missed_order_ids, 'MISSED')
            print(f"  ⚠️ MISSED APPOINTMENTS: {len(drift.missed_order_ids)} order(s) abandoned.")

        if drift.event_type == "FULFILLED_ORDER" and drift.fulfilled_order_id:
            queries.update_order_statuses(db, [drift.fulfilled_order_id], 'FULFILLED')

        # ---------------------------------------------------------
        # C. GARBAGE COLLECTION (Business Rules)
        # ---------------------------------------------------------
        sweepers.expire_missed_orders(db, pending_orders, drift, current_time)
        sweepers.stop_expired_medications(db, active_medications, current_time)

        # ---------------------------------------------------------
        # D. THE CLINICAL ENCOUNTER (Doctor AI)
        # ---------------------------------------------------------
        # Also inject still pending orders by the time of new encounter
        pending_orders_at_encounter = queries.get_pending_orders(db, patient_id)
        calendar_text = utils.calc_days_left_for_pending_orders(pending_orders_at_encounter, current_time)
        print(f"    └ Remaining pending orders at time of encounter: {calendar_text}")
        print("  Generating clinical encounter note...")
        encounter = encounter_engine.generate_clinical_encounter(llm,
            patient, drift, current_age, active_conditions, active_medications, patient_id, calendar_text, recent_obs
        )
        # Calculate Micro-Timestamps dynamically
        encounter_start = current_time
        encounter_end = encounter_start + timedelta(minutes=encounter.total_encounter_duration_minutes)
        print(f"\n--- ENCOUNTER AT {encounter_start.strftime('%Y-%m-%d %H:%M')} ---")

        # ---------------------------------------------------------
        # E. THE MEMORY AGENT (Chart Scribe AI)
        # ---------------------------------------------------------
        print("  Updating longitudinal clinical memory...")
        # Parse old memory safely
        old_memory = {
            'pmh_summary': patient.get('pmh_summary', ''),
            'significant_diagnostics': json.loads(patient.get('significant_diagnostics') or '[]')
        }
        new_memory = memory_agent.update_patient_memory(llm,
            old_memory, encounter.clinical_note, encounter.observations, current_time.isoformat()
        )
        print(f"    └ PMH Updated: {new_memory['pmh_summary']}")
        print(f"    └ Baselines Tracked: {new_memory['significant_diagnostics']}")

        # ---------------------------------------------------------
        # F. WRITE STATE & COMMIT
        # ---------------------------------------------------------
        # UPDATE ACTIVE STATE
        # Add new diagnoses to ongoing problem list or remove resolved ones
        active_conditions = update_condition_list(active_conditions, encounter.condition_changes)
        # Update active medication regimen
        active_medications = update_medication_list(active_medications, encounter.medication_changes, encounter_end)
        with db.transaction():
            queries.save_encounter_to_db(db, patient_id, encounter_start, encounter)
            queries.save_patient_memory(db, patient_id, new_memory)

        # Update local patient dictionary with the new memory for the next loop iteration
        patient['pmh_summary'] = new_memory['pmh_summary']
        patient['significant_diagnostics'] = json.dumps(new_memory['significant_diagnostics'])
        #Print encounter details
        utils.print_encounter_details(encounter, encounter_start, encounter_end, active_conditions, active_medications)

        # ---------------------------------------------------------
        # G. MORTALITY CHECK
        # ---------------------------------------------------------
        if drift.survival_status == "FATAL_EVENT" or encounter.primary_cause_of_death:
            cause = encounter.primary_cause_of_death or drift.triggering_event
            print(f"\n💀 TERMINAL EVENT OCCURRED AT AGE {current_age:.1f}")
            print(f"   Primary Cause of Death: {cause}")
            break

        print(f"  [✓] Cycle Complete. Data saved to SQLite.")

    print(f"\n==================================================")
    print(f"🏁 SIMULATION COMPLETE FOR PATIENT {patient_id[:8]}")
    print(f"   Final Age: {current_age:.1f}")
    print(f"==================================================\n")


if __name__ == "__main__":
    master_loop()
