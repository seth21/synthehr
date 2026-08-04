from datetime import datetime, timedelta
from core.classes import ClinicalEncounterResult


def calc_days_left_for_pending_orders(pending_orders: list[dict], current_time) -> str:
    calendar_list = []
    for order in pending_orders:
        target_date = datetime.fromisoformat(order["target_date"])
        days_away = (target_date - current_time).days
        if days_away >= 0:
            calendar_list.append(f"- {order["description"]} (Scheduled in {days_away} days)")
    calendar_text = "\n".join(calendar_list) if calendar_list else "None scheduled."
    return calendar_text

def print_encounter_details(encounter: ClinicalEncounterResult, encounter_start, encounter_end, active_conditions, active_medications) -> None:
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