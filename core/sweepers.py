from datetime import datetime
from core.classes import LatentSubclinicalProgress
from db.queries import update_order_statuses

def expire_missed_orders(cursor, pending_orders: list[dict], drift: LatentSubclinicalProgress, current_time):
    """Marks orders as MISSED if their target date has passed."""
    auto_missed_orders_count = 0
    for order in pending_orders:
        target_date = datetime.fromisoformat(order['target_date'])
        if current_time > target_date:
            # If it wasn't explicitly handled by the LLM at that cycle, force it to MISSED
            if order['order_id'] not in drift.missed_order_ids and drift.fulfilled_order_id != order['order_id']:
                update_order_statuses([order['order_id']], 'MISSED')
                auto_missed_orders_count += 1
    if auto_missed_orders_count > 0:
        print(f"  SYSTEM SWEEP: Auto-expired {auto_missed_orders_count} past-due order(s) as MISSED.")

def stop_expired_medications(cursor, active_medications, current_time):
    """Auto-stops acute medications (like antibiotics) once their duration passes."""
    expired_meds = []
    # Convert items to list so we can safely delete from the dictionary while iterating
    for key, data in list(active_medications.items()):
        if data["end_date"] and current_time > data["end_date"]:
            expired_meds.append(data["display"])
            del active_medications[key]
    if expired_meds:
        print(
            f"  SYSTEM SWEEP: Auto-stopped {len(expired_meds)} expired acute medication(s): {', '.join(expired_meds)}")