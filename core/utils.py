from datetime import datetime


def calc_days_left_for_pending_orders(pending_orders: list[dict], current_time) -> str:
    calendar_list = []
    for order in pending_orders:
        target_date = datetime.fromisoformat(order["target_date"])
        days_away = (target_date - current_time).days
        if days_away >= 0:
            calendar_list.append(f"- {order["description"]} (Scheduled in {days_away} days)")
    calendar_text = "\n".join(calendar_list) if calendar_list else "None scheduled."
    return calendar_text