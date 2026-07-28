import sqlite3
import json
from classes import LatentSubclinicalProgress
from typing import List
from openai import OpenAI
import instructor

# --- OLLAMA CLIENT SETUP ---
client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON,
)
MODEL = "gemma4:e4b"

# --- DATABASE FETCH LOGIC ---
def fetch_patient_baseline(patient_id: str, db_name="synthetic_ehr.db"):
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


# --- THE DRIFT ENGINE ---
def simulate_latent_gap(patient_data: dict, current_age: float,
                        active_conditions: dict, pending_orders: List[dict]) -> LatentSubclinicalProgress:
    """Calculates biological and social drift over unobserved time."""
    # Dynamic Age & Epidemiology Guardrails
    if current_age < 35 and not active_conditions:
        age_guardrail = """
            EPIDEMIOLOGY GUARDRAIL: This patient is young and generally healthy. 
            - TIME GAPS MUST BE LONG: 1 to 5 years (365 to 1800 days) between visits.
            - NO CHRONIC DISEASES: Do NOT accumulate silent pathology like hypertension, diabetes, or heart disease unless specifically driven by their genetics. 
            - VALID TRIGGERS: Routine physicals, minor sports injuries, simple infections (e.g., strep throat, UTI), or completely healthy gaps.
            """
    elif current_age < 50:
        age_guardrail = "EPIDEMIOLOGY GUARDRAIL: Patient is middle-aged. Chronic lifestyle diseases (hypertension, pre-diabetes) may slowly begin to emerge depending on SDOH and behavior."
    else:
        age_guardrail = "EPIDEMIOLOGY GUARDRAIL: Patient is older. Age-related decline, chronic diseases, and more frequent visits are expected."

    prompt = f"""
    You are evaluating a patient's timeline and compliance.
    
    PATIENT DEMOGRAPHICS: {patient_data['demographics']}
    AGE: {current_age}
    ACTIVE CONDITIONS: {active_conditions}
    {age_guardrail}
    PENDING ORDERS CALENDAR: {pending_orders if pending_orders else "No pending orders."}
    BEHAVIOR: {patient_data['behavior']}
    GENETIC RISKS: {patient_data['genetics'].get('genetic_risk_factors', [])}
    SDOH STRESSORS: {patient_data['sdoh'].get('current_life_stressors', [])}
    FINANCIAL STRAIN: {patient_data['sdoh'].get('financial_strain', 'Unknown')}

    Based on their calendar, biological anchors and social reality:
    1. Document the silent pathology accumulating in the dark.
    2. Determine the triggering event that ends the gap.
    3. If there are pending orders, does the patient's behavior and SDOH allow them to attend on time? (Care Avoiders or severe poverty often miss routine/specialist visits).
    4. If they attend an order, set event_type to 'FULFILLED_ORDER' and advance days_passed by the target timeframe.
    5. If they skip/no-show, set event_type to 'MISSED_ORDERS_AND_DELAYED_VISIT' and list the missed IDs. 
    CRITICAL: If they skip, they do NOT return a few days later. You MUST advance days_passed by months or years (e.g., 90 to 700+ days) until their silent pathology causes a severe acute event.
    6. If an acute emergency happens BEFORE a scheduled visit, set to 'UNSCHEDULED_ACUTE_EVENT'.
    7. Decide if they SURVIVED after the gap or it led to a FATAL_EVENT.
    COMPLIANCE RULES: If the patient is HIGHLY_COMPLIANT and financially secure, they almost ALWAYS attend scheduled orders. Only CARE_AVOIDER or SEVERE_POVERTY patients should routinely miss appointments.
    
    WILDCARD EVENTS: People get sick randomly sometimes, like 10% of the time. When simulating an 'UNSCHEDULED_ACUTE_EVENT' or forcing a visit due to missed orders, do NOT always use heart attacks or diabetes complications. Consider random acute events like:
    - Orthopedic injuries (e.g. sprains, fractures, back pain)
    - Infectious diseases (e.g. Pneumonia, UTI, severe Gastroenteritis, Cellulitis)
    - Dermatological issues (e.g. Severe allergic reactions, unknown rashes)
    - Gastrointestinal crises (e.g. Appendicitis, Gallstones)
    """

    print(f"Simulating time drift for patient at age {current_age}...")

    drift_state = client.chat.completions.create(
        model=MODEL,
        response_model=LatentSubclinicalProgress,
        messages=[
            {"role": "system", "content": "You are a deterministically grounded medical AI. Output valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5

    )
    return drift_state
