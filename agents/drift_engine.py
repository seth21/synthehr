import json
from LLM import LLMService
from core.classes import LatentSubclinicalProgress
from typing import List

# --- THE DRIFT ENGINE ---
def simulate_latent_gap(llm: LLMService, patient_data: dict, current_age: float,
                        active_conditions: dict, pending_orders: List[dict]) -> LatentSubclinicalProgress:
    """Calculates biological and social drift over unobserved time."""
    # Safely parse the PMH and Diagnostics
    sig_diag_raw = patient_data.get('significant_diagnostics')
    sig_diag_list = json.loads(sig_diag_raw) if sig_diag_raw else []

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
    PAST MEDICAL & SURGICAL HISTORY: {patient_data.get('pmh_summary', 'No significant history.')}
    MAJOR PAST DIAGNOSTIC BASELINES: {sig_diag_list if sig_diag_list else 'No significant baseline diagnostics on file.'}
    PENDING ORDERS CALENDAR: {pending_orders if pending_orders else "No pending orders."}
    BEHAVIOR: {patient_data['behavior']}
    GENETIC RISKS: {patient_data['genetics'].get('genetic_risk_factors', [])}
    SDOH STRESSORS: {patient_data['sdoh'].get('current_life_stressors', [])}
    FINANCIAL STRAIN: {patient_data['sdoh'].get('financial_strain', 'Unknown')}

    INSTRUCTIONS
    1. TIME JUMP: Decide how many days pass until the next medical encounter. If they are compliant, they will usually attend their next pending order on time. If they are a care avoider or have severe financial strain, they may miss orders, delaying care by months or years.
    2. EVENT DECISION: Determine the reason for the next visit. Choose exactly ONE of these paths:
       - FULFILLED_ORDER: They attend a scheduled pending order. In this case, advance days_passed by the target timeframe.
       - MISSED_ORDERS_AND_DELAYED_VISIT: They skip their pending orders and only show up much later, which could be months or years. List the missed IDs.
       - UNSCHEDULED_ACUTE_EVENT: A sudden medical crisis forces an ER/Urgent visit. This could be before their scheduled orders or after if they were missed.
       - ROUTINE_GAP: If no pending orders exist, time passes until they randomly seek routine care.
    3. BEHAVIORAL LOGIC: Heavily weigh their Persona and Financial Strain. A patient living paycheck-to-paycheck might abandon an expensive elective surgery or specialist referral.
       CRITICAL: If they skip, they do NOT return a few days later. You MUST advance days_passed by months or years (e.g., 90 to 700+ days) until their silent pathology causes a severe acute event.
    4. SURGICAL AWARENESS: Read the Past Medical & Surgical History carefully. DO NOT generate acute events for organs that have been surgically removed (e.g., no appendicitis if they had an appendectomy).
    5. SILENT PATHOLOGY: Generate a short list of underlying biological changes happening during this time gap (e.g., 'worsening arterial plaque', 'unchecked hyperglycemia'). Base this on their Active Conditions, History, and Genetics.
    6. TRIGGERING EVENT: Write a brief, specific string explaining exactly what brings them into the clinic today (e.g., "3 days of severe right lower quadrant pain", "Routine Annual Checkup").
    7. DEATH: If the patient is in critical condition and especially if they are old with a lot of comorbidities they might die. If they die set survival_status to 'FATAL_EVENT', otherwise set to 'SURVIVED'.
    8. WILDCARD EVENTS: People get sick randomly sometimes, like 10% of the time. When simulating an 'UNSCHEDULED_ACUTE_EVENT' or forcing a visit due to missed orders, do NOT always use heart attacks or diabetes complications. Consider random acute events like:
       - Orthopedic injuries (e.g. sprains, fractures, back pain)
       - Infectious diseases (e.g. Pneumonia, UTI, severe Gastroenteritis, Cellulitis)
       - Dermatological issues (e.g. Severe allergic reactions, unknown rashes)
       - Gastrointestinal crises (e.g. Appendicitis, Gallstones)
    """

    print(f"Simulating time drift for patient at age {current_age}...")

    drift_state = llm.complete(response_model=LatentSubclinicalProgress,
        messages=[
            {"role": "system", "content": "You are a deterministically grounded medical AI. Output valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5)

    return drift_state
