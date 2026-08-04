from LLM import LLMService
from core.classes import ClinicalEncounterResult
import json

# Import Step 3 functions to pull the latent drift data
from agents.drift_engine import LatentSubclinicalProgress


# --- THE CLINICAL ENCOUNTER ENGINE ---
def generate_clinical_encounter(
        llm: LLMService,
        patient_data: dict,
        latent_drift: LatentSubclinicalProgress,
        patient_age_at_end_of_gap: int,
        active_conditions: dict,
        active_medications: dict,
        patient_id: str,
        still_pending_orders_txt: str,
        last_observations_txt: str,
) -> ClinicalEncounterResult:
    """Generates the structured clinical visit output from the drift event."""
    active_meds_display = [data["display"] for data in active_medications.values()]
    active_conds_display = list(active_conditions.values())

    # Safely parse the significant_diagnostics JSON
    sig_diag = patient_data.get('significant_diagnostics')
    sig_diag_list = json.loads(sig_diag) if sig_diag else []
    # Get past medical history
    recent_obs_text = last_observations_txt

    prompt = f"""
    You are an attending physician documenting a clinical encounter.

    FULL NAME: {patient_data['first_name']} {patient_data['last_name']}
    PATIENT DEMOGRAPHICS: {patient_data['demographics']}
    AGE TODAY: {patient_age_at_end_of_gap}
    SURVIVAL STATUS: {latent_drift.survival_status}

    ACTIVE CONDITIONS PRIOR TO VISIT: {active_conds_display}
    ACTIVE MEDICATIONS PRIOR TO VISIT: {active_meds_display}
    
    PAST MEDICAL & SURGICAL HISTORY: {patient_data.get('pmh_summary', 'No significant history.')}
    MAJOR PAST DIAGNOSTIC BASELINES: {sig_diag_list if sig_diag_list else 'No significant baseline diagnostics on file.'}
    RECENT FLOWSHEET (Last 5 Labs/Vitals): {recent_obs_text}
    FUTURE SCHEDULED ORDERS (DO NOT DUPLICATE THESE): {still_pending_orders_txt}
    REASON FOR VISIT / TRIGGERING EVENT: {latent_drift.triggering_event}
    SILENT PATHOLOGY THAT ACCUMULATED PRIOR TO VISIT: {latent_drift.silent_pathology_accumulated}

    SOCIAL DETERMINANTS OF HEALTH (SDOH):
    - Financial Strain: {patient_data['sdoh'].get('financial_strain')}
    - Stressors: {patient_data['sdoh'].get('current_life_stressors')}

    INSTRUCTIONS:
    1. Write a detailed clinical note with 100-300 words. If SDOH/finances contributed to skipped meds, document it.
    2. Record realistic observations associated with the encounter (e.g. vitals, lab tests, imaging tests).
    3. SPECIALTY ALIGNMENT: Read the REASON FOR VISIT carefully. If this is a specific scheduled order (e.g., "Cardiology Consultation", "Renal Ultrasound", "Colonoscopy"), you MUST act as that specific specialist or department. 
    4. EXECUTE THE ORDER: Ensure your clinical note and observations directly reflect the results of the requested procedure or consultation.
    5. MANAGE THE PROBLEM LIST (ICD-10) taking into account the silent pathology:
       - If a new disease is diagnosed, START it with a valid ICD-10 code.
       - If an acute condition from their active list (e.g., Acute bronchitis, fractures, acute infections) has healed, RESOLVE it using its exact ICD-10 code.
       - Do NOT resolve chronic, incurable diseases (e.g., Type 2 Diabetes, Amputations).
       - CRITICAL: If this is a routine checkup for a young/healthy patient, it is PERFECTLY NORMAL to have NO new diagnoses and NO medication changes. Do not invent pathology if they are healthy.
       - Do NOT output a START action for a diagnosis that is already present in the ACTIVE CONDITIONS list.
    6. Adjust, start or stop medications as needed. Use numeric RxNorm RxCUIs for all medication changes. 
    You cannot STOP or perform a DOSE_CHANGE on a medication unless it is explicitly listed in the ACTIVE MEDICATIONS PRIOR TO VISIT. If starting a new drug, you must use START.
    DIAGNOSIS-MEDICATION LINKAGE: You MUST NOT prescribe a medication (like Metformin) unless the underlying disease (like Diabetes) is explicitly diagnosed and added to the Problem List today or is already active.
    THERAPEUTIC DUPLICATION: If you are switching a patient to a new medication within the same class (e.g., switching from Simvastatin to Atorvastatin), you MUST output a STOP action for the old medication.
    7. If SURVIVAL_STATUS is FATAL_EVENT, document the resuscitation/fatal outcome and populate 'primary_cause_of_death'.
    8. PLAN & NEXT STEPS: If the patient requires future follow-up, specialist evaluation, or imaging, generate orders_placed with realistic timelines (e.g. a 3-month routine follow-up).
    - FUTURE ORDERS ONLY: Use `orders_placed` strictly for appointments or labs happening TOMORROW or later (target_days_from_now >= 1). Do NOT re-order routine labs, consults, or follow-ups if they are already on the "Future Scheduled Orders" list
    - SAME-DAY DIAGNOSTICS: If you need a lab test or imaging study TODAY (e.g., STAT Troponin, CMP, Chest X-Ray), DO NOT put it in `orders_placed`. Instead, assume the test was already performed. Invent realistic results and place them directly into the `observations` array. Evaluate these results in your clinical note.
    - If the patient is young and healthy, the only order might be "Follow up in 1 year" (365 days) or NO orders at all.
    """

    print(f"Generating clinical encounter note for age {patient_age_at_end_of_gap}...")

    encounter = llm.complete(response_model=ClinicalEncounterResult,
        messages=[
            {"role": "system", "content": "You are a professional clinical AI. Output valid JSON."},
            {"role": "user", "content": prompt}
        ])

    return encounter