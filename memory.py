from LLM import llm_service
from classes import PatientMemory

def update_patient_memory(old_memory: dict, encounter_note: str, encounter_observations: list,
                          current_date: str) -> dict:
    # Format today's observations for the LLM to review
    obs_strings = [f"{o.name}: {o.value} {o.unit or ''}" for o in encounter_observations]

    prompt = f"""
    You are a medical scribe updating a patient's permanent chart.

    CURRENT DATE: {current_date}

    EXISTING PMH NARRATIVE: {old_memory.get('pmh_summary', 'None')}
    EXISTING SIGNIFICANT DIAGNOSTICS: {old_memory.get('significant_diagnostics', [])}

    TODAY'S VISIT NOTE:
    {encounter_note}

    TODAY'S OBSERVATIONS (Labs/Imaging/Vitals):
    {obs_strings}

    INSTRUCTIONS:
    1. PMH SUMMARY: Update the PMH NARRATIVE by merging Today's Visit Note into the Existing PMH Narrative.
       - Keep it under 120-150 words.
       - Use terse, clinical shorthand.
       - STRICTLY AVOID FULL SENTENCES. Use short bullet points.
       - Focus ONLY on major events, surgeries, chronic disease progressions, and severe acute events.
       - Drop minor, resolved issues (e.g., a cold from 5 years ago).
    2. Update the SIGNIFICANT DIAGNOSTICS list. 
       - ADD any abnormal imaging (e.g. ECG, X-Ray, Echo), biopsies, or extreme lab values from today.
       - Remove older, redundant diagnostics to keep the list under 5 items.
       - DO NOT add routine/normal vitals or normal labs.
       - Format as: 'YYYY-MM-DD [Test Name]: [Finding]'
    """

    result = llm_service.complete(response_model=PatientMemory,
        messages=[{"role": "user", "content": prompt}])

    return result.model_dump()