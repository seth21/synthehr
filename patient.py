import uuid
from datetime import datetime
import random

from LLM import LLMService
from core.classes import PatientBaseState
from db import queries
from db.database import Database


def generate_patient_base(llm: LLMService, birth_year: int, gender: str, ethnicity: str) -> PatientBaseState:
    """Generates the starting genetic and social state of the patient."""
    # Force clinical diversity by injecting a random archetype
    archetypes = [
        "Autoimmune (e.g., Rheumatoid Arthritis, Lupus, Psoriasis)",
        "Neurological (e.g., Migraines, Early-onset Parkinson's, Epilepsy)",
        "Musculoskeletal/Orthopedic (e.g., Osteoarthritis, Chronic back pain, Gout)",
        "Pulmonary (e.g., Asthma, COPD, Interstitial lung disease)",
        "Gastrointestinal (e.g., Crohn's, Ulcerative Colitis, Severe GERD)",
        "Psychiatric (e.g., Bipolar disorder, Major Depressive Disorder, Severe Anxiety)",
        "Cardio-Metabolic (e.g., Diabetes, Hypertension, Heart Failure)",
        "Generally Healthy"
    ]
    assigned_archetype = random.choices(archetypes, weights=[5, 10, 20, 20, 15, 10, 30, 20], k=1)[0]
    compliance_types = [
        "Highly Compliant", "Needs Reminders", "Care Avoider"
    ]
    assigned_compliance = random.choices(compliance_types, weights=[60, 30, 10], k=1)[0]

    prompt = f"""
    You are a clinical data generator. Generate a realistic medical backstory for a patient.
    
    Demographics: {gender}, {ethnicity}, born {birth_year}
    
    CLINICAL DESTINY ARCHETYPE: {assigned_archetype}
    COMPLIANCE ARCHETYPE: {assigned_compliance}

    INSTRUCTIONS:
    1. Generate the patient's genetic risks and family history strictly focused on the CLINICAL DESTINY ARCHETYPE provided above. 
    2. Give them a realistic family history with at least one notable genetic risk factor.
    3. Do NOT default to common diseases like Diabetes or Hypertension unless it is the assigned archetype. 
    4. Generate a realistic behavioral and SDOH profile and make the profile cohesive.
    """

    print(f"Generating base state for {gender} born in {birth_year}...")

    patient_state = llm.complete(response_model=PatientBaseState,
        messages=[
            {"role": "system", "content": "You output strict, valid JSON matching the schema."},
            {"role": "user", "content": prompt}
        ],
        temperature = 0.7)

    return patient_state

def initialize_patient(llm: LLMService, db: Database, birth_year: int = 1970,
        gender: str = "Male",
        ethnicity: str = "African American") -> str:
    patient_id = str(uuid.uuid4())
    # Generate seed, genetics, and SDOH baseline
    # Initialize the Macro Clock (Random DOB in the given year)
    dob = datetime(birth_year, random.randint(1, 12), random.randint(1, 28))
    base_state = generate_patient_base(llm, birth_year, gender, ethnicity)
    with db.transaction():
        queries.save_patient_to_db(db, patient_id, dob, base_state)
        patient = queries.get_patient_profile(db, patient_id)
    print(f"Full Name: {patient['first_name']} {patient['last_name']}")
    print(f"Demographics: {patient['demographics']}")
    print(f"Behavior Persona: {patient['behavior']}")
    print(f"Genetic Risks: {patient['genetics'].get('genetic_risk_factors')}")
    print(f"Financial Strain: {patient['sdoh'].get('financial_strain')}")
    print("--------------------------------------------------\n")
    return patient_id
