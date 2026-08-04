from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional
import re
# --- PYDANTIC SCHEMAS ---

class FamilyMemberCondition(BaseModel):
    relationship: str = Field(..., description="e.g., 'Father', 'Maternal Grandmother'")
    condition: str = Field(..., description="e.g., 'Type 2 Diabetes', 'Breast Cancer'")
    age_of_onset: Optional[int]
    caused_death: bool
    age_at_death: Optional[int]

class FamilyHistoryProfile(BaseModel):
    known_relatives: List[FamilyMemberCondition]
    genetic_risk_factors: List[str] = Field(..., description="e.g., 'High Risk for Early CAD', 'BRCA-1 Suspected'")

class SDOHProfile(BaseModel):
    housing_status: Literal["STABLE", "HOUSING_INSECURE", "UNHOUSED"]
    food_security: Literal["SECURE", "FOOD_INSECURE", "FOOD_DESERT"]
    financial_strain: Literal["COMFORTABLE", "LIVING_PAYCHECK_TO_PAYCHECK", "SEVERE_POVERTY"]
    transportation_access: bool
    occupational_hazard: Literal["NONE", "HIGH_STRESS_SEDENTARY", "MANUAL_LABOR_HAZARDOUS"]
    current_life_stressors: List[str]

class PatientBaseState(BaseModel):
    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    demographics: str = Field(..., description="e.g., 'Male, Caucasian, born 1980'")
    behavioral_profile: Literal[
        "HIGHLY_COMPLIANT",
        "NEEDS_REMINDERS",
        "CARE_AVOIDER"
    ]
    family_history: FamilyHistoryProfile
    sdoh_profile: SDOHProfile

class LatentSubclinicalProgress(BaseModel):
    days_passed: int = Field(..., description="Days passed before the next medical interaction", ge=1)
    #patient_age_at_end_of_gap: float = Field(..., description="Patient's age at the end of this gap")
    silent_pathology_accumulated: List[str] = Field(
        ...,
        description="e.g., 'Worsening insulin resistance', 'Microvascular damage', 'Undetected tumor growth'"
    )
    triggering_event: str = Field(
        ...,
        description="What brings them to the doctor? e.g., 'Routine Checkup', 'Acute Myocardial Infarction', 'Job-required physical'"
    )
    survival_status: Literal["SURVIVED", "FATAL_EVENT"]
    # The Event Router
    event_type: Literal["FULFILLED_ORDER", "MISSED_ORDERS_AND_DELAYED_VISIT", "UNSCHEDULED_ACUTE_EVENT", "ROUTINE_GAP"]
    fulfilled_order_id: Optional[str] = Field(None,
                                              description="If FULFILLED_ORDER, provide the specific order_id they attended")
    missed_order_ids: List[str] = Field(default_factory=list,
                                        description="Any order_ids the patient skipped or no-showed")

class GenericObservation(BaseModel):
    name: str = Field(..., description="Name of the test or vital sign (e.g., 'Systolic BP', 'HbA1c', 'Chest X-Ray')")
    value: str = Field(..., description="The actual result. Can be a number ('145', '8.2'), a ratio ('120/80'), or text ('Mild cardiomegaly', 'Negative'). DO NOT put units here.")
    unit: Optional[str] = Field(None, description="The unit of measurement (e.g., 'mmHg', '%', 'mg/dL'). Leave null for text/imaging results.")
    offset_minutes: int = Field(..., description="Minutes after encounter start when this was measured (e.g., triage for vitals is 5-15 mins, lab results might take 30-90 mins)")

class GenericMedicationChange(BaseModel):
    action: Literal["START", "STOP", "DOSE_CHANGE"]
    rxcui: str = Field(..., description="Numeric RxNorm Concept Unique Identifier (e.g., '8640')")
    medication_name: str = Field(..., description="The active ingredient of the medication e.g. 'Furosemide', 'Lisinopril', 'Omeprazole'")
    dosage: int = Field(..., description="The exact dosage of the active ingredient's prescribed form without the unit e.g. 20, 40, 60")
    unit: str = Field(..., description="The unit the active ingredient is measured in e.g., 'mg' for milligrams, 'g' for grams")
    reason: str
    offset_minutes: int = Field(..., description="Minutes after encounter start when drug was prescribed (e.g., 20-180 mins)")
    duration_days: Optional[int] = Field(None, description="If this is a short-term acute drug (e.g., antibiotics for 7 days), put 7. If chronic, leave null.")

    @field_validator('rxcui')
    @classmethod
    def validate_rxcui(cls, v: str) -> str:
        """Forces the LLM to output a purely numeric RxCUI."""
        v = v.strip()
        if not v.isdigit():
            raise ValueError(
                f"Invalid RxCUI format: '{v}'. "
                "RxNorm codes must be purely numeric strings (e.g., '314076')."
            )
        return v

class ConditionChange(BaseModel):
    action: Literal["START", "RESOLVE"]
    icd10_code: str = Field(..., description="Valid ICD-10 code (e.g., 'E11.9', 'J01.90')")
    condition_name: str = Field(..., description="Standardized ICD-10 diagnosis name")
    offset_minutes: int = Field(..., description="Minutes after encounter start when doctor made diagnosis (e.g., 15-120 mins)")

    @field_validator('icd10_code')
    @classmethod
    def validate_icd10_format(cls, v: str) -> str:
        """
        Forces the LLM to output a valid ICD-10 structure:
        One Letter, Two Digits, optional period, up to 4 alphanumeric extensions.
        """
        # Strip any accidental whitespace the LLM might have added
        v = v.strip().upper()

        # ICD-10 Regex Pattern
        pattern = r'^[A-Z]\d{2}(?:\.[A-Z0-9]{1,4})?$'

        if not re.match(pattern, v):
            raise ValueError(
                f"Invalid ICD-10 code format: '{v}'. "
                "Must be a letter followed by two digits, optionally followed by a decimal and up to 4 characters (e.g., 'I10', 'E11.9')."
            )
        return v

class PendingOrder(BaseModel):
    order_type: Literal["IMAGING", "LAB_DRAW", "SPECIALIST_REFERRAL", "PROCEDURE", "ROUTINE_FOLLOW_UP"]
    description: str = Field(..., description="e.g., 'Echocardiogram', 'Cardiology Consult'")
    target_days_from_now: int = Field(..., gt=1, description="MUST BE >= 1. Orders are for FUTURE days only. Same-day tests must go into observations.")
    urgency: Literal["ROUTINE", "URGENT", "STAT"]

    # 2. Add a validator to feed a specific correction back to the LLM if it fails
    @field_validator('target_days_from_now')
    @classmethod
    def validate_future_days(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(
                f"Invalid target_days_from_now: {v}. "
                "Orders CANNOT be for 0 days. If the lab or imaging was done today, "
                "remove it from orders_placed and put the result in the 'observations' list instead."
            )
        return v

class ClinicalEncounterResult(BaseModel):
    encounter_type: Literal["ROUTINE_OUTPATIENT", "URGENT_CARE", "ER_VISIT", "HOSPITAL_ADMISSION", "SURGERY"]
    reason_for_visit: str
    clinical_note: str = Field(..., description="Comprehensive clinical note written by the attending provider. Usually between 150-300 words.")
    total_encounter_duration_minutes: int = Field(..., description="Total length of the visit in minutes")
    condition_changes: List[ConditionChange] = Field(default_factory=list, description="Diagnosis changes established during this visit")
    medication_changes: List[GenericMedicationChange] = Field(default_factory=list)
    observations: List[GenericObservation] = Field(default_factory=list)
    orders_placed: List[PendingOrder] = Field(default_factory=list)
    primary_cause_of_death: Optional[str] = Field(
        None,
        description="Populate ONLY if the event was fatal (e.g., 'Acute Massive Myocardial Infarction')"
    )

class PatientMemory(BaseModel):
    pmh_summary: str = Field(
        ...,
        description="A compressed narrative of the patient's Past Medical/Surgical History (max 150 words)."
    )
    significant_diagnostics: List[str] = Field(
        default_factory=list,
        description="A running list of major abnormal labs, imaging, or pathology with their dates with format 'YYYY-MM-DD Exam: Finding' e.g., '2021-04-23 ECG: Atrial Fibrillation', '2019-03-15 Colonoscopy: Benign polyps'"
    )

