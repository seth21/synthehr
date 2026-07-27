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
        "CARE_AVOIDER",
        "FINANCIALLY_CONSTRAINED"
    ]
    family_history: FamilyHistoryProfile
    sdoh_profile: SDOHProfile

class LatentSubclinicalProgress(BaseModel):
    unobserved_years_passed: float = Field(..., description="Years passed before the next medical interaction", ge=1)
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

class GenericObservation(BaseModel):
    name: str = Field(..., description="e.g., 'Systolic BP', 'HbA1c', 'Heart Rate', 'Chest X-Ray'")
    value: Optional[float] = Field(None, description="Populate ONLY for observations associated with a numeric value e.g., '145' for Systolic BP, '8.2' for HbA1c'")
    string_value: Optional[str] = Field(None, description="Populate ONLY for observations associated with free-form text e.g., 'Mild pulmonary edema' for Chest X-Ray")
    unit: Optional[str] = Field(None, description="e.g., 'mmHg', '%', 'bpm' or leave empty for observations without a unit")
    offset_minutes: int = Field(..., description="Minutes after encounter start when this was measured (e.g., triage for vitals is 5-15 mins, lab results might take 30-90 mins)")

class GenericMedicationChange(BaseModel):
    action: Literal["START", "STOP", "DOSE_CHANGE"]
    rxcui: str = Field(..., description="Numeric RxNorm Concept Unique Identifier (e.g., '8640')")
    medication_name: str = Field(..., description="The active ingredient of the medication e.g. 'Furosemide', 'Lisinopril', 'Omeprazole'")
    dosage: int = Field(..., description="The exact dosage of the active ingredient's prescribed form without the unit e.g. 20, 40, 60")
    unit: str = Field(..., description="The unit the active ingredient is measured in e.g., 'mg' for milligrams, 'g' for grams")
    reason: str
    offset_minutes: int = Field(..., description="Minutes after encounter start when drug was prescribed (e.g., 20-180 mins)")

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

class ClinicalEncounterResult(BaseModel):
    encounter_type: Literal["ROUTINE_OUTPATIENT", "URGENT_CARE", "ER_VISIT", "HOSPITAL_ADMISSION", "SURGERY"]
    reason_for_visit: str
    clinical_note: str = Field(..., description="Comprehensive clinical note written by the attending provider. Usually between 150-300 words.")
    total_encounter_duration_minutes: int = Field(..., description="Total length of the visit in minutes")
    condition_changes: List[ConditionChange] = Field(default_factory=list, description="Diagnosis changes established during this visit")
    medication_changes: List[GenericMedicationChange] = Field(default_factory=list)
    observations: List[GenericObservation] = Field(default_factory=list)
    primary_cause_of_death: Optional[str] = Field(
        None,
        description="Populate ONLY if the event was fatal (e.g., 'Acute Massive Myocardial Infarction')"
    )