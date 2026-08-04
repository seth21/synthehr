# SynthEHR: Agentic Clinical Data Generator

**SynthEHR** is a deterministic, SDOH-aware (Social Determinants of Health) synthetic patient data generator. It combines small, local Large Language Models (LLMs) with strict Python state management and Pydantic schemas to generate realistic, longitudinal Electronic Health Record (EHR) timelines without "LLM amnesia," mode collapse, or temporal hallucinations.

---

## Key Features

*   **Latent Drift Engine:** Calculates realistic time gaps between visits based on age, behavioral persona (e.g., `HIGHLY_COMPLIANT`), and active medical conditions.
*   **SDOH & Wildcard Injectors:** Prevents mode collapse by assigning diverse clinical archetypes and injecting random acute events (e.g., fractures, appendicitis) influenced by financial strain.
*   **Point-of-Care Diagnostics:** Automatically intercepts STAT/same-day orders, allowing the LLM to generate physical lab values and clinical reasoning in a single pass without breaking the calendar.
*   **Automated Garbage Collection:** Python sweeps automatically expire missed appointments and auto-stop acute medications (like antibiotics) once their duration passes.
*   **Dynamic Specialist Routing:** Injects scheduled order descriptions into the prompt, forcing the LLM to adopt the correct medical specialty (e.g., Gastroenterology) to execute procedures.
*   **Strict State Deduplication:** Prevents continuous re-diagnosing of active conditions or prescribing therapeutic duplicates.

---

## Architecture

SynthEHR operates on a continuous feedback loop between the LLM and Python:

1.  **Initialization:** Generates baseline demographics, genetics, and SDOH profiles.
2.  **Latent Drift:** Evaluates the pending calendar against the patient's persona to determine compliance, jump time forward, or trigger acute events.
3.  **State Sweepers:** Python automatically removes expired acute medications and marks abandoned orders as missed.
4.  **Clinical Encounter:** The LLM acts as the physician to evaluate the patient, write the clinical note, diagnose (ICD-10), prescribe, and schedule future orders.
5.  **Persistence:** Data is normalized, deduplicated, and saved to a relational SQLite database.

---

## Tech Stack

*   **Python 3.10+**
*   **Pydantic:** Enforces strict JSON schemas and physical constraints.
*   **Instructor:** Patches the LLM client to guarantee structured data extraction.
*   **SQLite:** Local relational database for tracking longitudinal state.
*   **Ollama / OpenAI API:** For running local (e.g., Llama 3, Gemma) or cloud-based LLMs.

---

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install pydantic instructor openai
   ```
2. Ensure your local LLM is running (e.g., via Ollama):

    ```Bash
    ollama run gemma4:e4b
    ```
3. Initialize the database:

    ```Bash
    python db_setup.py
    ```

4. Run the Master Simulation Loop:

    ```Bash
    python main.py
    ```
---

## Database Schema
The SQLite database (synthetic_ehr.db) maintains a highly relational, FHIR-ready structure:

*   patients: Demographics, SDOH, genetic risks.

*   encounters: Timestamps, triggering events, and clinical MDM notes.

*   observations: Vitals, labs, and imaging (separated cleanly by value and unit).

*   conditions: ICD-10 problem list with START/RESOLVE tracking.

*   medications: Active prescriptions with calculated expirations.

*   pending_orders: The forward-looking scheduling calendar.

---

## Roadmap
*[x] Establish deterministic time-travel and SQLite persistence.

*[x] Implement Point-of-Care diagnostic bypass.

*[x] Build Medication and Order Garbage Collectors.

*[ ] HL7 FHIR R4 Export: Map the SQLite database into standardized FHIR JSON bundles.

*[ ] API Integration: Replace LLM string-matching with definitive RxNorm & UMLS terminology lookups.

---

## License Details

[See License Details](LICENSE.md)