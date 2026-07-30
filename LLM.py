from openai import OpenAI
import instructor

LLM_MODEL = "gemma4:e4b"

# --- OLLAMA CLIENT SETUP ---

class LLMService:

    def __init__(self):
        self.client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            mode = instructor.Mode.JSON
        )

    def complete(self, messages, response_model, temperature:float = 0.1):
        return self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            response_model=response_model,
            temperature=temperature
        )

llm_service = LLMService()
