import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def run_prompt(
    prompt: str,
    model: str = "llama-3.1-8b-instant",  # ✅ important
    retries: int = 3,
    delay: int = 1
):
    """
    Run LLM prompt with retry + latency tracking
    """

    last_error = None

    for attempt in range(retries):
        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model=model,  # ✅ dynamic model
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            latency = round(time.time() - start_time, 3)

            output = response.choices[0].message.content.strip()

            return {
                "output": output,
                "latency": latency,
                "error": None
            }

        except Exception as e:
            last_error = str(e)

            if attempt < retries - 1:
                time.sleep(delay)

    return {
        "output": "",
        "latency": None,
        "error": last_error
    }