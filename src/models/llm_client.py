from dataclasses import dataclass
from typing import Any, Dict, List
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_PATH = os.path.join(ROOT_DIR, ".env")

load_dotenv(ENV_PATH)



@dataclass
class LLMConfig:
    model: str
    max_completion_tokens: int = 512
    temperature: float | None = None  # Only used when model allows it
    seed: int | None = None




def call_llm(
    system_prompt: str,
    user_prompt: str,
    cfg: LLMConfig,
    json_mode: bool = False
) -> str:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build request payload
    request: Dict[str, Any] = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_completion_tokens": cfg.max_completion_tokens,
    }

    # Temperature allowed only for certain models
    if cfg.temperature is not None:
        request["temperature"] = cfg.temperature

    # Seeds supported on gpt-5-series
    if cfg.seed is not None:
        request["seed"] = cfg.seed

    # JSON mode enabled if requested
    if json_mode:
        request["response_format"] = {"type": "json_object"}

    # Send request
    response = client.chat.completions.create(**request)

    # Return the text output
    return response.choices[0].message.content


#Json parse
def parse_json_or_throw(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        # Fallback: extract JSON substring
        import re
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        raise ValueError(f"LLM did not return valid JSON.\nOutput:\n{text}")
