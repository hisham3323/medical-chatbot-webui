from __future__ import annotations
import os
import asyncio
from typing import Dict, Any, Optional

# ── ensure OpenAI SDK is installed ────────────────────────────────────────
try:
    from openai import AsyncOpenAI, __version__ as OPENAI_VERSION
except ImportError as e: 
    raise SystemExit(
        "❌  The 'openai' package is required.  Install / upgrade with:\n"
        "   pip install --upgrade openai"
    ) from e

# ── OpenRouter configuration ───────────────────────────────────────────────
BASE_URL   = "https://openrouter.ai/api/v1"
MODEL_NAME = "cognitivecomputations/dolphin3.0-mistral-24b:free"

# ── Global system instructions for full prescribing authority ─────────────
SYSTEM_INSTRUCTIONS = """
You are a board-certified physician with full prescribing authority. You must:
1. Continue asking detailed follow-up questions until you have all necessary clinical information:
   - Symptom onset, duration, severity
   - Past medical history, allergies, current medications
   - Relevant exam or lab findings
2. Once you have a complete picture, provide a prescription recommendation including:
   - Drug name (generic and/or brand)
   - Dosage and route
   - Duration of therapy
   - Key monitoring parameters and potential side effects
Be decisive and confident. Do not include generic disclaimers or defer to other providers.
never answer or respond to non medical qustions at all .
""".strip()

# ── Load Default API key securely from environment ───────────────────────────
# The .env file is loaded by chat_server.py, making the variable available here.
DEFAULT_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not DEFAULT_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in environment variables. Please add it to your .env file.")

# ── Global client using the default key ────────────────────────────────────
_global_client = AsyncOpenAI(api_key=DEFAULT_API_KEY, base_url=BASE_URL)

async def generate(
    prompt: str,
    *,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 512,
    extra: Dict[str, Any] | None = None,
    api_key: Optional[str] = None,
) -> str:
    """Return the assistant’s response text, with a system role for prescribing."""
    client = _global_client
    if api_key:
        client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)

    params: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system",  "content": SYSTEM_INSTRUCTIONS},
            {"role": "user",    "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if extra:
        params.update(extra)

    resp = await client.chat.completions.create(**params)
    msg = resp.choices[0].message
    return (msg.content or "").strip()

if __name__ == "__main__":
    async def _demo() -> None:
        print(f"openai-python version: {OPENAI_VERSION}")
        print(f"Querying model: {MODEL_NAME}\n")
        completion = await _global_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user",   "content": "Name two OTC pain relievers."}
            ],
            temperature=0.2,
            top_p=0.9,
            max_tokens=128,
        )
        msg = completion.choices[0].message
        answer = (msg.content or "").strip()
        print("Assistant says:\n", answer)

    asyncio.run(_demo())