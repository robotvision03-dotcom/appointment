"""Persian LLM client (Ollama) with a structured JSON contract for the receptionist."""

from __future__ import annotations

from typing import Any

import httpx

from src.config import config
from src.utils import extract_json_object, log

SYSTEM_PROMPT = """تو منشی تلفنی یک کلینیک پزشکی هستی. همیشه به فارسی محاوره‌ای، کوتاه و مؤدب پاسخ بده.
وظیفه تو گرفتن نوبت است. اطلاعات لازم: نام بیمار، نام پزشک، تاریخ، ساعت.

پزشکان کلینیک:
{doctors}

قوانین:
- فقط یک سؤال در هر پاسخ بپرس مگر اینکه در حال تأیید نهایی باشی.
- اگر بیمار خواست با منشی انسان صحبت کند، intent را transfer بگذار.
- تاریخ را به صورت YYYY-MM-DD و ساعت را به صورت HH:MM (۲۴ ساعته) برگردان.
- اگر چیزی نامفهوم است، مؤدبانه دوباره بپرس.

خروجی را فقط به صورت JSON با این کلیدها برگردان:
{{
  "reply": "متن گفتاری که باید برای بیمار پخش شود",
  "intent": "continue" یا "book" یا "transfer" یا "cancel",
  "extracted": {{
    "patient_name": null یا رشته,
    "doctor_name": null یا رشته,
    "date": null یا "YYYY-MM-DD",
    "time": null یا "HH:MM",
    "confirmed": null یا true یا false
  }},
  "phase": "greeting" یا "ask_name" یا "ask_doctor" یا "ask_date" یا "ask_time" یا "confirm" یا "booked" یا "transfer"
}}
"""


class PersianLLM:
    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self.url = (url or config.ollama_url).rstrip("/")
        self.model = model or config.ollama_model
        self.timeout = 25.0

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def generate_response(self, prompt: str, context: list[dict[str, str]] | None = None) -> str:
        """Send a prompt to Ollama and return generated text (empty string on failure)."""
        messages: list[dict[str, str]] = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 256},
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
            text = (data.get("message") or {}).get("content") or data.get("response") or ""
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            log.error("Ollama request failed: %s", exc)
            return ""

    def interpret_turn(
        self,
        user_text: str,
        history: list[dict[str, str]],
        doctors_blob: str,
        patient_info: dict[str, Any],
        phase: str,
    ) -> dict[str, Any] | None:
        """Ask the LLM for a structured next action. Returns None on timeout/parse failure."""
        system = SYSTEM_PROMPT.format(doctors=doctors_blob)
        prompt = (
            f"فاز فعلی: {phase}\n"
            f"اطلاعات جمع‌شده تا الان: {patient_info}\n"
            f"آخرین گفته بیمار: {user_text}\n"
            "JSON را برگردان."
        )
        messages = [{"role": "system", "content": system}, *history]
        raw = self.generate_response(prompt, messages)
        if not raw:
            return None
        parsed = extract_json_object(raw)
        if not parsed:
            log.warning("LLM returned non-JSON: %s", raw[:200])
            return {
                "reply": raw,
                "intent": "continue",
                "extracted": {},
                "phase": phase,
            }
        return parsed


llm = PersianLLM()
