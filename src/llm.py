"""Persian LLM client (Ollama) for a formal clinic receptionist."""

from __future__ import annotations

import time
from typing import Any

import httpx

from src.config import config
from src.utils import extract_json_object, log

SYSTEM_PROMPT = """تو منشی رسمی کلینیک هستی. فقط به فارسی معیار و مؤدب صحبت کن (خطاب با «شما»).
هرگز انگلیسی حرف نزن. هرگز نگو هوش مصنوعی یا مدل هستی. هرگز JSON را برای بیمار نخوان.

هدف: گرفتن نوبت. فیلدها: نام کامل، پزشک از فهرست زیر، تاریخ، ساعت.
پزشکان:
{doctors}

سبک:
- کوتاه (حداکثر دو جمله گفتاری).
- در هر نوبت فقط یک سؤال بپرس، مگر وقتی خلاصه نوبت را برای تأیید می‌خوانی.
- اگر چند اطلاعات یک‌جا گفته شد، همان‌ها را ثبت کن و فقط مورد ناقص را بپرس.
- اگر خواستند با انسان صحبت کنند intent=transfer.
- تاریخ: YYYY-MM-DD — ساعت: HH:MM بیست‌وچهارساعته.

خروجی فقط JSON:
{{
  "reply": "جمله‌ای که باید با صدای منشی پخش شود",
  "intent": "continue" | "book" | "transfer" | "cancel",
  "extracted": {{
    "patient_name": null | string,
    "doctor_name": null | string,
    "date": null | "YYYY-MM-DD",
    "time": null | "HH:MM",
    "confirmed": null | true | false
  }},
  "phase": "ask_name" | "ask_doctor" | "ask_date" | "ask_time" | "confirm" | "booked" | "transfer"
}}
"""


class PersianLLM:
    def __init__(self, url: str | None = None, model: str | None = None) -> None:
        self.url = (url or config.ollama_url).rstrip("/")
        self.model = model or config.ollama_model
        self.timeout = 35.0
        self.last_error: str | None = None
        self._avail: bool | None = None
        self._avail_at = 0.0

    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(timeout=timeout or self.timeout, trust_env=False)

    def _candidate_urls(self) -> list[str]:
        urls = [self.url]
        for extra in ("http://127.0.0.1:11434", "http://localhost:11434"):
            if extra not in urls:
                urls.append(extra)
        return urls

    def is_available(self) -> bool:
        now = time.monotonic()
        if self._avail is not None and now - self._avail_at < 20:
            return self._avail
        self.last_error = None
        last_exc = None
        ok = False
        for base in self._candidate_urls():
            try:
                with self._client(timeout=2.5) as client:
                    r = client.get(f"{base}/api/tags")
                if r.status_code == 200:
                    if base != self.url:
                        self.url = base
                    ok = True
                    break
                last_exc = f"HTTP {r.status_code} from {base}"
            except Exception as exc:  # noqa: BLE001
                last_exc = f"{base}: {exc}"
        self._avail = ok
        self._avail_at = now
        self.last_error = None if ok else last_exc
        return ok

    def list_models(self) -> list[str]:
        if not self.is_available():
            return []
        try:
            with self._client(timeout=5.0) as client:
                data = client.get(f"{self.url}/api/tags").json()
            return [m.get("name") for m in data.get("models") or [] if m.get("name")]
        except Exception:  # noqa: BLE001
            return []

    def generate_response(
        self,
        prompt: str,
        context: list[dict[str, str]] | None = None,
        timeout: float | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if context:
            messages.extend(context[-8:])
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 80, "top_p": 0.5},
        }
        try:
            with self._client(timeout=timeout or 12.0) as client:
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
        system = SYSTEM_PROMPT.format(doctors=doctors_blob)
        missing = []
        if not patient_info.get("patient_name"):
            missing.append("نام")
        if not patient_info.get("doctor_id") and not patient_info.get("doctor_name"):
            missing.append("پزشک")
        if not patient_info.get("date"):
            missing.append("تاریخ")
        if not patient_info.get("time"):
            missing.append("ساعت")
        prompt = (
            f"مرحله: {phase}\n"
            f"ثبت‌شده: {patient_info}\n"
            f"هنوز ناقص: {', '.join(missing) or 'هیچ (فقط تأیید)'}\n"
            f"گفته بیمار: {user_text}\n"
            "JSON."
        )
        messages = [{"role": "system", "content": system}, *history]
        raw = self.generate_response(prompt, messages)
        if not raw:
            return None
        parsed = extract_json_object(raw)
        if not parsed:
            log.warning("LLM returned non-JSON: %s", raw[:200])
            return None
        return parsed


llm = PersianLLM()
