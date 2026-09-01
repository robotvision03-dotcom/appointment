"""Per-call conversation state and the booking dialogue state machine.

The LLM is used when Ollama is reachable. If it times out or is missing, a
deterministic Persian flow still collects name → doctor → date → time → confirm.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src import db
from src.config import config
from src.llm import llm
from src.utils import (
    is_no,
    is_yes,
    log,
    normalize_persian,
    parse_relative_date,
    parse_time,
    wants_transfer,
)

PHASE_GREETING = "greeting"
PHASE_ASK_NAME = "ask_name"
PHASE_ASK_DOCTOR = "ask_doctor"
PHASE_ASK_DATE = "ask_date"
PHASE_ASK_TIME = "ask_time"
PHASE_CONFIRM = "confirm"
PHASE_BOOKED = "booked"
PHASE_TRANSFER = "transfer"
PHASE_DONE = "done"

GREETING_TEXT = "سلام. نام کامل‌تان چیست؟"
ASK_REPEAT = "متوجه نشدم. کوتاه بفرمایید."
ASK_DOCTOR = "نام پزشک؟"
ASK_DATE = "چه روزی؟ امروز، فردا، یا تاریخ."
ASK_TIME = "چه ساعتی؟"
APOLOGY_DB = "از اختلال پیش‌آمده پوزش می‌خواهیم. لطفاً دوباره برای ثبت نوبت تلاش کنید."
TRANSFER_ACK = (
    "چشم، شما را به منشی انسانی کلینیک وصل می‌کنم. لطفاً روی خط بمانید."
)


@dataclass
class CallState:
    call_sid: str
    stream_sid: str = ""
    from_number: str = ""
    to_number: str = ""
    context: list[dict[str, str]] = field(default_factory=list)
    patient_info: dict[str, Any] = field(default_factory=dict)
    phase: str = PHASE_GREETING
    audio_buffer: bytearray = field(default_factory=bytearray)
    speaking: bool = False
    last_partial: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    appointment_id: int | None = None
    transfer_requested: bool = False

    def transcript_history(self) -> str:
        lines = []
        for msg in self.context:
            role = "بیمار" if msg["role"] == "user" else "منشی"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)


class CallManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_calls: dict[str, CallState] = {}

    def start_call(self, call_sid: str, from_number: str = "", to_number: str = "") -> CallState:
        with self._lock:
            state = CallState(call_sid=call_sid, from_number=from_number, to_number=to_number)
            self.active_calls[call_sid] = state
            log.info("Call started %s from %s", call_sid, from_number)
            return state

    def get(self, call_sid: str) -> CallState | None:
        with self._lock:
            return self.active_calls.get(call_sid)

    def end_call(self, call_sid: str) -> None:
        with self._lock:
            self.active_calls.pop(call_sid, None)
            log.info("Call ended %s", call_sid)

    def get_context(self, call_sid: str) -> list[dict[str, str]]:
        state = self.get(call_sid)
        return list(state.context) if state else []

    def update_context(self, call_sid: str, user_message: str, assistant_reply: str) -> None:
        state = self.get(call_sid)
        if not state:
            return
        if user_message:
            state.context.append({"role": "user", "content": user_message})
        if assistant_reply:
            state.context.append({"role": "assistant", "content": assistant_reply})

    def greeting(self) -> str:
        return GREETING_TEXT

    def handle_user_text(self, call_sid: str, user_text: str) -> dict[str, Any]:
        """
        Process one patient utterance.

        Returns dict with keys: reply, phase, intent, patient_info, appointment_id.
        """
        state = self.get(call_sid)
        if state is None:
            state = self.start_call(call_sid)

        text = normalize_persian(user_text)
        if not text:
            reply = ASK_REPEAT
            self.update_context(call_sid, user_text, reply)
            return self._result(state, reply, "continue")

        if wants_transfer(text) and state.phase not in (PHASE_BOOKED, PHASE_DONE):
            return self._begin_transfer(state, text)

        llm_result = None
        if config.ollama_enabled and _needs_llm(state, text) and llm.is_available():
            llm_result = llm.interpret_turn(
                text,
                state.context,
                _doctors_prompt(),
                state.patient_info,
                state.phase,
            )
        if llm_result:
            self._merge_extracted(state, llm_result.get("extracted") or {})
            intent = (llm_result.get("intent") or "continue").lower()
            if intent == "transfer":
                return self._begin_transfer(state, text)
            spoken = _spoken_reply(llm_result.get("reply"))
            machine = self._advance_machine(state, text, llm_reply=spoken, prefer_llm=True)
            return machine

        return self._advance_machine(state, text)

    def _begin_transfer(self, state: CallState, user_text: str) -> dict[str, Any]:
        state.phase = PHASE_TRANSFER
        state.transfer_requested = True
        self.update_context(state.call_sid, user_text, TRANSFER_ACK)
        log.info("Warm transfer requested for %s", state.call_sid)
        return self._result(state, TRANSFER_ACK, "transfer")

    def _merge_extracted(self, state: CallState, extracted: dict[str, Any]) -> None:
        info = state.patient_info
        name = extracted.get("patient_name")
        if name and not info.get("patient_name"):
            cleaned = str(name).strip()
            if cleaned and "دکتر" not in cleaned:
                info["patient_name"] = cleaned
        doctor_name = extracted.get("doctor_name") or extracted.get("doctor")
        if doctor_name and not info.get("doctor_id"):
            doc = db.find_doctor_by_name(str(doctor_name), strict=False)
            if doc:
                info["doctor_id"] = doc["id"]
                info["doctor_name"] = doc["name"]
                info["specialty"] = doc["specialty"]
        date_val = extracted.get("date")
        if date_val and not info.get("date"):
            info["date"] = str(date_val)
        time_val = extracted.get("time")
        if time_val and not info.get("time"):
            info["time"] = str(time_val)

    def _ingest(self, state: CallState, text: str) -> None:
        """Fill every field this utterance contains. Never overwrite a filled slot."""
        info = state.patient_info
        if not info.get("doctor_id"):
            doc = db.find_doctor_by_name(text, strict=not bool(info.get("patient_name")))
            if doc:
                info["doctor_id"] = doc["id"]
                info["doctor_name"] = doc["name"]
                info["specialty"] = doc["specialty"]
        if not info.get("patient_name"):
            name = _guess_name(text)
            if name:
                info["patient_name"] = name
        if not info.get("date"):
            parsed = parse_relative_date(text)
            if parsed:
                info["date"] = parsed
        if not info.get("time"):
            parsed_time = parse_time(text)
            if parsed_time:
                info["time"] = parsed_time

    def _advance_machine(
        self,
        state: CallState,
        text: str,
        llm_reply: str | None = None,
        prefer_llm: bool = False,
    ) -> dict[str, Any]:
        info = state.patient_info
        self._ingest(state, text)

        if state.phase in (PHASE_BOOKED, PHASE_DONE):
            reply = "نوبت شما ثبت شده است. اگر موضوع دیگری نیست، می‌توانید گفتگو را پایان دهید."
            self.update_context(state.call_sid, text, reply)
            return self._result(state, reply, "continue")

        missing = _missing_slot(info)
        if missing is None:
            if is_yes(text):
                return self._book_or_fail(state, text)
            if is_no(text):
                info.pop("date", None)
                info.pop("time", None)
                state.phase = PHASE_ASK_DATE
                reply = "بسیار خب. تاریخ دیگری را بفرمایید."
                self.update_context(state.call_sid, text, reply)
                return self._result(state, reply, "continue")
            if info.get("doctor_id") and info.get("date") and info.get("time"):
                if not db.is_slot_free(int(info["doctor_id"]), info["date"], info["time"]):
                    slots = db.get_available_slots(int(info["doctor_id"]), info["date"])
                    info.pop("time", None)
                    state.phase = PHASE_ASK_TIME
                    reply = (
                        f"آن ساعت در دسترس نیست. ساعات آزاد: "
                        f"{'، '.join(slots[:8]) or 'ظرفیتی در این روز نمانده'}."
                    )
                    self.update_context(state.call_sid, text, reply)
                    return self._result(state, reply, "continue")
            if state.phase == PHASE_CONFIRM:
                reply = "برای ثبت نهایی لطفاً «بله» یا «خیر» بفرمایید."
                self.update_context(state.call_sid, text, reply)
                return self._result(state, reply, "continue")
            state.phase = PHASE_CONFIRM
            reply = (
                f"ثبت شود؟ {info.get('patient_name')}، {info.get('doctor_name')}، "
                f"{info.get('date')} {info.get('time')}. بله یا خیر."
            )
            self.update_context(state.call_sid, text, reply)
            return self._result(state, reply, "continue")

        if missing == "date" and info.get("doctor_id") and info.get("date"):
            slots = db.get_available_slots(int(info["doctor_id"]), info["date"])
            if not slots:
                info.pop("date", None)
                missing = "date"

        state.phase = {
            "name": PHASE_ASK_NAME,
            "doctor": PHASE_ASK_DOCTOR,
            "date": PHASE_ASK_DATE,
            "time": PHASE_ASK_TIME,
        }[missing]
        canned = _question_for(missing, info)
        # Do not reuse an LLM line that would re-ask a filled field.
        reply = canned
        self.update_context(state.call_sid, text, reply)
        return self._result(state, reply, "continue")

    def _book_or_fail(self, state: CallState, text: str) -> dict[str, Any]:
        info = state.patient_info
        try:
            appt_id = db.book_appointment(
                patient_name=info.get("patient_name", "ناشناس"),
                phone=state.from_number or info.get("phone", ""),
                doctor_id=int(info["doctor_id"]),
                date=info["date"],
                time=info["time"],
            )
        except ValueError as exc:
            state.phase = PHASE_ASK_TIME
            info.pop("time", None)
            reply = f"{exc} لطفاً ساعت دیگری انتخاب کنید."
            self.update_context(state.call_sid, text, reply)
            return self._result(state, reply, "continue")
        except Exception as exc:  # noqa: BLE001
            log.error("DB book failed: %s", exc)
            reply = APOLOGY_DB
            self.update_context(state.call_sid, text, reply)
            return self._result(state, reply, "continue")
        state.appointment_id = appt_id
        state.phase = PHASE_BOOKED
        reply = (
            f"نوبت شما با شماره پیگیری {appt_id} برای {info['doctor_name']} "
            f"در تاریخ {info['date']} ساعت {info['time']} ثبت شد. "
            "موفق و سلامت باشید."
        )
        self.update_context(state.call_sid, text, reply)
        return self._result(state, reply, "book")

    def transfer_summary(self, call_sid: str) -> str:
        state = self.get(call_sid)
        if not state:
            return "تماس از یک بیمار برای صحبت با منشی."
        info = state.patient_info
        name = info.get("patient_name") or "نامشخص"
        doctor = info.get("doctor_name") or "نامشخص"
        snippet = state.context[-1]["content"] if state.context else ""
        return f"تماس از بیمار {name} برای دکتر {doctor} در مورد مشکل {snippet[:80] or 'نوبت'}."

    def sms_body(self, call_sid: str) -> str | None:
        state = self.get(call_sid)
        if not state or not state.appointment_id:
            return None
        info = state.patient_info
        return (
            f"نوبت شما ثبت شد. کد {state.appointment_id} — "
            f"{info.get('doctor_name')} — {info.get('date')} ساعت {info.get('time')}."
        )

    def _result(self, state: CallState, reply: str, intent: str) -> dict[str, Any]:
        return {
            "reply": reply,
            "phase": state.phase,
            "intent": intent,
            "patient_info": dict(state.patient_info),
            "appointment_id": state.appointment_id,
            "call_sid": state.call_sid,
        }


def _doctor_names() -> str:
    return "، ".join(d["name"] for d in db.list_doctors())


def _doctors_prompt() -> str:
    lines = [f"- {d['name']} ({d['specialty']})" for d in db.list_doctors()]
    return "\n".join(lines) or "لیست پزشک خالی است."


_NAME_STOP = {
    "امروز",
    "فردا",
    "پس‌فردا",
    "پسفردا",
    "ساعت",
    "بله",
    "خیر",
    "صبح",
    "عصر",
    "شب",
    "دکتر",
    "نوبت",
    "ویزیت",
    "می‌خواهم",
    "میخوام",
}


def _guess_name(text: str) -> str | None:
    t = normalize_persian(text)
    if "دکتر" in t:
        return None
    if parse_relative_date(t) and len(t.split()) <= 2:
        return None
    if parse_time(t) and len(t.split()) <= 4:
        return None
    for prefix in ("اسمم", "نام من", "نامم", "هستم"):
        t = t.replace(prefix, " ")
    t = t.replace("هستم", "").strip()
    parts = [p for p in t.split() if p not in _NAME_STOP and p not in {"است", "هست", "من"}]
    if len(parts) >= 2 and all(len(p) >= 2 for p in parts[:3]):
        return " ".join(parts[:4])
    if len(parts) == 1 and len(parts[0]) >= 3:
        return parts[0]
    return None


def _missing_slot(info: dict) -> str | None:
    if not info.get("patient_name"):
        return "name"
    if not info.get("doctor_id"):
        return "doctor"
    if not info.get("date"):
        return "date"
    if not info.get("time"):
        return "time"
    return None


def _question_for(missing: str, info: dict) -> str:
    if missing == "name":
        return "نام کامل‌تان؟"
    if missing == "doctor":
        return ASK_DOCTOR
    if missing == "date":
        return ASK_DATE
    if missing == "time":
        slots = db.get_available_slots(int(info["doctor_id"]), info["date"])
        shown = "، ".join(slots[:6]) or "—"
        return f"ساعت؟ آزاد: {shown}"
    return ASK_REPEAT


def _needs_llm(state: CallState, text: str) -> bool:
    """Skip Ollama on clear checklist answers so 7B/14B models do not stall every turn."""
    if wants_transfer(text):
        return False
    if is_yes(text) or is_no(text):
        return False
    phase = state.phase
    info = state.patient_info
    if (phase in (PHASE_GREETING, PHASE_ASK_NAME) or not info.get("patient_name")) and _guess_name(text):
        return False
    if (phase == PHASE_ASK_DOCTOR or not info.get("doctor_id")) and db.find_doctor_by_name(
        text, strict=not bool(info.get("patient_name"))
    ):
        return False
    if (phase == PHASE_ASK_DATE or not info.get("date")) and parse_relative_date(text):
        return False
    if (phase == PHASE_ASK_TIME or not info.get("time")) and parse_time(text):
        return False
    if phase in (PHASE_BOOKED, PHASE_DONE, PHASE_CONFIRM):
        return False
    return len(text.split()) >= 4


def _spoken_reply(raw: str | None) -> str | None:
    if not raw:
        return None
    text = str(raw).strip().strip('"').strip("'")
    if "{" in text or "}" in text:
        return None
    persian = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    if persian < 10:
        return None
    if len(text) > 280:
        text = text[:277] + "…"
    return text


def _pick_reply(canned: str, llm_reply: str | None, prefer_llm: bool) -> str:
    if prefer_llm and llm_reply:
        return llm_reply
    return canned


call_manager = CallManager()
