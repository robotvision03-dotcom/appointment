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

GREETING_TEXT = "به سامانه نوبت‌دهی خوش آمدید. نام و نام خانوادگی خود را بفرمایید."
ASK_REPEAT = "متوجه نشدم. لطفاً دوباره بفرمایید."
ASK_DOCTOR = "برای کدام پزشک وقت می‌خواهید؟"
ASK_DATE = "چه تاریخی برای ویزیت مد نظر شماست؟ امروز، فردا، یا تاریخ دقیق."
ASK_TIME = "ساعت مورد نظر شما چند است؟"
APOLOGY_DB = "متأسفم، در ثبت نوبت مشکلی پیش آمد. لطفاً دوباره تلاش کنید."
TRANSFER_ACK = "حتماً. الان شما را به منشی وصل می‌کنم. لطفاً روی خط بمانید."


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

        # Try LLM first; fall back to the state machine.
        llm_result = None
        if llm.is_available():
            doctors_blob = _doctors_prompt()
            llm_result = llm.interpret_turn(
                text, state.context, doctors_blob, state.patient_info, state.phase
            )

        if llm_result:
            self._merge_extracted(state, llm_result.get("extracted") or {})
            intent = (llm_result.get("intent") or "continue").lower()
            if intent == "transfer" or wants_transfer(text):
                return self._begin_transfer(state, text)
            reply = (llm_result.get("reply") or "").strip()
            # Still drive booking ourselves so the DB stays consistent.
            machine = self._advance_machine(state, text, llm_reply=reply)
            if machine:
                return machine
            if not reply:
                reply = ASK_REPEAT
            if state.phase == PHASE_GREETING:
                state.phase = PHASE_ASK_NAME
            self.update_context(call_sid, text, reply)
            return self._result(state, reply, intent)

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
        if name:
            info["patient_name"] = str(name).strip()
        doctor_name = extracted.get("doctor_name") or extracted.get("doctor")
        if doctor_name:
            doc = db.find_doctor_by_name(str(doctor_name))
            if doc:
                info["doctor_id"] = doc["id"]
                info["doctor_name"] = doc["name"]
                info["specialty"] = doc["specialty"]
        date_val = extracted.get("date")
        if date_val:
            info["date"] = str(date_val)
        time_val = extracted.get("time")
        if time_val:
            info["time"] = str(time_val)

    def _advance_machine(
        self, state: CallState, text: str, llm_reply: str | None = None
    ) -> dict[str, Any]:
        info = state.patient_info
        phase = state.phase

        if phase in (PHASE_GREETING, PHASE_ASK_NAME) or not info.get("patient_name"):
            if phase == PHASE_GREETING:
                state.phase = PHASE_ASK_NAME
            name = _guess_name(text)
            if name:
                info["patient_name"] = name
                state.phase = PHASE_ASK_DOCTOR
                reply = f"{name} عزیز، {ASK_DOCTOR} پزشکان ما: {_doctor_names()}."
                self.update_context(state.call_sid, text, reply)
                return self._result(state, reply, "continue")
            reply = llm_reply or "لطفاً نام و نام خانوادگی خود را بفرمایید."
            self.update_context(state.call_sid, text, reply)
            return self._result(state, reply, "continue")

        if phase == PHASE_ASK_DOCTOR or not info.get("doctor_id"):
            doc = db.find_doctor_by_name(text)
            if doc:
                info["doctor_id"] = doc["id"]
                info["doctor_name"] = doc["name"]
                info["specialty"] = doc["specialty"]
                state.phase = PHASE_ASK_DATE
                reply = f"نوبت {doc['name']} ({doc['specialty']}). {ASK_DATE}"
                self.update_context(state.call_sid, text, reply)
                return self._result(state, reply, "continue")
            reply = (
                llm_reply
                or f"پزشک مورد نظر پیدا نشد. لطفاً یکی از این اسامی را بفرمایید: {_doctor_names()}."
            )
            self.update_context(state.call_sid, text, reply)
            return self._result(state, reply, "continue")

        if phase == PHASE_ASK_DATE or not info.get("date"):
            parsed = parse_relative_date(text)
            if parsed:
                info["date"] = parsed
                state.phase = PHASE_ASK_TIME
                slots = db.get_available_slots(info["doctor_id"], parsed)
                if not slots:
                    reply = (
                        f"در تاریخ {parsed} نوبت خالی برای {info['doctor_name']} نیست. "
                        "تاریخ دیگری بفرمایید."
                    )
                    info.pop("date", None)
                    state.phase = PHASE_ASK_DATE
                else:
                    shown = "، ".join(slots[:6])
                    reply = f"تاریخ {parsed} ثبت شد. {ASK_TIME} ساعات خالی: {shown}."
                self.update_context(state.call_sid, text, reply)
                return self._result(state, reply, "continue")
            reply = llm_reply or "تاریخ را متوجه نشدم. مثلاً بگویید فردا یا ۱۴۰۳/۰۶/۱۵."
            self.update_context(state.call_sid, text, reply)
            return self._result(state, reply, "continue")

        if phase == PHASE_ASK_TIME or not info.get("time"):
            parsed_time = parse_time(text)
            if parsed_time:
                if not db.is_slot_free(info["doctor_id"], info["date"], parsed_time):
                    slots = db.get_available_slots(info["doctor_id"], info["date"])
                    reply = (
                        f"ساعت {parsed_time} پر است. ساعات خالی: {'، '.join(slots[:8]) or 'هیچ'}."
                    )
                    self.update_context(state.call_sid, text, reply)
                    return self._result(state, reply, "continue")
                info["time"] = parsed_time
                state.phase = PHASE_CONFIRM
                reply = (
                    f"نوبت شما برای {info['doctor_name']} در تاریخ {info['date']} "
                    f"ساعت {info['time']} ثبت می‌شود. آیا تأیید می‌کنید؟"
                )
                self.update_context(state.call_sid, text, reply)
                return self._result(state, reply, "continue")
            reply = llm_reply or "ساعت را متوجه نشدم. مثلاً بگویید ساعت ده صبح یا ۱۴:۳۰."
            self.update_context(state.call_sid, text, reply)
            return self._result(state, reply, "continue")

        if phase == PHASE_CONFIRM:
            if is_yes(text):
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
                    f"نوبت شما با شماره {appt_id} برای {info['doctor_name']} "
                    f"در تاریخ {info['date']} ساعت {info['time']} ثبت شد. "
                    "پیامک تأیید برایتان ارسال می‌شود. سلامت باشید."
                )
                self.update_context(state.call_sid, text, reply)
                return self._result(state, reply, "book")
            if is_no(text):
                state.phase = PHASE_ASK_DATE
                info.pop("date", None)
                info.pop("time", None)
                reply = "اشکال ندارد. از کدام تاریخ شروع کنیم؟"
                self.update_context(state.call_sid, text, reply)
                return self._result(state, reply, "continue")
            reply = llm_reply or "لطفاً با بله یا خیر تأیید کنید."
            self.update_context(state.call_sid, text, reply)
            return self._result(state, reply, "continue")

        if phase in (PHASE_BOOKED, PHASE_DONE):
            reply = "نوبت شما ثبت شده است. اگر کار دیگری ندارید می‌توانید تماس را قطع کنید."
            self.update_context(state.call_sid, text, reply)
            return self._result(state, reply, "continue")

        reply = llm_reply or ASK_REPEAT
        self.update_context(state.call_sid, text, reply)
        return self._result(state, reply, "continue")

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


def _guess_name(text: str) -> str | None:
    t = normalize_persian(text)
    for prefix in ("اسمم", "نام من", "من", "هستم"):
        t = t.replace(prefix, " ")
    t = t.replace("هستم", "").strip()
    parts = [p for p in t.split() if p not in {"است", "هست", "می‌باشد"}]
    if len(parts) >= 2 and all(len(p) >= 2 for p in parts[:3]):
        return " ".join(parts[:4])
    if len(parts) == 1 and len(parts[0]) >= 3:
        return parts[0]
    return None


call_manager = CallManager()
