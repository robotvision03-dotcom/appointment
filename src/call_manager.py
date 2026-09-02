"""Voice/text dialogue for booking a free car appraisal visit."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from src import db
from src.cars import match_car, parse_km, parse_year
from src.config import config
from src.jalali import format_jalali
from src.sms import extract_iran_mobile
from src.utils import is_yes, log, normalize_persian, parse_relative_date, parse_time

PHASE_ASK_TYPE = "ask_type"
PHASE_ASK_MODEL = "ask_model"
PHASE_ASK_YEAR = "ask_year"
PHASE_ASK_KM = "ask_km"
PHASE_ASK_NAME = "ask_name"
PHASE_ASK_SLOT = "ask_slot"
PHASE_BOOKED = "booked"

GREETING = "سلام وقت بخیر. خودروی شما چه نوع است؟"
ASK_REPEAT = "متوجه نشدم. کوتاه بفرمایید."
FREE_NOTE = "کارشناسی و تعیین قیمت برای فروشنده هیچ هزینه‌ای ندارد."
ADDRESS = config.office_address


@dataclass
class CallState:
    call_sid: str
    stream_sid: str = ""
    from_number: str = ""
    to_number: str = ""
    context: list[dict[str, str]] = field(default_factory=list)
    patient_info: dict[str, Any] = field(default_factory=dict)
    phase: str = PHASE_ASK_TYPE
    audio_buffer: bytearray = field(default_factory=bytearray)
    speaking: bool = False
    last_partial: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    appointment_id: int | None = None
    transfer_requested: bool = False
    last_connect: dict[str, Any] | None = None
    offered_slots: list[dict[str, str]] = field(default_factory=list)

    def transcript_history(self) -> str:
        lines = []
        for msg in self.context:
            role = "فروشنده" if msg["role"] == "user" else "کارشناس"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)


class CallManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_calls: dict[str, CallState] = {}

    def start_call(self, call_sid: str, from_number: str = "", to_number: str = "") -> CallState:
        with self._lock:
            state = CallState(
                call_sid=call_sid,
                from_number=from_number,
                to_number=to_number,
                phase=PHASE_ASK_TYPE,
            )
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
        return GREETING

    def transfer_summary(self, call_sid: str) -> str:
        state = self.get(call_sid)
        if not state:
            return ""
        info = state.patient_info
        return (
            f"{info.get('seller_name', '')} — {info.get('make', '')} {info.get('model', '')} "
            f"مدل {info.get('year', '')}"
        ).strip()

    def sms_body(self, call_sid: str) -> str | None:
        state = self.get(call_sid)
        if not state or not state.appointment_id:
            return None
        info = state.patient_info
        return (
            f"نوبت کارشناسی خودرو ثبت شد. {info.get('make', '')} {info.get('model', '')} "
            f"{info.get('date', '')} ساعت {info.get('time', '')}. "
            f"آدرس: {ADDRESS}. {FREE_NOTE}"
        )

    def handle_user_text(self, call_sid: str, user_text: str) -> dict[str, Any]:
        state = self.get(call_sid)
        if state is None:
            state = self.start_call(call_sid)

        text = normalize_persian(user_text)
        if text.startswith("{") and "ys_log_probs" in text:
            text = ""
        if not text:
            return self._reply(state, user_text, ASK_REPEAT)

        phone = extract_iran_mobile(text)
        if phone:
            state.from_number = phone
            state.patient_info["phone"] = phone

        if state.phase == PHASE_BOOKED:
            return self._reply(
                state,
                text,
                "نوبت شما ثبت شده است. اگر نوبت تازه‌ای می‌خواهید تماس را از نو شروع کنید.",
            )

        if state.phase == PHASE_ASK_TYPE:
            return self._on_type(state, text)
        if state.phase == PHASE_ASK_MODEL:
            return self._on_model(state, text)
        if state.phase == PHASE_ASK_YEAR:
            return self._on_year(state, text)
        if state.phase == PHASE_ASK_KM:
            return self._on_km(state, text)
        if state.phase == PHASE_ASK_NAME:
            return self._on_name(state, text)
        if state.phase == PHASE_ASK_SLOT:
            return self._on_slot(state, text)
        return self._on_type(state, text)

    def _on_type(self, state: CallState, text: str) -> dict[str, Any]:
        car = match_car(text)
        if not car:
            return self._reply(
                state,
                text,
                "نوع خودرو را نگرفتم. مثلاً بگویید پژو پارس، سمند، پراید یا شاهین.",
            )
        info = state.patient_info
        info["make"] = car["make"]
        if car.get("model"):
            info["model"] = car["model"]
            state.phase = PHASE_ASK_YEAR
            return self._reply(
                state,
                text,
                f"{car['make']} {car['model']} ثبت شد. مدل یا سال ساخت آن چند است؟",
            )
        state.phase = PHASE_ASK_MODEL
        return self._reply(
            state,
            text,
            f"برند {car['make']} ثبت شد. مدل خودرو چیست؟",
        )

    def _on_model(self, state: CallState, text: str) -> dict[str, Any]:
        car = match_car(text)
        info = state.patient_info
        if car and car.get("model"):
            info["make"] = car["make"] or info.get("make")
            info["model"] = car["model"]
        else:
            info["model"] = text
        state.phase = PHASE_ASK_YEAR
        return self._reply(
            state,
            text,
            f"{info.get('make', '')} {info.get('model', '')} ثبت شد. سال ساخت یا مدل چند است؟",
        )

    def _on_year(self, state: CallState, text: str) -> dict[str, Any]:
        year = parse_year(text)
        if not year:
            return self._reply(state, text, "سال ساخت را عددی بگویید. مثلاً ۱۳۹۹ یا ۲۰۱۸.")
        state.patient_info["year"] = year
        state.phase = PHASE_ASK_KM
        return self._reply(state, text, "چند کیلومتر کار کرده است؟")

    def _on_km(self, state: CallState, text: str) -> dict[str, Any]:
        km = parse_km(text)
        if km is None:
            return self._reply(state, text, "کارکرد را به کیلومتر بگویید. مثلاً ۸۰ هزار.")
        state.patient_info["km"] = km
        state.phase = PHASE_ASK_NAME
        return self._reply(state, text, "نام و نام خانوادگی شما چیست؟")

    def _on_name(self, state: CallState, text: str) -> dict[str, Any]:
        name = text.replace("اسمم", "").replace("نامم", "").strip()
        if len(name) < 2:
            return self._reply(state, text, "نام خود را بفرمایید.")
        state.patient_info["seller_name"] = name
        state.patient_info["patient_name"] = name
        return self._offer_slots(state, text)

    def _offer_slots(self, state: CallState, text: str) -> dict[str, Any]:
        slots = db.next_open_slots(5)
        state.offered_slots = slots
        state.phase = PHASE_ASK_SLOT
        if not slots:
            return self._reply(
                state,
                text,
                "متأسفانه وقت خالی در تقویم نیست. بعداً دوباره تماس بگیرید.",
            )
        first = slots[0]
        others = "، ".join(s["label"] for s in slots[1:3])
        extra = f" وقت‌های بعدی: {others}." if others else ""
        reply = (
            f"{first['label']} اولین وقت خالی دفتر است. اگر مناسب است بگویید بله. "
            f"{extra} می‌توانید از تقویم هم ساعت نیم‌ساعته انتخاب کنید. "
            f"{FREE_NOTE}"
        )
        return self._reply(state, text, reply, extra={"offered_slots": slots, "calendar": True})

    def _on_slot(self, state: CallState, text: str) -> dict[str, Any]:
        chosen_date = None
        chosen_time = None
        if is_yes(text) and state.offered_slots:
            chosen_date = state.offered_slots[0]["date"]
            chosen_time = state.offered_slots[0]["time"]
        rel = parse_relative_date(text)
        tm = parse_time(text)
        if rel:
            chosen_date = rel
        if tm:
            h, m = map(int, tm.split(":"))
            if m < 15:
                m = 0
            elif m < 45:
                m = 30
            else:
                h += 1
                m = 0
            h = min(max(h, 9), 16)
            chosen_time = f"{h:02d}:{m:02d}"
        if "اول" in text and state.offered_slots:
            chosen_date = state.offered_slots[0]["date"]
            chosen_time = state.offered_slots[0]["time"]
        if chosen_date and not chosen_time and state.offered_slots:
            for s in state.offered_slots:
                if s["date"] == chosen_date:
                    chosen_time = s["time"]
                    break
            if not chosen_time:
                free = db.available_slots(chosen_date)
                chosen_time = free[0] if free else None
        if not chosen_date or not chosen_time:
            return self._reply(
                state,
                text,
                "تاریخ و ساعت را بگویید، مثلاً فردا ساعت ده، یا روی تقویم یک نوبت نیم‌ساعته بزنید.",
                extra={"offered_slots": state.offered_slots, "calendar": True},
            )
        return self._book(state, text, chosen_date, chosen_time)

    def _book(self, state: CallState, text: str, day: str, time: str) -> dict[str, Any]:
        info = state.patient_info
        try:
            appt_id = db.book_inspection(
                seller_name=str(info.get("seller_name") or "فروشنده"),
                phone=str(info.get("phone") or state.from_number or ""),
                make=str(info.get("make") or "نامشخص"),
                model=str(info.get("model") or ""),
                year=str(info.get("year") or ""),
                km=info.get("km"),
                day=day,
                time=time,
            )
        except ValueError as exc:
            state.offered_slots = db.next_open_slots(5)
            return self._reply(
                state,
                text,
                f"{exc} نزدیک‌ترین وقت‌ها: " + "، ".join(s["label"] for s in state.offered_slots[:3]),
                extra={"offered_slots": state.offered_slots, "calendar": True},
            )
        state.appointment_id = appt_id
        state.phase = PHASE_BOOKED
        info["date"] = day
        info["time"] = time
        day_fa = format_jalali(date.fromisoformat(day))
        km = info.get("km")
        km_txt = f"{km:,} کیلومتر".replace(",", "٬") if km is not None else ""
        reply = (
            f"نوبت کارشناسی شماره {appt_id} برای {info.get('seller_name')} ثبت شد. "
            f"{info.get('make')} {info.get('model')} مدل {info.get('year')} {km_txt}. "
            f"لطفاً {day_fa} ساعت {time} به آدرس {ADDRESS} مراجعه کنید "
            f"تا خودروی شما تعیین قیمت و کارشناسی شود. {FREE_NOTE}"
        )
        return self._reply(
            state,
            text,
            reply,
            intent="book",
            extra={"appointment_id": appt_id, "appointment": db.get_inspection(appt_id)},
        )

    def _reply(
        self,
        state: CallState,
        user_text: str,
        reply: str,
        intent: str = "continue",
        extra: dict | None = None,
    ) -> dict[str, Any]:
        self.update_context(state.call_sid, user_text, reply)
        return self._result(state, reply, intent, extra)

    def _result(
        self, state: CallState, reply: str, intent: str, extra: dict | None = None
    ) -> dict[str, Any]:
        payload = {
            "reply": reply,
            "phase": state.phase,
            "intent": intent,
            "patient_info": dict(state.patient_info),
            "appointment_id": state.appointment_id,
            "call_sid": state.call_sid,
            "address": ADDRESS,
            "free_note": FREE_NOTE,
        }
        if extra:
            payload.update(extra)
        return payload


call_manager = CallManager()
