"""FastAPI entrypoint: Twilio webhooks, Media Streams WebSocket, demo UI, health."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src import db
from src.call_manager import call_manager
from src.config import ROOT_DIR, config
from src.llm import llm
from src.stt import stt
from src.tts import tts
from src.handoff import start_warm_transfer
from src.live_voice import handle_browser_voice
from src.sip_bridge import sip_turn
from src.sms import send_booking_sms
from src.twilio_handler import (
    handle_incoming_call,
    handle_media_stream,
    receptionist_join_twiml,
    twiml_response,
)
from src.utils import log

app = FastAPI(
    title="Persian AI Voice Agent",
    description="منشی صوتی فارسی برای نوبت‌دهی پزشکی",
    version="1.0.0",
)

STATIC_DIR = ROOT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    log.info(
        "Agent ready host=%s port=%s stt=%s tts=%s ollama=%s kavenegar=%s",
        config.host,
        config.port,
        stt.available,
        tts.available,
        llm.is_available(),
        config.kavenegar_configured,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "stt": {"available": stt.available, "path": str(config.vosk_model_path)},
        "tts": {"available": tts.available, "path": str(config.piper_model_path)},
        "llm": {
            "available": llm.is_available(),
            "url": llm.url,
            "model": llm.model,
            "error": llm.last_error,
            "installed": llm.list_models() if llm.is_available() else [],
        },
        "sms": {"provider": "kavenegar", "available": config.kavenegar_configured},
        "voice": {"browser": True, "stt": stt.available, "tts": tts.available},
        "receptionist_number": config.receptionist_number,
        "doctors": len(db.list_doctors()),
    }


@app.post("/voice/incoming")
async def voice_incoming(request: Request) -> Response:
    return await handle_incoming_call(request)


@app.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket) -> None:
    await handle_media_stream(websocket)


@app.websocket("/voice/live")
async def voice_live(websocket: WebSocket) -> None:
    """Microphone session from the clinic web app (usable in Iran)."""
    await handle_browser_voice(websocket)


@app.post("/voice/transfer-status")
async def transfer_status(request: Request) -> JSONResponse:
    form = await request.form()
    log.info("Transfer status CallSid=%s Status=%s", form.get("CallSid"), form.get("CallStatus"))
    return JSONResponse({"ok": True})


@app.post("/voice/receptionist-join")
@app.get("/voice/receptionist-join")
async def receptionist_join(request: Request) -> Response:
    conference = request.query_params.get("conference") or "clinic"
    play = request.query_params.get("play") or ""
    return twiml_response(receptionist_join_twiml(conference, play))


@app.get("/api/doctors")
def api_doctors() -> list[dict]:
    return db.list_doctors()


@app.get("/api/slots")
def api_slots(doctor_id: int, date: str) -> dict:
    return {"slots": db.get_available_slots(doctor_id, date)}


@app.post("/api/book")
async def api_book(request: Request) -> JSONResponse:
    """Patient-managed booking: no conversation required."""
    body = await request.json()
    name = str(body.get("patient_name") or body.get("name") or "").strip()
    phone = str(body.get("phone") or "")
    try:
        doctor_id = int(body.get("doctor_id"))
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "error": "پزشک را انتخاب کنید."}, status_code=400)
    date = str(body.get("date") or "").strip()
    time = str(body.get("time") or "").strip()
    if not name or not date or not time:
        return JSONResponse({"ok": False, "error": "نام، تاریخ و ساعت لازم است."}, status_code=400)
    if not db.get_doctor(doctor_id):
        return JSONResponse({"ok": False, "error": "پزشک نامعتبر است."}, status_code=400)
    try:
        appt_id = db.book_appointment(name, phone, doctor_id, date, time)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    appt = db.get_appointment(appt_id)
    if phone:
        from src.call_manager import call_manager

        sid = f"form-{appt_id}"
        call_manager.end_call(sid)
        st = call_manager.start_call(sid, from_number=phone)
        st.appointment_id = appt_id
        st.patient_info = {
            "patient_name": name,
            "doctor_name": (appt or {}).get("doctor_name"),
            "date": date,
            "time": time,
        }
        send_booking_sms(sid)
        call_manager.end_call(sid)
    return JSONResponse({"ok": True, "appointment": appt})


@app.get("/api/appointments")
def api_appointments() -> list[dict]:
    return db.list_appointments()


@app.post("/api/simulate")
async def api_simulate(request: Request) -> JSONResponse:
    """Text-in / text-out conversation for local testing without Twilio."""
    body = await request.json()
    session_id = str(body.get("session_id") or "demo-local")
    text = str(body.get("text") or "")
    if call_manager.get(session_id) is None:
        call_manager.start_call(
            session_id, from_number=str(body.get("phone") or "")
        )
        if not text:
            return JSONResponse(
                {
                    "reply": call_manager.greeting(),
                    "phase": "ask_name",
                    "intent": "continue",
                    "patient_info": {},
                    "appointment_id": None,
                    "call_sid": session_id,
                }
            )
    phone = str(body.get("phone") or "")
    if phone:
        state = call_manager.get(session_id)
        if state:
            state.from_number = phone
    result = call_manager.handle_user_text(session_id, text)
    if result.get("intent") == "book":
        result["sms"] = send_booking_sms(session_id)
    if result.get("intent") == "transfer":
        result["transfer"] = start_warm_transfer(session_id)
        extra = (result["transfer"] or {}).get("reply_extra")
        if extra:
            result["reply"] = f"{result.get('reply', '')} {extra}".strip()
    return JSONResponse(result)


@app.post("/api/simulate/reset")
async def api_simulate_reset(request: Request) -> JSONResponse:
    body = await request.json()
    session_id = str(body.get("session_id") or "demo-local")
    call_manager.end_call(session_id)
    call_manager.start_call(session_id, from_number=str(body.get("phone") or ""))
    return JSONResponse({"ok": True, "reply": call_manager.greeting(), "phase": "ask_name"})


@app.post("/sip/turn")
async def sip_turn_endpoint(request: Request) -> JSONResponse:
    """One dialogue turn for an Iranian Asterisk/SIP PBX."""
    body = await request.json()
    result = sip_turn(
        str(body.get("session_id") or body.get("call_id") or "sip-local"),
        str(body.get("text") or ""),
        str(body.get("phone") or ""),
    )
    return JSONResponse(result)


@app.get("/api/tts")
def api_tts(text: str) -> Response:
    wav = tts.synthesize_wav(text)
    return Response(content=wav, media_type="audio/wav")


def create_app() -> FastAPI:
    return app
