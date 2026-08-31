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
from src.twilio_handler import (
    handle_incoming_call,
    handle_media_stream,
    receptionist_join_twiml,
    start_warm_transfer,
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
        "Agent ready host=%s port=%s stt=%s tts=%s ollama=%s twilio=%s",
        config.host,
        config.port,
        stt.available,
        tts.available,
        llm.is_available(),
        config.twilio_configured,
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
        "llm": {"available": llm.is_available(), "url": config.ollama_url, "model": config.ollama_model},
        "twilio": config.twilio_configured,
        "doctors": len(db.list_doctors()),
    }


@app.post("/voice/incoming")
async def voice_incoming(request: Request) -> Response:
    return await handle_incoming_call(request)


@app.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket) -> None:
    await handle_media_stream(websocket)


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
        call_manager.start_call(session_id, from_number=str(body.get("phone") or "+989120000001"))
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
    result = call_manager.handle_user_text(session_id, text)
    if result.get("intent") == "transfer":
        result["transfer"] = start_warm_transfer(session_id)
    return JSONResponse(result)


@app.post("/api/simulate/reset")
async def api_simulate_reset(request: Request) -> JSONResponse:
    body = await request.json()
    session_id = str(body.get("session_id") or "demo-local")
    call_manager.end_call(session_id)
    call_manager.start_call(session_id, from_number="+989120000001")
    return JSONResponse({"ok": True, "reply": call_manager.greeting(), "phase": "ask_name"})


@app.get("/api/tts")
def api_tts(text: str) -> Response:
    wav = tts.synthesize_wav(text)
    return Response(content=wav, media_type="audio/wav")


def create_app() -> FastAPI:
    return app
