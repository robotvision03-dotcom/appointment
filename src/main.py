"""FastAPI: car appraisal office — voice booking + Airbnb-style calendar."""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src import db
from src.call_manager import call_manager
from src.config import ROOT_DIR, config
from src.jalali import to_jalali
from src.llm import llm
from src.live_voice import handle_browser_voice
from src.sip_bridge import sip_turn
from src.sms import send_booking_sms
from src.stt import stt
from src.twilio_handler import handle_incoming_call, handle_media_stream, twiml_response
from src.utils import log

app = FastAPI(
    title="دفتر کارشناسی خودرو",
    description="خرید خودرو از فروشنده: کارشناسی و تعیین قیمت رایگان با نوبت نیم‌ساعته",
    version="2.0.0",
)

STATIC_DIR = ROOT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    log.info(
        "Car office ready address=%s hearing=%s stt=%s",
        config.office_address,
        getattr(stt, "engine", "none"),
        stt.available,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "office": {
            "address": config.office_address,
            "hours": f"{config.office_hours_start}–{config.office_hours_end}",
            "slot_minutes": 30,
            "closed": "جمعه",
        },
        "stt": {
            "available": stt.available,
            "engine": stt.engine,
            "head": getattr(stt, "head", ""),
            "error": stt.last_error,
        },
        "tts": {"available": False},
        "llm": {"available": llm.is_available(), "model": llm.model},
        "sms": {"provider": "kavenegar", "available": config.kavenegar_configured},
        "voice": {"browser": True, "stt": stt.available, "tts": False},
        "cars": len(db.list_cars()),
    }


@app.post("/voice/incoming")
async def voice_incoming(request: Request) -> Response:
    return await handle_incoming_call(request)


@app.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket) -> None:
    await handle_media_stream(websocket)


@app.websocket("/voice/live")
async def voice_live(websocket: WebSocket) -> None:
    await handle_browser_voice(websocket)


@app.get("/api/cars")
def api_cars(q: str = "") -> list[dict]:
    rows = db.list_cars()
    if not q:
        return rows
    needle = q.strip()
    return [r for r in rows if needle in r["make"] or needle in r["model"] or needle in (r.get("keywords") or "")]


@app.get("/api/cars/makes")
def api_makes() -> list[str]:
    return db.list_makes()


@app.get("/api/calendar")
def api_calendar(year: int | None = None, month: int | None = None) -> dict:
    today = date.today()
    jy, jm, _jd = to_jalali(today)
    return db.month_calendar(year or jy, month or jm)


@app.get("/api/slots")
def api_slots(date: str) -> dict:
    return {
        "date": date,
        "slots": db.available_slots(date),
        "all": db.slot_times(),
        "taken": sorted(db.taken_times(date)),
        "address": config.office_address,
    }


@app.post("/api/book")
async def api_book(request: Request) -> JSONResponse:
    body = await request.json()
    name = str(body.get("seller_name") or body.get("name") or "").strip()
    phone = str(body.get("phone") or "").strip()
    make = str(body.get("make") or "").strip()
    model = str(body.get("model") or "").strip()
    year = str(body.get("year") or "").strip()
    day = str(body.get("date") or "").strip()
    time = str(body.get("time") or "").strip()
    km_raw = body.get("km")
    km = int(km_raw) if str(km_raw or "").isdigit() else None
    if not name or not make or not day or not time:
        return JSONResponse(
            {"ok": False, "error": "نام، نوع خودرو، تاریخ و ساعت لازم است."},
            status_code=400,
        )
    try:
        appt_id = db.book_inspection(name, phone, make, model, year, km, day, time)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    appt = db.get_inspection(appt_id)
    return JSONResponse(
        {
            "ok": True,
            "appointment": appt,
            "message": (
                f"نوبت ثبت شد. {day} ساعت {time} به {config.office_address} بیایید. "
                "کارشناسی برای فروشنده رایگان است."
            ),
        }
    )


@app.get("/api/appointments")
def api_appointments() -> list[dict]:
    return db.list_inspections()


@app.post("/api/simulate")
async def api_simulate(request: Request) -> JSONResponse:
    body = await request.json()
    session_id = str(body.get("session_id") or "demo-local")
    text = str(body.get("text") or "")
    if call_manager.get(session_id) is None:
        call_manager.start_call(session_id, from_number=str(body.get("phone") or ""))
        if not text:
            return JSONResponse(
                {
                    "reply": call_manager.greeting(),
                    "phase": "ask_type",
                    "intent": "continue",
                    "patient_info": {},
                    "appointment_id": None,
                    "call_sid": session_id,
                    "address": config.office_address,
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
    return JSONResponse(result)


@app.post("/api/simulate/reset")
async def api_simulate_reset(request: Request) -> JSONResponse:
    body = await request.json()
    session_id = str(body.get("session_id") or "demo-local")
    call_manager.end_call(session_id)
    call_manager.start_call(session_id, from_number=str(body.get("phone") or ""))
    return JSONResponse({"ok": True, "reply": call_manager.greeting(), "phase": "ask_type"})


@app.post("/sip/turn")
async def sip_turn_endpoint(request: Request) -> JSONResponse:
    body = await request.json()
    result = sip_turn(
        str(body.get("session_id") or body.get("call_id") or "sip-local"),
        str(body.get("text") or ""),
        str(body.get("phone") or ""),
    )
    return JSONResponse(result)


@app.get("/api/tts")
def api_tts(text: str) -> Response:
    return Response(content=b"", media_type="audio/wav", status_code=204)


def create_app() -> FastAPI:
    return app
