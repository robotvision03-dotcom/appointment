"""Browser microphone live voice over WebSocket — works in Iran without Twilio."""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.call_manager import PHASE_ASK_TYPE, call_manager
from src.config import config
from src.handoff import start_warm_transfer
from src.hearing import speech_onset, update_noise_floor
from src.sms import send_booking_sms
from src.stt import stt
from src.understand import understand
from src.utils import log, pcm16_rms

FRAME_MS_ESTIMATE = 30  # ~480 samples at 16 kHz


async def handle_browser_voice(websocket: WebSocket) -> None:
    """
    Protocol:
      client JSON  {"event":"start","session_id":"...","phone":"0912..."}
      client binary  Int16 little-endian PCM, 16 kHz mono
      client JSON  {"event":"stop"}
      server JSON  {"event":"assistant"|"user"|"status"|"error", ...}
    """
    await websocket.accept()
    call_sid = ""
    speech_started = False
    silence_ms = 0
    voiced_ms = 0
    pcm_buffer = bytearray()
    preroll = bytearray()
    sample_rate = config.stt_sample_rate
    noise_floor = 28.0
    peak_energy = 0.0
    # Whisper large-v3 needs seconds per utterance, so it runs off the receive
    # loop; otherwise incoming audio piles up and the reply lands after hangup.
    # Only the newest utterance waits: older audio is stale by the time we
    # would reach it, and the seller has already moved on.
    pending: dict[str, bytes | None] = {"chunk": None}
    worker: asyncio.Task | None = None
    tasks: set[asyncio.Task] = set()

    log.info("Browser voice WebSocket connected")

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text"):
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                event = msg.get("event")
                if event == "start":
                    call_sid = str(msg.get("session_id") or "browser-live")
                    try:
                        sample_rate = int(msg.get("sample_rate") or config.stt_sample_rate)
                    except (TypeError, ValueError):
                        sample_rate = config.stt_sample_rate
                    if sample_rate < 8000:
                        sample_rate = config.stt_sample_rate
                    phone = str(msg.get("phone") or "")
                    state = call_manager.get(call_sid) or call_manager.start_call(
                        call_sid, from_number=phone
                    )
                    if phone:
                        state.from_number = phone
                    if not state.context:
                        state.phase = PHASE_ASK_TYPE
                    log.info(
                        "Browser voice start sid=%s sample_rate=%s stt=%s",
                        call_sid,
                        sample_rate,
                        stt.available,
                    )
                    greeting = call_manager.greeting()
                    if not state.context:
                        await _send_assistant(websocket, greeting, "ask_type", "continue")
                    await websocket.send_json(
                        {
                            "event": "status",
                            "stt": stt.available,
                            "tts": False,
                            "message": None
                            if stt.available
                            else (
                                "مدل شنیدن آماده نیست. python -m src download-whisper را اجرا کنید "
                                "(nezamisafa/whisper-persian-v4) یا متن را تایپ کنید."
                            ),
                        }
                    )
                elif event == "stop":
                    if call_sid:
                        log.info("Browser voice stop %s", call_sid)
                    break
                elif event == "text":
                    text = str(msg.get("text") or "")
                    if call_sid and text:
                        await _handle_utterance(websocket, call_sid, text, echo_user=False)

            elif message.get("bytes"):
                pcm = message["bytes"]
                if not call_sid or len(pcm) < 2:
                    continue
                energy = pcm16_rms(pcm)
                duration_ms = int(1000 * (len(pcm) // 2) / sample_rate) or FRAME_MS_ESTIMATE
                preroll_cap = sample_rate * 2 // 3  # ~300 ms
                # Keep clips short. Whisper large-v3 int8 on CPU is ~realtime
                # at 2–5s and tens of seconds at the old 12s cap.
                max_bytes = sample_rate * 2 * max(2, int(config.whisper_max_seconds))
                noise_floor = update_noise_floor(noise_floor, float(energy), speech_started)
                loud = speech_onset(float(energy), noise_floor, float(config.energy_threshold))

                if loud:
                    if not speech_started:
                        pcm_buffer = bytearray(preroll)
                        speech_started = True
                        peak_energy = float(energy)
                        log.info(
                            "Hearing onset energy=%s floor=%.1f sid=%s",
                            energy,
                            noise_floor,
                            call_sid,
                        )
                    pcm_buffer.extend(pcm)
                    peak_energy = max(peak_energy, float(energy))
                    if len(pcm_buffer) > max_bytes:
                        del pcm_buffer[: len(pcm_buffer) - max_bytes]
                    voiced_ms += duration_ms
                    silence_ms = 0
                else:
                    preroll.extend(pcm)
                    if len(preroll) > preroll_cap:
                        del preroll[: len(preroll) - preroll_cap]
                    if speech_started:
                        pcm_buffer.extend(pcm)
                        silence_ms += duration_ms
                        if (
                            silence_ms >= config.vad_silence_ms
                            and voiced_ms >= config.min_utterance_ms
                        ):
                            chunk = bytes(pcm_buffer)
                            pcm_buffer = bytearray()
                            preroll = bytearray()
                            speech_started = False
                            silence_ms = 0
                            voiced_ms = 0
                            if peak_energy < max(config.energy_threshold, noise_floor * 2.5):
                                peak_energy = 0.0
                                continue
                            peak_energy = 0.0
                            queued = _queue_latest(pending, chunk)
                            await _send(
                                websocket,
                                {"event": "dictation", "text": "…", "hearing": True},
                            )
                            if queued:
                                log.info(
                                    "Utterance queued behind a slower one sid=%s", call_sid
                                )
                            if worker is None or worker.done():
                                worker = asyncio.create_task(
                                    _drain_utterances(
                                        websocket, call_sid, sample_rate, pending
                                    )
                                )
                                tasks.add(worker)
                                worker.add_done_callback(tasks.discard)

    except WebSocketDisconnect:
        log.info("Browser voice disconnected %s", call_sid)
    except Exception as exc:  # noqa: BLE001
        log.exception("Browser voice error: %s", exc)
    finally:
        for task in list(tasks):
            task.cancel()


async def _send(websocket: WebSocket, payload: dict) -> bool:
    """Send unless the browser already hung up."""
    if websocket.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError) as exc:
        log.info("Browser voice send skipped: %s", exc)
        return False


def _queue_latest(pending: dict[str, bytes | None], chunk: bytes) -> bool:
    """Keep only the newest utterance. Returns True when it displaced one."""
    replaced = pending["chunk"] is not None
    pending["chunk"] = chunk
    return replaced


async def _drain_utterances(
    websocket: WebSocket,
    call_sid: str,
    sample_rate: int,
    pending: dict[str, bytes | None],
) -> None:
    while True:
        chunk = pending["chunk"]
        if chunk is None:
            return
        pending["chunk"] = None
        await _transcribe_and_reply(websocket, call_sid, chunk, sample_rate)
        if websocket.client_state != WebSocketState.CONNECTED:
            return


async def _transcribe_and_reply(
    websocket: WebSocket,
    call_sid: str,
    chunk: bytes,
    sample_rate: int,
) -> None:
    started = time.monotonic()
    try:
        transcript = await asyncio.to_thread(stt.transcribe, chunk, sample_rate)
        elapsed = time.monotonic() - started
        if not transcript:
            log.info(
                "STT empty sid=%s bytes=%s rate=%s took=%.1fs",
                call_sid,
                len(chunk),
                sample_rate,
                elapsed,
            )
            await _send(
                websocket,
                {
                    "event": "status",
                    "message": "هنوز گفتار واضحی نبود. پس از صدای کلام، نام خودرو را بگویید.",
                },
            )
            return
        if websocket.client_state != WebSocketState.CONNECTED:
            log.info("Transcript %r arrived after hangup (%.1fs)", transcript, elapsed)
            return
        state = call_manager.get(call_sid)
        phase = state.phase if state else PHASE_ASK_TYPE
        info = state.patient_info if state else {}
        understood = await asyncio.to_thread(understand, transcript, phase, info)
        canonical = understood.get("text") or transcript
        shown = transcript if canonical == transcript else f"{transcript} → {canonical}"
        await _send(websocket, {"event": "dictation", "text": canonical})
        await _handle_utterance(websocket, call_sid, canonical, echo_text=shown)
    except Exception as exc:  # noqa: BLE001
        log.exception("Transcription task failed: %s", exc)


async def _handle_utterance(
    websocket: WebSocket,
    call_sid: str,
    text: str,
    echo_user: bool = True,
    echo_text: str | None = None,
) -> None:
    if echo_user:
        await _send(websocket, {"event": "user", "text": echo_text or text})
    result = await asyncio.to_thread(call_manager.handle_user_text, call_sid, text)
    if result.get("intent") == "book":
        await asyncio.to_thread(send_booking_sms, call_sid)
    if result.get("intent") == "transfer":
        result["transfer"] = start_warm_transfer(call_sid)
        extra = (result["transfer"] or {}).get("reply_extra")
        if extra:
            result["reply"] = f"{result.get('reply', '')} {extra}".strip()
    await _send_assistant(
        websocket,
        result.get("reply") or "",
        result.get("phase") or "",
        result.get("intent") or "continue",
        extra={
            "transfer": result.get("transfer"),
            "appointment_id": result.get("appointment_id"),
            "connect": result.get("connect"),
            "providers": result.get("providers"),
            "patient_info": result.get("patient_info"),
        },
    )


async def _send_assistant(
    websocket: WebSocket,
    text: str,
    phase: str,
    intent: str,
    extra: dict | None = None,
) -> None:
    payload = {"event": "assistant", "text": text, "phase": phase, "intent": intent}
    if extra:
        payload.update({k: v for k, v in extra.items() if v is not None})
    await _send(websocket, payload)
