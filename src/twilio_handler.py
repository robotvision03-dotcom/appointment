"""Twilio Programmable Voice: incoming webhooks, Media Streams, warm transfer, SMS."""

from __future__ import annotations

import audioop
import asyncio
import base64
import json
from typing import Any

from fastapi import Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.twiml.voice_response import Connect, Dial, Stream, VoiceResponse

from src.call_manager import PHASE_ASK_NAME, call_manager
from src.config import config
from src.stt import stt
from src.tts import tts
from src.utils import log, pcm16_rms, pcm16_to_mulaw

# Twilio sends ~20 ms frames of μ-law. Accumulate until silence after speech.
FRAME_MS = 20


def twiml_response(xml: str) -> Response:
    return Response(content=xml, media_type="application/xml")


def incoming_call_twiml(websocket_url: str) -> str:
    """TwiML: greet is spoken over the Media Stream after `start`, not here."""
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=websocket_url)
    connect.append(stream)
    response.append(connect)
    return str(response)


async def handle_incoming_call(request: Request) -> Response:
    """Twilio voice webhook: start a bidirectional Media Stream to /voice/stream."""
    form = await request.form()
    call_sid = str(form.get("CallSid") or "")
    from_number = str(form.get("From") or "")
    to_number = str(form.get("To") or "")
    log.info("Incoming call CallSid=%s From=%s To=%s", call_sid, from_number, to_number)
    if call_sid:
        call_manager.start_call(call_sid, from_number=from_number, to_number=to_number)

    # Twilio requires wss:// for Media Streams.
    proto = "wss" if request.url.scheme == "https" else "ws"
    # Prefer PUBLIC_BASE_URL so ngrok/https works even behind a proxy.
    if config.public_base_url.startswith("https://"):
        ws_url = config.public_base_url.replace("https://", "wss://", 1) + "/voice/stream"
    elif config.public_base_url.startswith("http://"):
        ws_url = config.public_base_url.replace("http://", "ws://", 1) + "/voice/stream"
    else:
        ws_url = f"{proto}://{request.url.hostname}:{request.url.port or 443}/voice/stream"

    xml = incoming_call_twiml(ws_url)
    log.info("Returning stream TwiML url=%s", ws_url)
    return twiml_response(xml)


async def handle_media_stream(websocket: WebSocket) -> None:
    """Receive Twilio Media Stream events, run STT→dialogue→TTS, stream audio back."""
    await websocket.accept()
    call_sid = ""
    stream_sid = ""
    recognizer = stt.make_recognizer(config.stt_sample_rate)
    speech_started = False
    silence_ms = 0
    voiced_ms = 0
    pcm_buffer = bytearray()
    inbound_ratecv_state = None

    log.info("Media stream WebSocket connected")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = msg.get("event")

            if event == "connected":
                log.info("Twilio stream connected protocol=%s", msg.get("protocol"))

            elif event == "start":
                start = msg.get("start") or {}
                stream_sid = start.get("streamSid") or msg.get("streamSid") or ""
                call_sid = (start.get("callSid") or "") or call_sid
                from_num = ((start.get("customParameters") or {}).get("from")) or ""
                state = call_manager.get(call_sid) or call_manager.start_call(call_sid)
                state.stream_sid = stream_sid
                if from_num:
                    state.from_number = from_num
                state.phase = PHASE_ASK_NAME
                log.info("Stream start call=%s stream=%s", call_sid, stream_sid)
                await _play_text(websocket, stream_sid, call_manager.greeting())

            elif event == "media":
                payload = (msg.get("media") or {}).get("payload")
                if not payload or not call_sid:
                    continue
                mulaw = base64.b64decode(payload)
                pcm8k, _ = audioop.ulaw2lin(mulaw, 2)
                pcm16, inbound_ratecv_state = audioop.ratecv(
                    pcm8k, 2, 1, 8000, config.stt_sample_rate, inbound_ratecv_state
                )
                energy = pcm16_rms(pcm16)
                pcm_buffer.extend(pcm16)

                if recognizer is not None:
                    final, _partial = stt.transcribe_partial(recognizer, pcm16)
                    if final:
                        pcm_buffer.clear()
                        speech_started = False
                        silence_ms = 0
                        voiced_ms = 0
                        await _on_utterance(websocket, stream_sid, call_sid, final)
                        continue

                if energy >= config.energy_threshold:
                    speech_started = True
                    voiced_ms += FRAME_MS
                    silence_ms = 0
                elif speech_started:
                    silence_ms += FRAME_MS
                    if (
                        silence_ms >= config.vad_silence_ms
                        and voiced_ms >= config.min_utterance_ms
                    ):
                        transcript = stt.transcribe(bytes(pcm_buffer), config.stt_sample_rate)
                        pcm_buffer.clear()
                        speech_started = False
                        silence_ms = 0
                        voiced_ms = 0
                        if recognizer is not None:
                            recognizer = stt.make_recognizer(config.stt_sample_rate)
                        if transcript:
                            await _on_utterance(websocket, stream_sid, call_sid, transcript)
                        else:
                            await _play_text(
                                websocket,
                                stream_sid,
                                call_sid and "متوجه نشدم. لطفاً دوباره بفرمایید.",
                            )

            elif event == "stop":
                log.info("Stream stop call=%s", call_sid)
                if call_sid:
                    call_manager.end_call(call_sid)
                break

            elif event == "mark":
                log.debug("Stream mark %s", msg.get("mark"))

    except WebSocketDisconnect:
        log.info("Media stream disconnected call=%s", call_sid)
        if call_sid:
            call_manager.end_call(call_sid)
    except Exception as exc:  # noqa: BLE001
        log.exception("Media stream error: %s", exc)
        if call_sid:
            call_manager.end_call(call_sid)


async def _on_utterance(websocket: WebSocket, stream_sid: str, call_sid: str, text: str) -> None:
    log.info("Utterance call=%s text=%s", call_sid, text)
    result = await asyncio.to_thread(call_manager.handle_user_text, call_sid, text)
    reply = result.get("reply") or ""
    if reply:
        await _play_text(websocket, stream_sid, reply)
    if result.get("intent") == "book":
        await asyncio.to_thread(_maybe_send_sms, call_sid)
    if result.get("intent") == "transfer":
        await asyncio.to_thread(start_warm_transfer, call_sid)


async def _play_text(websocket: WebSocket, stream_sid: str, text: str) -> None:
    if not text or not stream_sid:
        return
    pcm = await asyncio.to_thread(tts.synthesize, text)
    if not pcm:
        return
    rate = tts.sample_rate if tts.available else 16000
    mulaw = pcm16_to_mulaw(pcm, source_rate=rate)
    # Twilio expects ~20ms of 8 kHz μ-law = 160 bytes per frame
    frame = 160
    for i in range(0, len(mulaw), frame):
        chunk = mulaw[i : i + frame]
        if len(chunk) < frame:
            chunk = chunk + b"\xff" * (frame - len(chunk))
        payload = base64.b64encode(chunk).decode("ascii")
        await websocket.send_text(
            json.dumps(
                {
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload},
                }
            )
        )
    await websocket.send_text(
        json.dumps(
            {
                "event": "mark",
                "streamSid": stream_sid,
                "mark": {"name": "tts_done"},
            }
        )
    )


def _twilio_client():
    if not config.twilio_configured:
        return None
    from twilio.rest import Client

    return Client(config.twilio_account_sid, config.twilio_auth_token)


def _maybe_send_sms(call_sid: str) -> None:
    body = call_manager.sms_body(call_sid)
    state = call_manager.get(call_sid)
    if not body or not state or not state.from_number:
        return
    client = _twilio_client()
    if client is None:
        log.warning("SMS skipped — Twilio credentials missing. Body would be: %s", body)
        return
    try:
        client.messages.create(
            body=body,
            from_=config.twilio_phone_number,
            to=state.from_number,
        )
        log.info("Confirmation SMS sent to %s", state.from_number)
    except Exception as exc:  # noqa: BLE001
        log.error("SMS failed: %s", exc)


def start_warm_transfer(call_sid: str) -> dict[str, Any]:
    """
    Dial the receptionist, play a Persian summary, then bridge both legs
    into a conference named after the CallSid. The Media Stream / AI leg
    is hung up once the conference is ready.
    """
    summary = call_manager.transfer_summary(call_sid)
    conference = f"clinic-{call_sid}"
    client = _twilio_client()
    if client is None:
        log.warning("Warm transfer skipped — Twilio not configured. Summary: %s", summary)
        return {"ok": False, "reason": "twilio_not_configured", "summary": summary, "conference": conference}

    wav_path = tts.synthesize_to_file(summary, filename=f"summary_{call_sid}.wav")
    play_url = f"{config.public_base_url}/static/generated/{wav_path.name}"
    status_url = f"{config.public_base_url}/voice/transfer-status"

    try:
        # Put the patient into the conference (they hear wait music until receptionist joins).
        client.calls(call_sid).update(
            twiml=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="en-US">Please wait.</Say>
  <Dial>
    <Conference startConferenceOnEnter="true" endConferenceOnExit="false"
                beep="false" waitUrl="">{conference}</Conference>
  </Dial>
</Response>"""
        )
        outbound = client.calls.create(
            to=config.receptionist_number,
            from_=config.twilio_phone_number,
            url=f"{config.public_base_url}/voice/receptionist-join?conference={conference}&play={play_url}",
            status_callback=status_url,
            status_callback_event=["completed"],
        )
        log.info("Warm transfer: receptionist call %s conference %s", outbound.sid, conference)
        return {
            "ok": True,
            "conference": conference,
            "receptionist_call_sid": outbound.sid,
            "summary": summary,
        }
    except Exception as exc:  # noqa: BLE001
        log.error("Warm transfer failed: %s", exc)
        return {"ok": False, "reason": str(exc), "summary": summary}


def receptionist_join_twiml(conference: str, play_url: str) -> str:
    """TwiML for the outbound receptionist call: play summary, then join conference."""
    response = VoiceResponse()
    if play_url:
        response.play(play_url)
    dial = Dial()
    dial.conference(
        conference,
        start_conference_on_enter=True,
        end_conference_on_exit=True,
        beep=False,
    )
    response.append(dial)
    return str(response)
