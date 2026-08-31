# Persian AI Voice Agent — Medical Appointment Booking

منشی تلفنی فارسی برای نوبت‌دهی پزشکی. تماس ورودی Twilio را می‌گیرد، گفتار را با مدل آفلاین فارسی پیاده می‌کند، نوبت را در SQLite ذخیره می‌کند، و در صورت نیاز تماس را گرم به منشی انسان منتقل می‌کند.

بدون Twilio، مدل‌ها، یا Ollama هم می‌توانید جریان مکالمه را از رابط وب محلی آزمایش کنید؛ موتور گفتگو یک ماشین حالت قطعی فارسی دارد و در صورت در دسترس بودن Ollama از آن برای فهم آزادتر استفاده می‌کند.

## Features

- Incoming calls: Twilio Programmable Voice webhook + Media Streams (μ-law ↔ PCM)
- Offline STT: Vosk Persian (`vosk-model-fa-0.5` / `vosk-model-fa-0.22`)
- Local LLM: Ollama (`persianllama:7b`, `gemma3:4b`, or any chat model)
- Offline TTS: Piper `fa_IR-mena-medium`
- Booking: SQLite doctors + appointments, slot conflict checks
- Warm transfer: summary TTS to receptionist, then conference bridge
- Confirmation SMS via Twilio after a successful booking
- Rotating file logs under `logs/agent.log`
- Local demo UI at `/` for typed Persian conversations

## Requirements

- Python 3.10+
- Linux (Twilio Media Streams + `audioop` resampling)
- Optional: Ollama, Vosk model (~1 GB), Piper ONNX voice (~60 MB), Twilio account

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit credentials if you have them
python -m src
```

Open the demo UI (default port **38471**):

```text
http://127.0.0.1:38471
```

Health check: `GET /health`.

Example booking in the simulator:

1. علی رضایی
2. دکتر کریمی
3. فردا
4. ساعت ده صبح
5. بله

Transfer: `می‌خواهم با منشی صحبت کنم`

## Project layout

```text
src/main.py              FastAPI app, webhooks, demo APIs
src/twilio_handler.py    TwiML, Media Streams, warm transfer, SMS
src/call_manager.py      Per-call state + booking dialogue
src/stt.py               Vosk offline STT
src/llm.py               Ollama Persian LLM client
src/tts.py               Piper offline TTS
src/db.py                SQLite doctors/appointments
src/config.py            Environment
src/utils.py             Audio, Persian date/time, logging
static/                  Demo UI + generated WAV for <Play>
models/                  Download Vosk + Piper here
scripts/download_models.sh
scripts/pull_ollama.sh
tests/
```

## Environment

See `.env.example`. Important variables:

| Variable | Purpose |
| --- | --- |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_PHONE_NUMBER` | Twilio REST + caller ID |
| `PUBLIC_BASE_URL` | Public HTTPS origin (ngrok) for Media Streams `wss://` and `<Play>` |
| `RECEPTIONIST_NUMBER` | E.164 number for warm transfer |
| `VOSK_MODEL_PATH` | Directory of the Persian Vosk model |
| `PIPER_MODEL_PATH` / `PIPER_CONFIG_PATH` | Piper ONNX + JSON |
| `OLLAMA_URL` / `OLLAMA_MODEL` | Local LLM |
| `DB_PATH` | SQLite file |

## Download Persian models

```bash
chmod +x scripts/*.sh
./scripts/download_models.sh
```

- Vosk: [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) — use `vosk-model-fa-0.5` or `vosk-model-fa-0.22`. Unpack so `models/vosk-model-fa/am/` (or `conf/model.conf`) exists.
- Piper: [rhasspy/piper-voices … fa_IR/mena-medium](https://huggingface.co/rhasspy/piper-voices/tree/main/fa/fa_IR/mena-medium) — `fa_IR-mena-medium.onnx` and `.onnx.json`.

Without these files the HTTP API and dialogue still run; STT returns empty transcripts and TTS plays a short placeholder tone.

## Ollama

```bash
# install Ollama, then:
./scripts/pull_ollama.sh
# or:
ollama pull llama3.2:3b
# set OLLAMA_MODEL in .env
```

If Ollama is down, the agent still books using the built-in Persian state machine.

## Twilio setup

1. Buy a number with Voice.
2. Expose the server: `ngrok http 38471`
3. Set `PUBLIC_BASE_URL` to the ngrok HTTPS URL.
4. Voice webhook: `POST https://<public>/voice/incoming`
5. Media Streams use `wss://<public>/voice/stream` (derived from `PUBLIC_BASE_URL`).

### Warm transfer

When the patient asks for a human receptionist:

1. A summary is synthesized: «تماس از بیمار [نام] برای دکتر [پزشک] …»
2. The live call is redirected into a conference named `clinic-<CallSid>`.
3. Twilio dials `RECEPTIONIST_NUMBER`; that leg plays the summary WAV then joins the same conference.
4. Optional SMS is sent after a confirmed booking.

## HTTP endpoints

| Method | Path | Role |
| --- | --- | --- |
| GET | `/` | Demo UI |
| GET | `/health` | Component status |
| POST | `/voice/incoming` | Twilio voice webhook |
| WS | `/voice/stream` | Twilio Media Streams |
| GET/POST | `/voice/receptionist-join` | TwiML for outbound receptionist call |
| POST | `/api/simulate` | `{session_id, text}` dialogue turn |
| GET | `/api/doctors` | Roster |
| GET | `/api/appointments` | Bookings |

## Tests

```bash
pytest tests/test_dialogue.py -q
python tests/test_ai_local.py
# with the server running:
python tests/test_twilio_webhook.py --url http://127.0.0.1:38471/voice/incoming
```

Pass `--wav path/to/16k_mono.wav` to `test_ai_local.py` to hit Vosk on a recording.

## Production notes

- Run behind HTTPS; Twilio Media Streams require `wss://`.
- Prefer 16 GB RAM if you load Vosk + Piper + a 7B LLM on CPU; smaller Ollama models work on CPU.
- Point `uvicorn` at `src.main:app` with multiple workers only if you share call state externally — in-memory `CallManager` is per-process.
- Put `logs/` on a persistent volume. Generated WAVs live in `static/generated/`.
