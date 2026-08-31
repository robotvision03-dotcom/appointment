# Persian AI Voice Agent — usable in Iran

منشی نوبت‌دهی پزشکی به زبان فارسی. **Twilio در ایران کار نمی‌کند** و دیگر مسیر اصلی نیست.

مسیرهایی که داخل ایران کار می‌کنند:

1. **وب (پیشنهادی)** — گفتگوی متنی یا تماس صوتی با میکروفون مرورگر روی همین سرور.
2. **پیامک کاوه‌نگار** — تأیید نوبت به موبایل ایرانی.
3. **سانترال ایرانی (اختیاری)** — ترانک SIP محلی + Asterisk که هر نوبت گفتار را به `POST /sip/turn` می‌فرستد.

گفتار و صدا **آفلاین** هستند (Vosk + Piper روی همان ماشین). به سرویس ابری خارجی وابسته نیستند.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env             # Linux/macOS: cp .env.example .env
python -m src
```

Open **http://127.0.0.1:38471**

Type a booking:

1. علی رضایی  
2. دکتر کریمی  
3. فردا  
4. ساعت ده صبح  
5. بله  

Or click **شروع تماس صوتی** (microphone). Without the Vosk model, type instead — Google speech APIs are often blocked in Iran.

Enter a mobile (`0912…`) to receive an SMS after booking if Kavenegar is configured.

## Iran stack

| Need | Tool | Why |
| --- | --- | --- |
| Chat / voice in clinic | This web app + WebSocket `/voice/live` | No foreign CPaaS |
| STT | Vosk Persian, local files | Offline |
| TTS | Piper `fa_IR-mena-medium`, local | Offline |
| SMS | [Kavenegar](https://kavenegar.com) | Iranian SMS gateway |
| Human receptionist | `tel:` click-to-call `RECEPTIONIST_NUMBER` | No Twilio conference |
| Landline PSTN | Local SIP trunk → Asterisk → `POST /sip/turn` | MCI / Shatel / Respina etc. |

## Kavenegar SMS

1. Sign up at https://panel.kavenegar.com  
2. Put the API key in `.env`:

```env
KAVENEGAR_API_KEY=your-key
KAVENEGAR_SENDER=1000xxxx
RECEPTIONIST_NUMBER=09121234567
```

Until the key is set, SMS is logged only (dry-run) and booking still succeeds.

## Microphone voice

```bash
./scripts/download_models.sh
```

Place Vosk under `models/vosk-model-fa` and Piper under `models/piper-voice-fa`. Then **شروع تماس صوتی** streams 16 kHz PCM to `/voice/live`.

## Iranian SIP / Asterisk (optional landline)

Point an Iranian SIP trunk at Asterisk. After local ASR (or Vosk on the PBX), POST each utterance:

```bash
curl -s http://127.0.0.1:38471/sip/turn \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"SIP-CALL-1","phone":"09121234567","text":"علی رضایی"}'
```

Play the reply with `GET /api/tts?text=...` (WAV). Example dialplan: `scripts/asterisk_dialplan.conf`.

## Environment

See `.env.example`. Twilio variables are unused for the Iran path.

## Tests

```bash
pytest tests/test_dialogue.py -q
python tests/test_ai_local.py
```
