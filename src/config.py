"""Load environment variables and expose a single application config object."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root: parent of src/
ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_env_file() -> None:
    """Notepad on Windows often saves .env as ANSI/cp1256, not UTF-8."""
    path = ROOT_DIR / ".env"
    if not path.is_file():
        return
    for encoding in ("utf-8-sig", "utf-8", "cp1256", "cp1252", "latin-1"):
        try:
            load_dotenv(path, encoding=encoding)
            return
        except UnicodeDecodeError:
            continue


_load_env_file()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


@dataclass
class Config:
    """Runtime configuration loaded from the environment."""

    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    public_base_url: str
    receptionist_number: str
    kavenegar_api_key: str
    kavenegar_sender: str

    shenava_model_id: str
    shenava_model_path: Path
    shenava_ctc_path: Path
    shenava_head: str
    shenava_threads: int
    gooya_api_url: str
    gooya_api_token: str
    gooya_timeout_s: float
    stt_engine: str
    whisper_model_id: str
    whisper_model_path: Path
    whisper_ct2_repo: str
    whisper_threads: int
    whisper_compute: str
    whisper_prompt: str
    whisper_persian_only: bool
    piper_model_path: Path
    piper_config_path: Path

    ollama_url: str
    ollama_model: str
    ollama_enabled: bool

    db_path: Path

    host: str
    port: int
    log_level: str
    log_dir: Path

    office_address: str
    office_hours_start: str
    office_hours_end: str

    # Audio / VAD
    twilio_sample_rate: int = 8000
    stt_sample_rate: int = 16000
    vad_silence_ms: int = 500
    min_utterance_ms: int = 400
    energy_threshold: int = 500

    @property
    def twilio_configured(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token and self.twilio_phone_number)

    @property
    def kavenegar_configured(self) -> bool:
        return bool(self.kavenegar_api_key)


def load_config() -> Config:
    return Config(
        twilio_account_sid=_env("TWILIO_ACCOUNT_SID"),
        twilio_auth_token=_env("TWILIO_AUTH_TOKEN"),
        twilio_phone_number=_env("TWILIO_PHONE_NUMBER"),
        public_base_url=_env("PUBLIC_BASE_URL", "http://127.0.0.1:38471").rstrip("/"),
        receptionist_number=_env("RECEPTIONIST_NUMBER", "+989120000000"),
        kavenegar_api_key=_env("KAVENEGAR_API_KEY"),
        kavenegar_sender=_env("KAVENEGAR_SENDER"),
        shenava_model_id=_env("SHENAVA_MODEL_ID", "Reza2kn/Shenava-Koochik-v1.5"),
        shenava_model_path=_resolve_path(
            _env("SHENAVA_MODEL_PATH", "./models/shenava-koochik-v1.5")
        ),
        shenava_ctc_path=_resolve_path(
            _env("SHENAVA_CTC_PATH", "./models/shenava-koochik-ctc")
        ),
        shenava_head=_env("SHENAVA_HEAD", "ctc").lower() or "ctc",
        shenava_threads=int(_env("SHENAVA_THREADS", "4") or "4"),
        gooya_api_url=_env("GOOYA_API_URL"),
        gooya_api_token=_env("GOOYA_API_TOKEN"),
        gooya_timeout_s=float(_env("GOOYA_TIMEOUT_S", "25") or "25"),
        stt_engine=(_env("STT_ENGINE", "whisper") or "whisper").lower(),
        whisper_model_id=_env("WHISPER_MODEL_ID", "nezamisafa/whisper-persian-v4"),
        whisper_model_path=_resolve_path(
            _env("WHISPER_MODEL_PATH", "./models/whisper-persian-v4-ct2")
        ),
        whisper_ct2_repo=_env("WHISPER_CT2_REPO", "AlexAnoshka/fast-whisper-persian-v4"),
        # CTranslate2 slows down badly past the physical core count: 12 threads
        # measured ~4x slower than 4 on the same machine. 0 = auto-cap.
        whisper_threads=int(_env("WHISPER_THREADS", "0") or "0"),
        whisper_compute=_env("WHISPER_COMPUTE", "int8") or "int8",
        whisper_prompt=_env("WHISPER_PROMPT"),
        whisper_persian_only=_env("WHISPER_PERSIAN_ONLY", "1").lower()
        not in ("0", "false", "no", "off"),
        piper_model_path=_resolve_path(
            _env("PIPER_MODEL_PATH", "./models/piper-voice-fa/fa_IR-mana-medium.onnx")
        ),
        piper_config_path=_resolve_path(
            _env("PIPER_CONFIG_PATH", "./models/piper-voice-fa/fa_IR-mana-medium.onnx.json")
        ),
        ollama_url=_env("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=_env("OLLAMA_MODEL", "llama3.2:3b"),
        ollama_enabled=_env("OLLAMA_ENABLED", "0").lower() not in ("0", "false", "no", "off"),
        db_path=_resolve_path(_env("DB_PATH", "./appointments.db")),
        host=_env("HOST", "0.0.0.0"),
        port=int(_env("PORT", "38471")),
        log_level=_env("LOG_LEVEL", "INFO"),
        log_dir=_resolve_path(_env("LOG_DIR", "./logs")),
        office_address=_env("OFFICE_ADDRESS", "خیابان ایثار، کوچه خواجه پلاک ۲"),
        office_hours_start=_env("OFFICE_HOURS_START", "09:00"),
        office_hours_end=_env("OFFICE_HOURS_END", "17:00"),
        energy_threshold=int(_env("VAD_MIN_ENERGY", "500") or "500"),
    )


config = load_config()
