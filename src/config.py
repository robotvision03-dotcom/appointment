"""Load environment variables and expose a single application config object."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root: parent of src/
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


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

    vosk_model_path: Path
    piper_model_path: Path
    piper_config_path: Path

    ollama_url: str
    ollama_model: str

    db_path: Path

    host: str
    port: int
    log_level: str
    log_dir: Path

    # Audio / VAD
    twilio_sample_rate: int = 8000
    stt_sample_rate: int = 16000
    vad_silence_ms: int = 900
    min_utterance_ms: int = 400
    energy_threshold: int = 180

    @property
    def twilio_configured(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token and self.twilio_phone_number)


def load_config() -> Config:
    return Config(
        twilio_account_sid=_env("TWILIO_ACCOUNT_SID"),
        twilio_auth_token=_env("TWILIO_AUTH_TOKEN"),
        twilio_phone_number=_env("TWILIO_PHONE_NUMBER"),
        public_base_url=_env("PUBLIC_BASE_URL", "http://127.0.0.1:38471").rstrip("/"),
        receptionist_number=_env("RECEPTIONIST_NUMBER", "+989120000000"),
        vosk_model_path=_resolve_path(_env("VOSK_MODEL_PATH", "./models/vosk-model-fa")),
        piper_model_path=_resolve_path(
            _env("PIPER_MODEL_PATH", "./models/piper-voice-fa/fa_IR-mena-medium.onnx")
        ),
        piper_config_path=_resolve_path(
            _env("PIPER_CONFIG_PATH", "./models/piper-voice-fa/fa_IR-mena-medium.onnx.json")
        ),
        ollama_url=_env("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=_env("OLLAMA_MODEL", "persianllama:7b"),
        db_path=_resolve_path(_env("DB_PATH", "./appointments.db")),
        host=_env("HOST", "0.0.0.0"),
        port=int(_env("PORT", "38471")),
        log_level=_env("LOG_LEVEL", "INFO"),
        log_dir=_resolve_path(_env("LOG_DIR", "./logs")),
    )


config = load_config()
