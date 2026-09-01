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

    vosk_model_path: Path
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

    # Audio / VAD
    twilio_sample_rate: int = 8000
    stt_sample_rate: int = 16000
    vad_silence_ms: int = 900
    min_utterance_ms: int = 400
    energy_threshold: int = 180

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
        vosk_model_path=_resolve_path(_env("VOSK_MODEL_PATH", "./models/vosk-model-fa")),
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
    )


config = load_config()
