"""Offline Persian TTS via Piper, with a WAV-file cache for Twilio <Play>."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.config import ROOT_DIR, config
from src.utils import generate_tone_pcm, log, pcm16_to_wav_bytes, write_wav

GENERATED_DIR = ROOT_DIR / "static" / "generated"


class OfflineTTS:
    """Convert Persian text to 16-bit PCM (or WAV) using a local Piper voice."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        self.model_path = Path(model_path or config.piper_model_path)
        self.config_path = Path(config_path or config.piper_config_path)
        self._voice = None
        self.available = False
        self.sample_rate = 22050
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            log.warning(
                "Piper model not found at %s — TTS will emit a short tone. "
                "Run scripts/download_models.sh",
                self.model_path,
            )
            return
        try:
            from piper import PiperVoice

            # config_path is optional if a sibling .onnx.json exists
            kwargs = {}
            if self.config_path.exists():
                kwargs["config_path"] = str(self.config_path)
            self._voice = PiperVoice.load(str(self.model_path), **kwargs)
            self.sample_rate = getattr(self._voice, "config", None) and getattr(
                self._voice.config, "sample_rate", 22050
            ) or 22050
            self.available = True
            log.info("Loaded Piper voice from %s", self.model_path)
        except TypeError:
            # Older piper-tts: PiperVoice.load(model_path, config_path)
            try:
                from piper import PiperVoice

                self._voice = PiperVoice.load(str(self.model_path), str(self.config_path))
                self.available = True
                log.info("Loaded Piper voice (legacy API) from %s", self.model_path)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to load Piper: %s", exc)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to load Piper: %s", exc)

    def synthesize(self, text: str) -> bytes:
        """Return 16-bit mono PCM bytes for the given Persian text."""
        text = (text or "").strip()
        if not text:
            return b""
        if not self.available or self._voice is None:
            log.warning("TTS unavailable; returning placeholder tone for: %s", text[:80])
            return generate_tone_pcm(duration_ms=350, freq=523)
        try:
            chunks: list[bytes] = []
            # piper-tts 1.2 yields AudioChunk objects or raw bytes depending on version
            for chunk in self._voice.synthesize(text):
                if isinstance(chunk, (bytes, bytearray)):
                    chunks.append(bytes(chunk))
                elif hasattr(chunk, "audio_int16_bytes"):
                    chunks.append(bytes(chunk.audio_int16_bytes))
                elif hasattr(chunk, "audio_float_array"):
                    import numpy as np

                    arr = np.asarray(chunk.audio_float_array)
                    pcm = (arr * 32767).clip(-32768, 32767).astype("<i2").tobytes()
                    chunks.append(pcm)
            audio = b"".join(chunks)
            if not audio:
                # Some versions write to a wav file-like via synthesize_wav
                import io
                import wave

                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    if hasattr(self._voice, "synthesize_wav"):
                        self._voice.synthesize_wav(text, wf)
                buf.seek(0)
                with wave.open(buf, "rb") as wf:
                    audio = wf.readframes(wf.getnframes())
            log.info("TTS synthesized %d bytes for %d chars", len(audio), len(text))
            return audio
        except Exception as exc:  # noqa: BLE001
            log.error("TTS failed: %s", exc)
            return generate_tone_pcm(duration_ms=350, freq=400)

    def synthesize_wav(self, text: str) -> bytes:
        pcm = self.synthesize(text)
        rate = self.sample_rate if self.available else 16000
        return pcm16_to_wav_bytes(pcm, sample_rate=rate)

    def synthesize_to_file(self, text: str, filename: str | None = None) -> Path:
        """Write WAV under static/generated and return the path (for Twilio <Play>)."""
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        path = GENERATED_DIR / (filename or f"tts_{digest}.wav")
        pcm = self.synthesize(text)
        rate = self.sample_rate if self.available else 16000
        write_wav(path, pcm, sample_rate=rate)
        return path


tts = OfflineTTS()
