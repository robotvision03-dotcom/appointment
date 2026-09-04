from src.whisper_fa import ENGINE, _car_prompt


def test_whisper_engine_name():
    assert ENGINE == "whisper-persian-v4"


def test_car_prompt_contains_iranian_models():
    prompt = _car_prompt()
    assert "پژو" in prompt
    assert "پارس" in prompt or "سمند" in prompt
    assert "RD" not in prompt


def test_whisper_uses_car_prompt_by_default(monkeypatch):
    from src import whisper_fa as w

    monkeypatch.setattr(w.config, "whisper_prompt", "")
    engine = w.WhisperPersianSTT()
    assert "پژو" in engine._prompt


def test_whisper_mode_does_not_run_shenava_on_empty():
    from src.stt import HearingSTT

    class FakeWhisper:
        engine = "whisper-persian-v4"
        available = True
        last_error = None

        def transcribe(self, audio_data, sample_rate=16000):
            return ""

    class FakeShenava:
        engine = "shenava-koochik-ctc"
        available = True
        last_error = None
        called = False

        def transcribe(self, audio_data, sample_rate=16000):
            self.called = True
            return "پارس پو پارس"

    hearing = HearingSTT.__new__(HearingSTT)
    hearing.gooya = type("G", (), {"configured": False, "engine": "gooya"})()
    hearing.whisper = FakeWhisper()
    hearing.shenava = FakeShenava()
    hearing._last_engine = ""
    hearing.last_error = None
    hearing.mode = "whisper"
    assert hearing.transcribe(b"\x00\x10" * 8000) == ""
    assert hearing.shenava.called is False


def test_ollama_stays_off_when_disabled(monkeypatch):
    from src.llm import PersianLLM

    monkeypatch.setattr("src.llm.config.ollama_enabled", False)
    assert PersianLLM().is_available() is False
