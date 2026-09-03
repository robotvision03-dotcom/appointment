from src.whisper_fa import ENGINE, _car_prompt


def test_whisper_engine_name():
    assert ENGINE == "whisper-persian-v4"


def test_car_prompt_contains_iranian_models():
    prompt = _car_prompt()
    assert "پژو" in prompt
    assert "پارس" in prompt or "سمند" in prompt
    assert "RD" not in prompt
