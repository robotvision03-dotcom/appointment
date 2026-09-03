"""Professional listen → denoise → speech-onset pipeline (16 kHz float/PCM16)."""

from __future__ import annotations

import numpy as np


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    raw = bytes(pcm)
    if len(raw) % 2:
        raw = raw[:-1]
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def highpass(wave: np.ndarray, sample_rate: int = 16000, cutoff_hz: float = 90.0) -> np.ndarray:
    """One-pole high-pass to drop rumble and fan noise before ASR."""
    if wave.size < 8:
        return wave
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)
    out = np.empty_like(wave)
    prev_x = wave[0]
    prev_y = 0.0
    for i, x in enumerate(wave):
        y = alpha * (prev_y + x - prev_x)
        out[i] = y
        prev_x, prev_y = x, y
    return out.astype(np.float32)


def noise_gate(wave: np.ndarray, floor: float) -> np.ndarray:
    """Attenuate bins quieter than the estimated noise floor."""
    if wave.size == 0:
        return wave
    frame = 320  # 20 ms at 16 kHz
    n = (wave.size // frame) * frame
    if n < frame:
        return wave
    shaped = wave[:n].reshape(-1, frame)
    rms = np.sqrt(np.mean(np.square(shaped), axis=1, keepdims=True))
    gain = np.clip((rms - floor) / max(floor, 1e-4), 0.0, 1.0)
    gain = np.sqrt(gain)
    gated = (shaped * gain).reshape(-1)
    if wave.size > n:
        gated = np.concatenate([gated, wave[n:]])
    return gated.astype(np.float32)


def speech_onset(energy: float, noise_floor: float, min_energy: float = 900.0) -> bool:
    """True when a speech highlight rises above background noise.

    Measured on a laptop mic: room noise peaks around 200–950 RMS while spoken
    words sit at 2500–3200, so the absolute floor matters more than the ratio.
    A noise trigger is expensive here — it costs a whole Whisper pass.
    """
    return energy >= max(min_energy, noise_floor * 4.0 + 60.0)


def update_noise_floor(noise_floor: float, energy: float, speaking: bool) -> float:
    if speaking:
        return noise_floor
    return 0.94 * noise_floor + 0.06 * energy


def prepare_for_asr(wave: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    hp = highpass(wave, sample_rate)
    frame = max(1, int(0.02 * sample_rate))
    n_frames = max(1, hp.size // frame)
    energies = np.sqrt(np.mean(np.square(hp[: n_frames * frame].reshape(n_frames, frame)), axis=1))
    noise = float(np.percentile(energies, 15)) if energies.size else 0.002
    peak_e = float(np.max(energies)) if energies.size else 0.0
    mid = float(np.median(energies)) if energies.size else 0.0
    if peak_e > mid * 1.8 and peak_e > 0.01:
        gated = noise_gate(hp, max(noise, 0.003))
    else:
        gated = hp
    peak = float(np.max(np.abs(gated))) if gated.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(gated)))) if gated.size else 0.0
    if rms > 1e-6:
        gated = gated * min(0.1 / rms, 12.0)
        np.clip(gated, -0.99, 0.99, out=gated)
    pad = np.zeros(int(0.25 * sample_rate), dtype=np.float32)
    return np.concatenate([gated, pad])
