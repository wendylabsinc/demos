"""Audio decoding checks and an opt-in real trained-model fixture test."""

import io
import os
import wave
from pathlib import Path

import numpy as np
import pytest

from yard_drone.voice import MAX_AUDIO_BYTES, VoiceService, decode_microphone_audio


def wav_bytes(samples, rate=16000, channels=1):
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return encoded.getvalue()


def test_decode_resamples_stereo_and_limits_duration():
    pytest.importorskip("av")
    times = np.arange(48000) / 48000
    left = np.sin(times * 440 * 2 * np.pi) * 10000
    audio = wav_bytes(np.column_stack([left, left]), rate=48000, channels=2)
    decoded = decode_microphone_audio(audio, "audio/wav")
    assert decoded.ndim == 1
    assert abs(decoded.size - 16000) < 2
    assert np.max(np.abs(decoded)) > 0.2
    long_audio = wav_bytes(np.zeros(13 * 16000, dtype=np.int16))
    with pytest.raises(ValueError, match="shorter than 12 seconds"):
        decode_microphone_audio(long_audio, "audio/wav")


def test_audio_size_and_format_validation():
    with pytest.raises(ValueError, match="empty"):
        decode_microphone_audio(b"", "audio/wav")
    with pytest.raises(ValueError, match="8 MiB"):
        decode_microphone_audio(b"x" * (MAX_AUDIO_BYTES + 1), "audio/webm")
    with pytest.raises(ValueError, match="format"):
        decode_microphone_audio(b"x", "application/json")


def test_browser_webm_opus_audio_decodes():
    av = pytest.importorskip("av")
    samples = (np.sin(np.arange(48000) / 48000 * 440 * 2 * np.pi) * 10000).astype(np.int16)
    encoded = io.BytesIO()
    with av.open(encoded, "w", format="webm") as container:
        stream = container.add_stream("libopus", rate=48000)
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 48000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    decoded = decode_microphone_audio(encoded.getvalue(), "audio/webm;codecs=opus")
    assert abs(decoded.size - 16000) < 2
    assert np.abs(decoded).max() > .2


def test_silence_never_loads_or_runs_model(tmp_path):
    pytest.importorskip("av")
    service = VoiceService(tmp_path)
    assert service.transcribe(wav_bytes(np.zeros(16000, dtype=np.int16)), "audio/wav") == ""
    assert service.status()["ready"] is False


def test_missing_weights_fail_locally(tmp_path):
    service = VoiceService(tmp_path)
    with pytest.raises(RuntimeError, match="not installed"):
        service.warmup()
    assert service.status()["error"]


def test_real_speech_fixture():
    """Set a real spoken fixture and its expected words to exercise Whisper."""
    fixture = os.environ.get("YARD_DRONE_VOICE_TEST_AUDIO")
    expected = os.environ.get("YARD_DRONE_VOICE_TEST_EXPECT", "")
    model = os.environ.get("YARD_DRONE_WHISPER_MODEL")
    if not fixture or not expected or not model:
        pytest.skip("Set a speech fixture, expected words, and local Whisper model path")
    pytest.importorskip("faster_whisper")
    service = VoiceService(model)
    service.warmup()
    transcript = service.transcribe(Path(fixture).read_bytes(), "audio/wav").lower()
    for word in expected.lower().split():
        assert word in transcript, transcript
