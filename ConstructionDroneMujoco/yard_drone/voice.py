"""Local microphone transcription with trained Whisper and Silero models.

The model is provisioned when the application image is built. Runtime requests
do not download weights, contact a speech service, or interpret flight intent.
The server passes the returned transcript to its language service.
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import numpy as np

MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_AUDIO_SECONDS = 12
SAMPLE_RATE = 16000
SUPPORTED_MIME = {
    "audio/webm", "video/webm", "audio/ogg", "audio/mp4", "video/mp4",
    "audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3",
    "audio/aac", "audio/flac", "audio/x-flac", "audio/x-m4a",
    "application/octet-stream",
}


def decode_microphone_audio(audio: bytes, mime: str) -> np.ndarray:
    """Decode a bounded recording into mono 16 kHz PCM with PyAV."""
    if not isinstance(audio, bytes) or not audio:
        raise ValueError("The microphone recording is empty.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("The microphone recording exceeds 8 MiB.")
    if str(mime).partition(";")[0].strip().lower() not in SUPPORTED_MIME:
        raise ValueError("Send a supported microphone audio format.")
    try:
        import av
    except ImportError as error:
        raise RuntimeError("The onboard audio decoder is unavailable.") from error

    chunks: list[np.ndarray] = []
    count = 0
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

    def append(frame):
        nonlocal count
        samples = frame.to_ndarray().reshape(-1)
        count += samples.size
        if count > MAX_AUDIO_SECONDS * SAMPLE_RATE:
            raise ValueError("Record a voice command shorter than 12 seconds.")
        chunks.append(samples.astype(np.float32) / 32768.0)

    try:
        with av.open(io.BytesIO(audio), mode="r") as container:
            if not container.streams.audio:
                raise ValueError("The recording contains no audio track.")
            for frame in container.decode(audio=0):
                frame.pts = None
                for output in resampler.resample(frame):
                    append(output)
            for output in resampler.resample(None):
                append(output)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("The microphone recording could not be decoded.") from error
    if not chunks or count < SAMPLE_RATE // 10:
        raise ValueError("The microphone recording is too short.")
    return np.concatenate(chunks)


class VoiceService:
    """One CPU int8 Whisper model shared by serialized voice requests."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or os.environ.get(
            "YARD_DRONE_WHISPER_MODEL", "/opt/models/whisper-tiny.en"))
        self._model = None
        self._lock = threading.Lock()
        self.error: str | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        required = ("model.bin", "config.json", "tokenizer.json")
        if not self.model_path.is_dir() or any(
                not (self.model_path / name).is_file() for name in required):
            raise RuntimeError("The onboard Whisper model is not installed.")
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                str(self.model_path), device="cpu", compute_type="int8",
                cpu_threads=max(1, int(os.environ.get("YARD_DRONE_VOICE_THREADS", "2"))),
                num_workers=1, local_files_only=True)
        except Exception as error:
            raise RuntimeError("The onboard Whisper model could not start.") from error
        return self._model

    def warmup(self):
        """Load local weights before the server advertises voice readiness."""
        with self._lock:
            try:
                self._load_model()
                self.error = None
            except RuntimeError as error:
                self.error = str(error)
                raise

    def status(self) -> dict:
        return {"ready": self._model is not None and self.error is None,
                "engine": "faster-whisper", "model": self.model_path.name,
                "device": "cpu", "precision": "int8", "error": self.error}

    def transcribe(self, audio: bytes, mime: str = "audio/webm") -> str:
        samples = decode_microphone_audio(audio, mime)
        # Digital silence must not become a hallucinated flight command.
        if float(np.max(np.abs(samples))) <= 1 / 32768:
            return ""
        with self._lock:
            try:
                model = self._load_model()
                segments, _info = model.transcribe(
                    samples, language="en", task="transcribe", beam_size=3,
                    temperature=0, condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 450,
                                    "min_speech_duration_ms": 100},
                    no_speech_threshold=0.6, log_prob_threshold=-1.0)
                transcript = " ".join(segment.text.strip() for segment in segments).strip()
                self.error = None
                return transcript
            except RuntimeError as error:
                self.error = str(error)
                raise
            except Exception as error:
                self.error = "The onboard speech model could not transcribe this recording."
                raise RuntimeError(self.error) from error
