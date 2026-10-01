"""Same-origin phone interface for the room assistant."""

from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, ImageDraw
import numpy as np

from .agent import MAX_AUDIO, MAX_MESSAGE, RobotAgent


LOGGER = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
AUDIO_TYPES = {"audio/webm", "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/ogg", "video/webm", "video/mp4"}


def map_picture(snapshot):
    """Draw only the observed occupancy grid and estimated pose."""
    cells = np.asarray(snapshot['cells'])
    if cells.ndim != 2 or not 1 <= max(cells.shape) <= 601:
        raise ValueError('Invalid sensor map dimensions.')
    rgb = np.full((*cells.shape, 3), (35, 48, 58), dtype=np.uint8)
    rgb[cells == 0] = (232, 239, 224)
    rgb[cells == 100] = (97, 133, 131)
    observed = np.argwhere(cells >= 0)
    if observed.size:
        lower = np.maximum(observed.min(axis=0) - 12, 0)
        upper = np.minimum(observed.max(axis=0) + 13, cells.shape)
    else:
        lower, upper = np.array([0, 0]), np.array(cells.shape)
    picture = Image.fromarray(rgb[lower[0]:upper[0], lower[1]:upper[1]][::-1])
    scale = min(690 / picture.width, 350 / picture.height)
    picture = picture.resize((max(1, round(picture.width * scale)), max(1, round(picture.height * scale))), Image.Resampling.NEAREST)
    result = Image.new('RGB', (720, 405), (23, 45, 53))
    offset = ((720 - picture.width) // 2, 38 + (350 - picture.height) // 2)
    result.paste(picture, offset)
    draw = ImageDraw.Draw(result)
    draw.text((18, 12), 'SENSOR MAP   pale = observed free   teal = blocked   dark = unknown', fill=(222, 239, 235))
    pose, origin, resolution = snapshot['pose'], snapshot['origin'], snapshot['resolution_m']
    x = offset[0] + ((pose['x'] - origin[0]) / resolution - lower[1]) * scale
    y = offset[1] + (upper[0] - 1 - (pose['y'] - origin[1]) / resolution) * scale
    yaw = pose['yaw']
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(246, 166, 94))
    draw.line((x, y, x + 16 * math.cos(yaw), y - 16 * math.sin(yaw)), fill=(246, 166, 94), width=3)
    return result


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=MAX_MESSAGE)


class PersonPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = Field(strict=True, ge=-20, le=20, allow_inf_nan=False)
    y: float = Field(strict=True, ge=-20, le=20, allow_inf_nan=False)


def create_app(assistant, *, agent=None, token: str | None = None,
               tick_interval: float | None = None, frame_interval: float = .05,
               frame_clock=time.monotonic) -> FastAPI:
    """Create the app. Tick and tool execution share one short-held robot lock."""
    lock = getattr(assistant, "lock", None) or threading.RLock()
    agent = agent or RobotAgent(assistant, lock=lock)
    lock = agent.lock
    token = os.getenv("GO2_HTTP_TOKEN", "") if token is None else token
    simulator_controls = (
        getattr(assistant.backend, "is_simulator", False) is True
        and callable(getattr(assistant.backend, "set_person", None))
    )
    if tick_interval is None:
        tick_interval = .05 if simulator_controls else .1
    if not math.isfinite(frame_interval) or frame_interval <= 0:
        raise ValueError('Frame interval must be positive and finite.')
    frame_cache = {}
    frame_sequence = 0

    def simulator_person():
        if not simulator_controls:
            return None
        getter = getattr(assistant.backend, 'get_person', None)
        person = getter() if callable(getter) else assistant.backend.status().get("person")
        if not isinstance(person, dict):
            return None
        if not all(isinstance(person.get(key), (int, float)) and not isinstance(person[key], bool)
                   and math.isfinite(person[key]) for key in ("x", "y")):
            return None
        return {"x": float(person["x"]), "y": float(person["y"])}

    def require_access(request: Request):
        if token:
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
                raise HTTPException(401, "Enter the server's access token.")
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "Requests must come from this page's origin.")

    def tick_once():
        with lock:
            assistant.tick(tick_interval)

    @asynccontextmanager
    async def lifespan(app):
        async def run_ticks():
            loop = asyncio.get_running_loop()
            deadline = loop.time() + tick_interval
            while True:
                await asyncio.sleep(max(0., deadline - loop.time()))
                try:
                    # MuJoCo's macOS CGL renderer must stay on the thread that
                    # created it. Physics advances in short increments here.
                    tick_once()
                except Exception:
                    LOGGER.exception("Robot tick failed; stopping the assistant")
                    app.state.tick_error = "Robot update failed. The assistant stopped. Restart the server after checking the backend."
                    await asyncio.to_thread(agent.stop)
                    return
                # Schedule from the previous deadline, rather than adding a
                # full sleep after the physics/render work. Bound catch-up so
                # a slow frame never creates an unbounded queue of updates.
                deadline += tick_interval
                if deadline < loop.time() - tick_interval:
                    deadline = loop.time()

        task = asyncio.create_task(run_ticks()) if tick_interval > 0 else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            await asyncio.to_thread(agent.stop)

    app = FastAPI(title="Go2 room assistant", lifespan=lifespan)
    app.state.assistant = assistant
    app.state.agent = agent
    app.state.lock = lock
    app.state.tick_error = None

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/config")
    def config():
        return {"mode": agent.mode, "token_required": bool(token), "transcription": agent.transcription_available,
                "simulator": simulator_controls, "frame_rate": 1 / frame_interval}

    @app.get("/api/status")
    def status(request: Request):
        require_access(request)
        with lock:
            return {
                "status": assistant.status(), "rooms": assistant.memory.rooms(),
                "objects": assistant.memory.objects(), "history": assistant.memory.history()[-50:],
                "mode": agent.mode, "error": app.state.tick_error, "simulator_person": simulator_person(),
            }

    @app.post("/api/sim/person")
    async def move_person(position: PersonPosition, request: Request):
        require_access(request)
        if not simulator_controls:
            raise HTTPException(403, "Person controls are available only in the simulator.")
        if app.state.tick_error:
            raise HTTPException(503, app.state.tick_error)
        with lock:
            try:
                # This moves the test human, never the robot. The backend checks
                # the floorplan and owns the main-thread rendering context.
                assistant.backend.set_person(position.x, position.y)
                frame_cache.clear()
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(503, "The simulator could not move the person.") from exc
            return {"person": simulator_person(), "status": assistant.status()}

    @app.get("/api/map")
    def sensor_map(request: Request):
        require_access(request)
        with lock:
            getter = getattr(assistant.backend, 'map_snapshot', None)
            if not callable(getter):
                raise HTTPException(503, 'This connection does not expose a sensor map.')
            return getter()

    @app.post("/api/chat")
    def chat(message: ChatMessage, request: Request):
        require_access(request)
        if app.state.tick_error:
            raise HTTPException(503, app.state.tick_error)
        try:
            return agent.respond(message.message)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/stop")
    def stop(request: Request):
        require_access(request)
        return agent.stop()

    @app.get("/api/frame/{view}")
    async def frame(view: str, request: Request):
        nonlocal frame_sequence
        require_access(request)
        if view not in {"camera", "overview", "map"}:
            raise HTTPException(404, "View must be camera, overview, or map.")
        try:
            cached = frame_cache.get(view)
            revision = getattr(assistant.backend, 'elapsed', None)
            if (cached and cached['revision'] == revision
                    and 0 <= frame_clock() - cached['at'] < frame_interval):
                return Response(cached['bytes'], media_type='image/jpeg', headers=cached['headers'])
            with lock:
                picture = map_picture(assistant.backend.map_snapshot()) if view == 'map' else assistant.backend.frame("robot" if view == "camera" else view)
                output = io.BytesIO()
                picture.save(output, format="JPEG", quality=82)
            frame_sequence += 1
            encoded = output.getvalue()
            headers = {"Cache-Control": "no-store", "X-Frame-Sequence": str(frame_sequence)}
            frame_cache[view] = {'at': frame_clock(), 'revision': revision, 'bytes': encoded, 'headers': headers}
            return Response(encoded, media_type="image/jpeg", headers=headers)
        except Exception as exc:
            frame_cache.pop(view, None)
            LOGGER.warning("Frame unavailable: %s", type(exc).__name__)
            raise HTTPException(503, "This view is currently unavailable.") from exc

    @app.post("/api/transcribe")
    async def transcribe(request: Request):
        require_access(request)
        if not agent.transcription_available:
            raise HTTPException(503, "Audio transcription is not configured. Use text or browser dictation.")
        content_type = request.headers.get("content-type", "").split(";")[0].strip()
        if content_type not in AUDIO_TYPES:
            raise HTTPException(415, "Upload a supported audio recording.")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_AUDIO:
                raise HTTPException(413, "Audio must be 8 MiB or smaller.")
            chunks.append(chunk)
        try:
            text = await asyncio.to_thread(
                agent.provider.transcribe, b"".join(chunks),
                request.headers.get("x-filename", "recording.webm"), content_type,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(502, str(exc)) from exc
        return {"text": text}

    return app
