"""Render a complete simulated cleaning mission, with camera cuts and telemetry."""
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .simulation import Simulator


def _font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _add_geom(scene, kind, size, position, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.array(size, dtype=float),
                        np.array(position, dtype=float), np.eye(3).ravel(),
                        np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def _frame(renderer, sim, state, frame_num, args):
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [sim.home["width"] / 2, sim.home["height"] / 2, -0.4]
    camera.distance = max(sim.home["width"], sim.home["height"]) * 1.65
    camera.azimuth = 112
    camera.elevation = -55
    if args.camera == "top":
        camera.elevation = -89
        camera.azimuth = 270
    elif args.camera == "tour" and frame_num / args.fps > 3:
        # A wide opening establishes the house, then alternate tracking and overview shots.
        shot = int(frame_num / args.fps / 8) % 3
        if shot == 1:
            camera.lookat[:] = [state["robot"]["x"], state["robot"]["y"], 0.1]
            camera.distance = 5.5
            camera.elevation = -48
            camera.azimuth = 112
        elif shot == 2:
            camera.elevation = -85
            camera.azimuth = 270
    renderer.update_scene(sim.data, camera)
    if mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "dust_0") < 0:
        for dust in state["dust"]:
            _add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      [0.028, 0.021, 0.007], [dust["x"], dust["y"], 0.035], [0.57, 0.37, 0.17, 1])
    trail = state["trail"][::3]
    for a, b in zip(trail, trail[1:]):
        if renderer.scene.ngeom >= renderer.scene.maxgeom:
            break
        _add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_CAPSULE,
                  [0.014, 0.01, 0], [a[0], a[1], 0.031], [0.16, 0.56, 0.37, 0.58])
        mujoco.mjv_connector(renderer.scene.geoms[renderer.scene.ngeom - 1],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, 0.014,
                             np.array([a[0], a[1], 0.031]), np.array([b[0], b[1], 0.031]))
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    overlay = Image.new("RGBA", img.size)
    draw = ImageDraw.Draw(overlay)
    scale = args.width / 1280
    margin = int(28 * scale)
    title_font, stat_font, small_font = (_font(int(n * scale)) for n in (26, 22, 14))
    draw.rounded_rectangle((margin, margin, int(385 * scale), int(113 * scale)),
                           radius=int(14 * scale), fill=(244, 242, 233, 240))
    draw.text((margin + 18 * scale, margin + 12 * scale), "SWEEP / Maple House", fill="#243d33", font=title_font)
    draw.text((margin + 18 * scale, margin + 50 * scale), "Autonomous vacuum  |  WendyOS + MuJoCo", fill="#5b6f64", font=small_font)
    bottom = args.height - margin
    draw.rounded_rectangle((margin, bottom - 80 * scale, args.width - margin, bottom),
                           radius=int(14 * scale), fill=(32, 53, 44, 238))
    phase = "Home, clean home" if state["status"] == "docked" else state.get("room", "Cleaning")
    elapsed = float(state["time"])
    stats = [str(phase), f'{state["coverage"]:.1f}% floor covered',
             f'{state["cleaned"]}/{state["total"]} dust collected', f'{int(elapsed)//60:02d}:{int(elapsed)%60:02d} mission']
    for i, label in enumerate(stats):
        x = margin + 20 * scale + i * (args.width - margin * 2) / 4
        draw.text((x, bottom - 56 * scale), label, fill="#f4f2e9", font=stat_font if i < 2 else small_font)
    draw.text((args.width - margin - 220 * scale, margin + 8 * scale),
              f"{args.speed:g}x simulation time", fill="#243d33", font=small_font)
    return np.asarray(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))


def render_mission(args):
    if args.width < 320 or args.height < 240 or args.width % 2 or args.height % 2:
        raise SystemExit("Choose even video dimensions of at least 320 by 240.")
    if not 1 <= args.fps <= 60 or not 0 < args.speed <= 100 or not 0 < args.max_time <= 7200:
        raise SystemExit("Use FPS 1..60, speed 0..100, and max-time 0..7200.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sim = Simulator(home_path=args.home, seed=args.seed)
    sim.model.vis.global_.offwidth = args.width
    sim.model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)
    writer = imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", quality=8,
                               macro_block_size=None, ffmpeg_params=["-movflags", "+faststart"])
    frame_num = 0
    state = sim.snapshot()
    print(f"Recording {sim.home['name']} to {args.output}", flush=True)
    try:
        # Establish the house at the dock before cleaning starts.
        initial = _frame(renderer, sim, state, frame_num, args)
        for _ in range(args.fps * 2):
            writer.append_data(initial)
            frame_num += 1
        sim.start()
        last_report = -1
        while float(state["time"]) < args.max_time and state["status"] != "docked":
            duration = args.speed / args.fps
            remaining = duration
            while remaining > 1e-8:
                step = min(0.05, remaining)
                sim.step(step)
                remaining -= step
            state = sim.snapshot()
            writer.append_data(_frame(renderer, sim, state, frame_num, args))
            frame_num += 1
            report = int(float(state["time"]) // 60)
            if report != last_report:
                print(f"  {float(state['time']):.0f}s simulated, {state['coverage']:.1f}% covered, "
                      f"{state['cleaned']}/{state['total']} dust, {state['status']}", flush=True)
                last_report = report
        final = _frame(renderer, sim, state, frame_num, args)
        for _ in range(args.fps * 3):
            writer.append_data(final)
    finally:
        writer.close()
        renderer.close()
    report_path = args.output.with_suffix(".json")
    report = {key: value for key, value in state.items() if key not in ("dust", "trail")}
    report.update(video=str(args.output), fps=args.fps, video_seconds=(frame_num + args.fps * 3) / args.fps,
                  seed=args.seed, engine="MuJoCo", success=state["status"] == "docked" and state["cleaned"] == state["total"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output} and {report_path}", flush=True)
    if not report["success"]:
        raise SystemExit("Mission did not finish within max-time. The partial video and report were saved.")
