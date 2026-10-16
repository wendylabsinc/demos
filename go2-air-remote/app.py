"""Go2 web remote: one page on the Jetson, a plain ROS 2 app.

Publishes /cmd_vel and /api/sport/request and shows /camera/color/image_raw,
the same contract as the WendyOS Go2 simulator and the Go2 ROS 2 bridge, so it
drives either. Hold a button to move, release to stop. The app sends zero
velocity when commands stop arriving, when the page disconnects, and on Stop.
"""

import asyncio
import hmac
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import rclpy
from aiohttp import WSMsgType, web
from geometry_msgs.msg import Twist
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from unitree_api.msg import Request
from unitree_go.msg import LowState

log = logging.getLogger("go2-air-remote")

TOKEN = os.environ["GO2_REMOTE_TOKEN"]
PORT = int(os.environ.get("PORT", "8883"))
# The physical Go2 ignores forward requests below about 0.5 m/s.
MAX_X = float(os.environ.get("GO2_MAX_SPEED", "0.55"))  # m/s
MAX_YAW = float(os.environ.get("GO2_MAX_YAW", "0.6"))  # rad/s
WATCHDOG_S = 0.4  # stop if no move command arrives for this long
VIDEO_WIDTH = 640

# Unitree sport API ids.
ACTIONS = {"stand": 1006, "sit": 1005, "hello": 1016}


def clamp(v, limit):
    return max(-limit, min(limit, float(v)))


class Robot(Node):
    def __init__(self):
        super().__init__("go2_air_remote")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.sport = self.create_publisher(Request, "/api/sport/request", 10)
        self.create_subscription(LowState, "/lowstate", self.on_lowstate, qos)
        self.image_sub = None
        self.last_state = 0.0
        self.jpeg = None
        self.jpeg_time = 0.0
        self.viewers = 0
        self.moving = False
        self.last_move = 0.0
        self.request_id = 0

    @property
    def status(self):
        return "connected" if time.monotonic() - self.last_state < 2 else "no robot data on /lowstate"

    def on_lowstate(self, _):
        self.last_state = time.monotonic()

    def watch_video(self, watching):
        """Subscribe to the camera only while someone watches, so the bridge
        decodes video only then."""
        self.viewers += 1 if watching else -1
        if self.viewers > 0 and self.image_sub is None:
            self.image_sub = self.create_subscription(Image, "/camera/color/image_raw", self.on_image, 1)
        elif self.viewers == 0 and self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            self.image_sub = None

    def on_image(self, msg):
        if msg.encoding != "rgb8" or time.monotonic() - self.jpeg_time < 0.1:
            return  # at most 10 frames per second
        image = PILImage.frombuffer("RGB", (msg.width, msg.height), bytes(msg.data), "raw", "RGB", msg.step, 1)
        if msg.width > VIDEO_WIDTH:
            image = image.resize((VIDEO_WIDTH, msg.height * VIDEO_WIDTH // msg.width))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=70)
        self.jpeg, self.jpeg_time = out.getvalue(), time.monotonic()

    def move(self, x, yaw):
        x, yaw = clamp(x, MAX_X), clamp(yaw, MAX_YAW)
        if x == 0 and yaw == 0:
            return self.stop()
        self.moving = True
        self.last_move = time.monotonic()
        twist = Twist()
        twist.linear.x, twist.angular.z = x, yaw
        self.cmd_vel.publish(twist)

    def stop(self):
        self.moving = False
        self.cmd_vel.publish(Twist())

    def action(self, name):
        self.stop()
        self.request_id += 1
        request = Request()
        request.header.identity.id = self.request_id
        request.header.identity.api_id = ACTIONS[name]
        self.sport.publish(request)

    async def watchdog(self):
        while True:
            await asyncio.sleep(0.1)
            if self.moving and time.monotonic() - self.last_move > WATCHDOG_S:
                log.info("watchdog stop")
                self.stop()


robot = None
controller = None  # only the newest page may drive


def authorized(key):
    return hmac.compare_digest(key or "", TOKEN)


async def index(_):
    return web.FileResponse(Path(__file__).with_name("index.html"))


async def video(request):
    if not authorized(request.query.get("key")):
        raise web.HTTPForbidden()
    response = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                                           "Cache-Control": "no-store"})
    await response.prepare(request)
    robot.watch_video(True)
    sent = None
    try:
        while True:
            jpeg = robot.jpeg
            if jpeg is not None and jpeg is not sent:
                await response.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                sent = jpeg
            await asyncio.sleep(0.05)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        robot.watch_video(False)
    return response


async def ws_handler(request):
    global controller
    ws = web.WebSocketResponse(heartbeat=2)
    await ws.prepare(request)

    first = await ws.receive(timeout=10)
    if first.type != WSMsgType.TEXT or not authorized(json.loads(first.data).get("token")):
        await ws.send_json({"error": "wrong key"})
        await ws.close()
        return ws

    if controller is not None:
        robot.stop()
        await controller.close()
    controller = ws

    async def report():
        while not ws.closed:
            await ws.send_json({"status": robot.status, "moving": robot.moving})
            await asyncio.sleep(0.5)

    reporter = asyncio.create_task(report())
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                break
            m = json.loads(msg.data)
            if "move" in m:
                robot.move(*m["move"])
            elif m.get("action") in ACTIONS:
                robot.action(m["action"])
            else:
                robot.stop()
    finally:
        reporter.cancel()
        robot.stop()
        if controller is ws:
            controller = None
    return ws


async def start_background(app):
    app["tasks"] = [asyncio.create_task(robot.watchdog())]


def main():
    global robot
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    robot = Robot()
    threading.Thread(target=rclpy.spin, args=(robot,), daemon=True).start()
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/video.mjpg", video)
    app.on_startup.append(start_background)
    try:
        web.run_app(app, port=PORT)
    finally:
        robot.stop()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
