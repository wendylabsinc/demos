"""Go2 Air web remote: one page on the Jetson, direct WebRTC to the robot.

Hold a button to move, release to stop. The server stops the robot when
commands stop arriving, when the page disconnects, and on Stop.
"""

import asyncio
import hmac
import json
import logging
import os
import time
from pathlib import Path

from aiohttp import WSMsgType, web
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)

log = logging.getLogger("go2-air-remote")

ROBOT_IP = os.environ["GO2_IP"]
AES_KEY = os.environ.get("GO2_AES_KEY") or None
TOKEN = os.environ["GO2_REMOTE_TOKEN"]
PORT = int(os.environ.get("PORT", "8883"))
MAX_X = float(os.environ.get("GO2_MAX_SPEED", "0.3"))  # m/s
MAX_YAW = float(os.environ.get("GO2_MAX_YAW", "0.6"))  # rad/s
WATCHDOG_S = 0.4  # stop if no move command arrives for this long

ACTIONS = {
    "stand": SPORT_CMD["RecoveryStand"],
    "sit": SPORT_CMD["StandDown"],
    "hello": SPORT_CMD["Hello"],
}


def clamp(v, limit):
    return max(-limit, min(limit, float(v)))


class Robot:
    def __init__(self):
        self.conn = None
        self.status = "connecting"
        self.moving = False
        self.last_move = 0.0

    @property
    def ready(self):
        pc = self.conn and self.conn.pc
        return bool(pc and pc.connectionState == "connected")

    async def keep_connected(self):
        while True:
            if not self.ready:
                self.status = "connecting"
                try:
                    if self.conn:
                        await self.conn.disconnect()
                    self.conn = UnitreeWebRTCConnection(
                        WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP, aes_128_key=AES_KEY
                    )
                    await asyncio.wait_for(self.conn.connect(), 20)
                    self.status = "connected"
                    log.info("connected to Go2 at %s", ROBOT_IP)
                except Exception as e:  # busy, unreachable, bad key...
                    self.status = f"not connected: {e}"
                    log.warning("Go2 connection failed: %s", e)
                    await asyncio.sleep(3)
                    continue
            await asyncio.sleep(1)

    def _send(self, api_id, parameter=None):
        if not self.ready:
            return False
        data = {
            "header": {"identity": {"id": int(time.time() * 1000) % 2147483648, "api_id": api_id}},
            "parameter": json.dumps(parameter) if parameter is not None else "",
        }
        self.conn.datachannel.pub_sub.publish_without_callback(
            RTC_TOPIC["SPORT_MOD"], data, "req"
        )
        return True

    def move(self, x, yaw):
        x, yaw = clamp(x, MAX_X), clamp(yaw, MAX_YAW)
        if x == 0 and yaw == 0:
            return self.stop()
        self.moving = True
        self.last_move = time.monotonic()
        return self._send(SPORT_CMD["Move"], {"x": x, "y": 0, "z": yaw})

    def stop(self):
        self.moving = False
        return self._send(SPORT_CMD["StopMove"])

    def action(self, name):
        self.stop()
        return self._send(ACTIONS[name])

    async def watchdog(self):
        while True:
            await asyncio.sleep(0.1)
            if self.moving and time.monotonic() - self.last_move > WATCHDOG_S:
                log.info("watchdog stop")
                self.stop()


robot = Robot()
controller = None  # only the newest page may drive


async def index(_):
    return web.FileResponse(Path(__file__).with_name("index.html"))


async def ws_handler(request):
    global controller
    ws = web.WebSocketResponse(heartbeat=2)
    await ws.prepare(request)

    first = await ws.receive(timeout=10)
    if first.type != WSMsgType.TEXT or not hmac.compare_digest(
        json.loads(first.data).get("token", ""), TOKEN
    ):
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
    app["tasks"] = [
        asyncio.create_task(robot.keep_connected()),
        asyncio.create_task(robot.watchdog()),
    ]


def main():
    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.on_startup.append(start_background)
    web.run_app(app, port=PORT)


if __name__ == "__main__":
    main()
