"""Go2 Air web remote: one page on the Jetson, robot commands via the Wendy agent.

The agent owns the paired robot connection and its key. Hold a button to move,
release to stop. The server stops the robot when commands stop arriving, when
the page disconnects, and on Stop; the agent also stops a move after 500 ms
unless it is refreshed.
"""

import asyncio
import hmac
import json
import logging
import os
import time
from pathlib import Path

import grpc
from aiohttp import WSMsgType, web

import unitree_service_pb2 as pairing_pb
import unitree_service_pb2_grpc as pairing_grpc
import unitree_teleop_service_pb2 as teleop_pb
import unitree_teleop_service_pb2_grpc as teleop_grpc

log = logging.getLogger("go2-air-remote")

SERIAL = os.environ["GO2_SERIAL"]
AGENT_SOCKET = os.environ["WENDY_AGENT_SOCKET"]  # set by the admin entitlement
TOKEN = os.environ["GO2_REMOTE_TOKEN"]
PORT = int(os.environ.get("PORT", "8883"))
MAX_X = float(os.environ.get("GO2_MAX_SPEED", "0.3"))  # m/s, agent limit 0.3
MAX_YAW = float(os.environ.get("GO2_MAX_YAW", "0.6"))  # rad/s, agent limit 0.6
WATCHDOG_S = 0.4  # stop if no move command arrives for this long

ACTIONS = {
    "stand": teleop_pb.UNITREE_COMMAND_STAND_UP,
    "sit": teleop_pb.UNITREE_COMMAND_LIE_DOWN,
    "hello": teleop_pb.UNITREE_COMMAND_HELLO,
}


def clamp(v, limit):
    return max(-limit, min(limit, float(v)))


class Robot:
    def __init__(self):
        self.status = "connecting"
        self.moving = False
        self.last_move = 0.0

    async def start(self):
        channel = grpc.aio.insecure_channel(f"unix:{AGENT_SOCKET}")
        self.pairing = pairing_grpc.WendyUnitreeServiceStub(channel)
        self.teleop = teleop_grpc.WendyUnitreeTeleopServiceStub(channel)

    async def watch_status(self):
        while True:
            try:
                reply = await self.pairing.ListUnitreePairings(
                    pairing_pb.ListUnitreePairingsRequest(), timeout=2
                )
                robot = next((r for r in reply.robots if r.serial == SERIAL), None)
                if robot is None:
                    self.status = f"not paired with {SERIAL}"
                elif robot.connected:
                    self.status = "connected"
                else:
                    self.status = "agent is reconnecting to the robot"
            except grpc.aio.AioRpcError as e:
                self.status = f"agent unavailable: {e.details()}"
            await asyncio.sleep(1)

    async def _send(self, command, x=0.0, yaw=0.0):
        try:
            await self.teleop.Command(
                teleop_pb.UnitreeCommandRequest(serial=SERIAL, command=command, forward=x, yaw=yaw),
                timeout=1,
            )
            return True
        except grpc.aio.AioRpcError as e:
            log.warning("command %s failed: %s", command, e.details())
            return False

    async def move(self, x, yaw):
        x, yaw = clamp(x, MAX_X), clamp(yaw, MAX_YAW)
        if x == 0 and yaw == 0:
            return await self.stop()
        self.moving = True
        self.last_move = time.monotonic()
        return await self._send(teleop_pb.UNITREE_COMMAND_MOVE, x, yaw)

    async def stop(self):
        self.moving = False
        return await self._send(teleop_pb.UNITREE_COMMAND_STOP)

    async def action(self, name):
        self.moving = False
        return await self._send(ACTIONS[name])

    async def watchdog(self):
        while True:
            await asyncio.sleep(0.1)
            if self.moving and time.monotonic() - self.last_move > WATCHDOG_S:
                log.info("watchdog stop")
                await self.stop()


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
        await robot.stop()
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
                await robot.move(*m["move"])
            elif m.get("action") in ACTIONS:
                await robot.action(m["action"])
            else:
                await robot.stop()
    finally:
        reporter.cancel()
        await robot.stop()
        if controller is ws:
            controller = None
    return ws


async def start_background(app):
    await robot.start()
    app["tasks"] = [
        asyncio.create_task(robot.watch_status()),
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
