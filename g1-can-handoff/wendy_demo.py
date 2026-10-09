"""Run inside Wendy Sandbox using: wendy-simrun wendy_demo.py"""

import os
import time
from g1_can_handoff.sim import scene
from g1_can_handoff.perception import SimCamera
from g1_can_handoff.task import execute
import wendymujoco

p = scene(
    float(os.environ.get("TABLE_HEIGHT", ".75")),
    float(os.environ.get("CAN_X", ".35")),
    float(os.environ.get("CAN_Y", ".18")),
)
p.step(0.3)
camera = SimCamera(p)
with wendymujoco.launch_passive(p.model, p.data, title="G1 Dex3 can pickup") as viewer:
    last = [-1.0]

    def sync(plant):
        if plant.data.time - last[0] > 0.03:
            last[0] = plant.data.time
            viewer.sync()

    p.frame_callback = sync
    try:
        observation, _ = camera.observe()
        execute(
            p,
            observation,
            lambda phase: viewer.hud(
                phase=phase, table_height=round(observation.table_height, 3)
            ),
            lambda: camera.observe()[0],
        )
        while viewer.is_running():
            p.step(0.02)
            time.sleep(0.02)
    except ValueError as error:
        viewer.hud(phase="stopped", reason=str(error))
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.03)
    finally:
        camera.close()
