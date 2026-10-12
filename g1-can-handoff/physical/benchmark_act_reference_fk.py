"""Compare ACT scratch FK/Jacobian stages without opening hardware IO."""
import json
import time
import numpy as np
import mujoco
from g1_can_handoff.sim import scene


def main():
    world = scene()
    model = world.model
    full, reduced = mujoco.MjData(model), mujoco.MjData(model)
    positions = []
    rng = np.random.default_rng(819714)
    bounds = model.jnt_range[world.joints[world.arm]]
    for _ in range(200):
        qpos = world.data.qpos.copy()
        qpos[world.qadr[world.arm]] = rng.uniform(bounds[:, 0], bounds[:, 1])
        positions.append(qpos)
    jac = [np.zeros((3, model.nv)), np.zeros((3, model.nv))]
    times = {'full_forward': [], 'kinematics_com': []}
    for turn in range(20):
        for qpos in positions:
            for name, data in [('full_forward', full), ('kinematics_com', reduced)][::(-1 if turn % 2 else 1)]:
                data.qpos[:] = qpos
                start = time.perf_counter_ns()
                if name == 'full_forward':
                    mujoco.mj_forward(model, data)
                else:
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_comPos(model, data)
                mujoco.mj_jacSite(model, data, *jac, world.site)
                times[name].append((time.perf_counter_ns() - start) / 1000)
    result = {name: {'samples': len(values), 'median_us': float(np.median(values)),
                     'p95_us': float(np.percentile(values, 95))}
              for name, values in times.items()}
    result['median_speedup'] = result['full_forward']['median_us'] / result['kinematics_com']['median_us']
    result['hardware_io'] = False
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
