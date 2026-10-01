"""Verify Go2 geometry/pose publication using Wendy's actual Sim bridge module."""
import argparse
import json
import os
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--module-dir', default='/opt/sandbox/sim')
p.add_argument('--output', default='results/wendy-bridge')
a = p.parse_args()
module_dir = Path(a.module_dir)
if not (module_dir / 'wendymujoco.py').is_file():
    p.error('Provide the Wendy Sandbox directory that contains wendymujoco.py.')
output = Path(a.output).resolve()
output.mkdir(parents=True, exist_ok=True)
os.environ['WENDY_WORLDSIM_DIR'] = str(output)
sys.path.insert(0, str(module_dir.resolve()))
import wendymujoco
from go2_assistant.sim import SimBackend
backend = SimBackend()
try:
    with wendymujoco.launch_passive(backend.sim.model, backend.sim.data,
                                  title='Wendy Go2 assistant', rerun=False) as viewer:
        viewer.hud(task='bridge verification', state='idle')
        backend.tick(.1)
        viewer.sync()
        scene = json.loads((output / 'scene.json').read_text())
        state = json.loads((output / 'state.json').read_text())
        assert scene['engine'] == 'mujoco' and scene['meshes']
        assert len(state['pose']) == backend.sim.model.ngeom
        assert state['hud']['state'] == 'idle'
        print(json.dumps({'passed': True, 'visible_geometries': len(scene['geoms']),
                          'meshes': len(scene['meshes']), 'pose_count': len(state['pose']),
                          'output': str(output)}, indent=2))
finally:
    backend.close()
