# Third-party code and assets

The root MIT license covers the collection's original code. The components
listed below retain their own terms. Read the full notices next to the files
and the manifests used by the asset provisioning scripts.

| Location | Component | Included attribution |
| --- | --- | --- |
| `ConstructionDroneMujoco` | Three.js, speech and language models, llama.cpp, and optional vision models | [Demo notices](ConstructionDroneMujoco/THIRD_PARTY.md), `assets.lock.json`, `models.lock.json`, and `vision-models.json` |
| `g1-can-handoff/g1_can_handoff/assets/unitree` | Unitree G1 geometry and meshes | [Source](g1-can-handoff/g1_can_handoff/assets/unitree/SOURCE.md) and [BSD license](g1-can-handoff/g1_can_handoff/assets/unitree/LICENSE) |
| `g1-can-handoff/g1_can_handoff/assets/locomotion` | Unitree RL Lab locomotion policy and conventions used by `walking_policy.py` | [Source manifest](g1-can-handoff/g1_can_handoff/assets/locomotion/SOURCE.json) and [Apache license](g1-can-handoff/g1_can_handoff/assets/locomotion/licenses/unitree_rl_lab.LICENSE) |
| `g1-minigolf/menagerie`, `warehouse-demo/menagerie` | Downloaded MuJoCo Menagerie robot models | Pinned revision in each `run.sh`; upstream and model-specific licenses in each downloaded bundle |
| `go2-assistant/go2_assistant/_physics` | Vendored WendyOS simulator code | `PROVENANCE.json`, `LICENSE`, and `NOTICE` in that directory |
| `go2-teleop/proto`, `go2-teleop/internal/agentpb` | WendyOS Go2 teleop protocol snapshot and generated bindings | [Source](go2-teleop/internal/agentpb/SOURCE.md), `LICENSE`, and `NOTICE` under `internal/agentpb` |
| `go2-showcase/sim/_physics` | Vendored WendyOS simulator code | `PROVENANCE.json`, `LICENSE`, and `NOTICE` in that directory |
| `go2-assistant/assets`, `go2-showcase/sim/assets` | Downloaded Go2 robot geometry, policy, and vision models | Checksummed `assets.lock.json` manifests and fetched license files |
| `go2-assistant/go2_assistant/scene_assets/person` | Converted Cesium Man geometry and textures | [CC BY attribution](go2-assistant/go2_assistant/scene_assets/person/LICENSE.md) |
| `robot-vacuum/static/vendor` | Three.js viewer | [MIT license](robot-vacuum/static/vendor/LICENSE) |
| `sassy-rosmaster/static/vendor` | Three.js viewer | [MIT license](sassy-rosmaster/static/vendor/LICENSE.three) |
| `sassy-rosmaster/models` | YOLOX-Tiny ONNX detector | [Model provenance and hash](sassy-rosmaster/models/README.md) and [Apache license](sassy-rosmaster/models/LICENSE.YOLOX) |
| `sassy-rosmaster/assets/person` | Converted Cesium Man geometry and textures | [CC BY attribution](sassy-rosmaster/assets/person/LICENSE.md) |
| `warehouse-demo/audio` | Recorded arrival bark | [Source, CC0 terms, and edits](warehouse-demo/audio/SOURCE.txt) |

The external Wendy Sandbox runtime required for interactive minigolf and
warehouse playback is not distributed in this repository. Obtain it through
your Wendy installation and set `WENDY_SIM_RUNTIME` as described in those demos.

Python packages and external model services retain their respective licenses.
Downloaded assets are local files; the `.gitignore` does not change their terms.
