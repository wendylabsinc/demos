# Wendy demos

Robot simulations and example applications for WendyOS. Each demo has its own
dependencies and run instructions. Several include a browser viewer, local
inference, or a Wendy deployment configuration.

## Demos

| Demo | What it does | Requirements beyond Python |
| --- | --- | --- |
| [Autonomous city taxi](autonomous-taxi/README.md) | Drives a MuJoCo taxi through city traffic with passenger trips, LiDAR and six cameras | uv; Wendy native app for the 3D view |
| [Robot vacuum](robot-vacuum/README.md) | Cleans a furnished MuJoCo home and returns to its dock | uv; Wendy CLI and QEMU for VM deployment |
| [Construction yard drone](ConstructionDroneMujoco/README.md) | Surveys a simulated yard, compares camera evidence, and follows a tagged worker | Provisioned speech and language models; container builder for Wendy deployment |
| [G1 can handoff](g1-can-handoff/README.md) | Grasps a can with a simulated Dex3 hand and waits for supervised release | uv; bundled G1 geometry and policy |
| [G1 minigolf](g1-minigolf/README.md) | Balances a G1 and strikes a golf ball through physical contact | Git for pinned Menagerie assets; Wendy Sandbox runtime for interactive playback |
| [Go2 assistant](go2-assistant/README.md) | Maps measured observations and accepts room tasks through voice or text | Downloaded robot, policy, and vision assets |
| [Go2 vision agent](go2-showcase/README.md) | Lets a vision-language model choose bounded Go2 actions | Ollama or a compatible vision inference endpoint; downloaded simulator assets |
| [Sassy ROSMASTER](sassy-rosmaster/README.md) | Explores a simulated room and responds to camera-observed people | uv and Ollama |
| [Warehouse camera search](warehouse-demo/README.md) | Finds a ball through fixed-camera detection and guides a simulated Go2 | Wendy Sandbox runtime, Menagerie assets, and a separate localization service |
| [Warehouse thermal fire watch](warehouse-fire/README.md) | Locates heat through 88 thermal cameras and activates local sprinklers in a 40 × 40 m MuJoCo twin | Supported browser for Web Push; Wendy native app for the 3D view |
| [Warehouse localization](warehouse-localization/README.md) | Runs ROS 2 mapping or AMCL localization with observation checks | ROS 2 Jazzy in its container; Wendy CLI or Docker |

Physical robot adapters have their own commissioning requirements and limits.
Read the selected demo's hardware instructions before using it with a robot.
Simulation results describe the simulated system.

## Start with the vacuum simulation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from this repository:

```sh
cd robot-vacuum
./scripts/run-local.sh
```

Open <http://127.0.0.1:18881> and select **Start cleaning**. This demo runs
locally without a robot or model service. Its README also covers WendyOS VM
deployment and recording.

For another demo, enter its directory and follow its README. Keep the Python
environments separate because the projects use different MuJoCo and NumPy
versions. The repository checks use Python 3.11 or newer; individual demos
specify their own Python requirements.

Install the [Wendy CLI](https://docs.wendy.dev/latest/) for Wendy deployments.
Run each deployment from the directory containing that demo's `wendy.json`,
and select the intended device explicitly.

## Assets and local files

Downloaded bundles with provisioning manifests stay outside Git. Follow the
demo's asset preparation instructions before launching it or building its
container. Required bundled assets retain their source and license notices.

The root `.gitignore` excludes marketing and promo folders, recordings,
archives, virtual environments, caches, build output, and generated simulation
state. These files can remain on your machine. Keep credentials in environment
variables or ignored `.env` files; use the checked-in examples for hardware
configuration.

## Checks and contributions

Run the dependency-free repository checks from the root:

```sh
python3.12 scripts/check_repository.py
```

GitHub Actions runs these checks, selected controller and transport tests, and
the vacuum's mission tests and package build. It does not deploy devices or
run the model-dependent and ROS integration scenarios. See
[CONTRIBUTING.md](CONTRIBUTING.md) for local test commands and changes to assets.

## License

Original code in this collection uses the [MIT license](LICENSE). Vendored
code, robot geometry, learned policies, models, and media retain their own
licenses. See [THIRD_PARTY.md](THIRD_PARTY.md) and the notices next to those
files.
