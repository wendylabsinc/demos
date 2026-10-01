# Contributing

Make changes in the demo directory they belong to. Use a separate Python
environment for each demo and follow its README for dependency installation,
asset preparation, and runtime checks.

Run the root repository checks before opening a pull request:

```sh
python3.12 scripts/check_repository.py
git diff --check
```

The checker reads Git's tracked and unignored files. It validates Python,
JSON, TOML, and shell syntax, rejects nested repositories, and limits individual
files to 50 MiB. It does not import the demos or download their dependencies.

## Controller tests

The lightweight CI tests use Python 3.12 with NumPy 1.26.4 and pytest 8:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install 'numpy==1.26.4' 'pytest>=8,<9' 'scipy>=1.15,<2'
```

Run the Go2 vision agent's tests from its directory:

```sh
../.venv/bin/python -m unittest discover -s tests -p test_ai.py -v
../.venv/bin/python -m unittest discover -s tests -p test_action_executor.py -v
../.venv/bin/python -m unittest discover -s tests -p test_transport.py -v
```

Run the ROSMASTER's tests from `sassy-rosmaster`:

```sh
../.venv/bin/python -m unittest discover -s tests -p test_hardware.py -v
../.venv/bin/python -m unittest discover -s tests -p test_navigation.py -v
../.venv/bin/python -m unittest discover -s tests -p test_speech.py -v
../.venv/bin/python -m unittest discover -s tests -p test_perception_ros.py -v
```

Run the can handoff's fault tests from `g1-can-handoff`:

```sh
../.venv/bin/python -m unittest discover -s tests -p test_faults.py -v
```

Run localization's quality tests from the repository root:

```sh
.venv/bin/python -m pytest warehouse-localization/tests/test_quality.py -q
.venv/bin/python -m pytest go2-assistant/tests/test_core.py go2-assistant/tests/test_hardware.py go2-assistant/tests/test_mapping.py -q
.venv/bin/python -m pytest ConstructionDroneMujoco/tests/test_perception.py -q
```

For the vacuum's full mission tests and package build, use its locked environment:

```sh
cd robot-vacuum
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv build
```

Other demo READMEs describe physics, rendering, model, and ROS integration
checks. Report which checks you ran and any runtime dependencies you could
not exercise. Hardware tests with fake messages do not establish physical
robot behavior.

## Assets and documentation

Keep marketing materials, recordings, model caches, and generated results out
of commits. Add a pinned download manifest and a verification command for new
large downloadable assets. Include the source, license, and any modifications
for vendored code and assets. Preserve `.obj` robot meshes as source assets.

Use relative paths in documentation and environment variables for optional
external checkouts. Describe required services and hardware limitations in the
demo README. Do not put credentials or personal device settings in examples.
