FROM ros:humble-ros-base-jammy
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 RMW_IMPLEMENTATION=rmw_fastrtps_cpp MUJOCO_GL=egl CYCLONEDDS_HOME=/usr/local GO2_HTTP_HOST=0.0.0.0
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-dev build-essential cmake git libegl1 libgl1 libgles2 libglib2.0-0 ros-humble-sensor-msgs ros-humble-nav-msgs ros-humble-geometry-msgs ros-humble-rmw-fastrtps-cpp && rm -rf /var/lib/apt/lists/*
# ABI pairing is copied from Wendy's proven Go2 deployment, not distro Cyclone.
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclone && cd /tmp/cyclone && git checkout 2cdd114cbd18340c606573b4cc8dc20cc161ec5a && cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_IDLC=ON && cmake --build build -j2 && cmake --install build && ldconfig && rm -rf /tmp/cyclone
WORKDIR /app
COPY requirements*.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt -r requirements-hardware.txt
RUN pip3 install --no-cache-dir --no-deps git+https://github.com/unitreerobotics/unitree_sdk2_python.git@db9b2d210081387fcd1e7ed9ac4c56a02983bb85
# Pinned upstream packaging omits b2 but imports it at package startup.
# Match the repair in samples/go2; this app never calls B2 APIs.
RUN python3 -c "import importlib.util,pathlib; p=pathlib.Path(importlib.util.find_spec('unitree_sdk2py').origin).parent/'b2'; p.mkdir(exist_ok=True); (p/'__init__.py').touch()"
COPY showcase ./showcase
COPY sim ./sim
EXPOSE 8788
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["python3", "-m", "showcase.server"]
