FROM ros:jazzy-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-nav2-amcl ros-jazzy-nav2-map-server ros-jazzy-nav2-lifecycle-manager \
    ros-jazzy-slam-toolbox ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-sensor-msgs-py python3-yaml python3-numpy python3-pytest \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
ENV PYTHONUNBUFFERED=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
EXPOSE 8790
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "launch", "/app/localization.launch.py"]
