# Обнаружение препятствий на пути по лидару: ROS 2 Humble + torch (CUDA 12.8) + RViz2 + rerun.
# Один образ для всего: прогон бэгов, инструменты, живой узел на поезде и просмотр в RViz.
#   docker build -t fod -f docker/Dockerfile .
# Скрипты run_bag.sh / run_tool.sh / run_live.sh собирают его сами, если образа ещё нет.
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip ffmpeg \
        ros-humble-rosbag2-py ros-humble-rosbag2-storage-default-plugins ros-humble-rosbag2-storage-mcap \
        ros-humble-sensor-msgs ros-humble-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# RViz2 — отдельным слоем после pip: слои с torch остаются в кэше сборки.
RUN apt-get update && apt-get install -y --no-install-recommends ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /fod
COPY models models
COPY fod fod
COPY fod_ros fod_ros
COPY scripts scripts
COPY config config
COPY docker/live.sh docker/live.sh

# Результаты пишутся от имени пользователя хоста (-u): кэши — не в /root.
ENV HOME=/tmp PYTHONPATH=/fod
# Веб-просмотрщик rerun (run_tool.sh rerun): страница и поток данных.
EXPOSE 9090 9877
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "fod_ros.node"]
