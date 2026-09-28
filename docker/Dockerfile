# Обнаружение препятствий на пути по лидару: ROS 2 Humble + torch (CUDA 12.8).
#   docker build -t fod -f docker/Dockerfile .                      # для поезда: узел и инструменты
#   docker build -t fod-viz --target viz -f docker/Dockerfile .     # то же плюс RViz2 (run_live.sh)
#   docker run --gpus all --net host --ipc host fod                 # живой узел
#   docker run --gpus all -v /данные/запись:/data/запись:ro -v $PWD/results:/results \
#       fod python3 -m fod_ros.run_bag /data/запись --out /results  # офлайн-прогон
FROM ros:humble-ros-base AS base

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

WORKDIR /fod
COPY models models
COPY fod fod
COPY fod_ros fod_ros
COPY scripts scripts
COPY config config
COPY docker/live.sh docker/live.sh

# Результаты пишутся от имени пользователя хоста (-u): кэши — не в /root.
ENV HOME=/tmp PYTHONPATH=/fod
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "-m", "fod_ros.node"]

# С RViz2 — для просмотра (run_live.sh); на поезде не нужен.
FROM base AS viz
RUN apt-get update && apt-get install -y --no-install-recommends ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# По умолчанию — образ для поезда, без RViz.
FROM base AS fod
