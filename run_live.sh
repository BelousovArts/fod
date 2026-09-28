#!/usr/bin/env bash
# Детектор в живом режиме с RViz2 — в контейнере fod-viz.
#
#   ./run_live.sh /путь/к/записи               # запись по кругу → узел → RViz
#   ./run_live.sh /путь/к/записи --rate 0.5    # вдвое медленнее
#   ./run_live.sh                               # живой лидар в сети (топик — config/train.yaml или первый PointCloud2)
#   ./run_live.sh --topic /lidar_points         # живой лидар, топик явно
#
# Установка лидара и прочие параметры узла — config/train.yaml (берётся с хоста).
# Нужен X-сервер (Linux, или WSLg в Windows 11). Образ: docker build -t fod-viz --target viz -f docker/Dockerfile .
# FOD_VIZ_IMAGE — образ (fod-viz), ROS_DOMAIN_ID — домен живого лидара, FOD_NO_DOCKER=1 — без контейнера.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")

BAG="" ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --topic|--rate|--start-offset) ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) BAG=$(realpath "$1"); shift ;;
    esac
done
if [ -n "$BAG" ]; then
    [ -f "$BAG" ] && BAG=$(dirname "$BAG")
    [ -f "$BAG/metadata.yaml" ] || { echo "Нет $BAG/metadata.yaml — это не запись ROS 2." >&2; exit 1; }
fi

if [ -n "${FOD_NO_DOCKER:-}" ]; then
    exec bash "$HERE/docker/live.sh" ${BAG:+"$BAG"} "${ARGS[@]}"
fi

[ -n "${DISPLAY:-}" ] || { echo "Нет DISPLAY: RViz некуда показать." >&2; exit 1; }
command -v xhost > /dev/null && xhost +local: > /dev/null 2>&1 || true
RUN=(docker run --rm --gpus all -u "$(id -u):$(id -g)"
     -e DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix
     -v "$HERE/config/train.yaml:/fod/config/train.yaml:ro")
[ -t 0 ] && RUN+=(-it)
[ -d /mnt/wslg ] && RUN+=(-v /mnt/wslg:/mnt/wslg -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR -e PULSE_SERVER)
if [ -n "$BAG" ]; then
    # Запись играется внутри контейнера, в своей сети: чужим узлам в сети не мешает.
    NAME=$(basename "$BAG")
    RUN+=(-v "$BAG:/data/$NAME:ro")
    ARGS=("/data/$NAME" "${ARGS[@]}")
else
    RUN+=(--net host --ipc host -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}")
fi
exec "${RUN[@]}" "${FOD_VIZ_IMAGE:-fod-viz}" bash /fod/docker/live.sh "${ARGS[@]}"
