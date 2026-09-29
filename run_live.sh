#!/usr/bin/env bash
# Детектор в живом режиме с RViz2.
#
#   ./run_live.sh /путь/к/бэгу               # бэг по кругу → детектор → RViz
#   ./run_live.sh /путь/к/бэгу --rate 0.5    # вдвое медленнее
#   ./run_live.sh                             # настоящий лидар в сети
#   ./run_live.sh --topic /lidar_points       # настоящий лидар, если облаков в сети несколько
#
# Нужен экран: Linux с графикой или Windows 11 (WSL2). Параметры узла — config/train.yaml.
# ROS_DOMAIN_ID — домен лидара в сети, FOD_NO_DOCKER=1 — без контейнера.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/docker/common.sh"

BAG="" ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --topic|--rate|--start-offset) ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) resolve_bag "$1"; shift ;;
    esac
done

if [ -n "${FOD_NO_DOCKER:-}" ]; then
    exec bash "$HERE/docker/live.sh" ${BAG:+"$BAG"} "${ARGS[@]}"
fi

[ -n "${DISPLAY:-}" ] || { echo "Нет DISPLAY: RViz некуда показать." >&2; exit 1; }
ensure_image
command -v xhost > /dev/null && xhost +local: > /dev/null 2>&1 || true
RUN=(docker run --rm --gpus all "${USER_FLAGS[@]}"
     -e DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix
     -v "$HERE/config/train.yaml:/fod/config/train.yaml:ro"
     -v "$HERE/config/detector.yaml:/fod/config/detector.yaml:ro")
[ -t 0 ] && RUN+=(-it)
[ -d /mnt/wslg ] && RUN+=(-v /mnt/wslg:/mnt/wslg -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR -e PULSE_SERVER)
if [ -n "$BAG" ]; then
    # Бэг играется внутри контейнера, в своей сети: чужим узлам в сети не мешает.
    RUN+=(-v "$BAG:/data/$NAME:ro")
    ARGS=("/data/$NAME" "${ARGS[@]}")
else
    RUN+=(--net host --ipc host -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}")
fi
exec "${RUN[@]}" "$IMAGE" bash /fod/docker/live.sh "${ARGS[@]}"
