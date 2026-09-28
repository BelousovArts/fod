#!/usr/bin/env bash
# Прогнать запись ROS 2 через решение в контейнере; результат в results/<имя записи>/.
#
#   ./run_bag.sh /путь/к/записи             # detections.csv, obstacles.csv, summary.json
#   ./run_bag.sh /путь/к/записи --video     # плюс video.mp4
#   ./run_bag.sh /путь/к/записи --until 60  # только первые 60 с
#
# Запись — папка с metadata.yaml или файл .db3 / .mcap внутри неё.
# FOD_RESULTS — куда писать (по умолчанию ./results), FOD_IMAGE — образ (fod),
# FOD_NO_DOCKER=1 — без контейнера, в текущем окружении с ROS 2 и зависимостями.
set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
BAG=$(realpath "$1")
shift
[ -f "$BAG" ] && BAG=$(dirname "$BAG")
[ -f "$BAG/metadata.yaml" ] || { echo "Нет $BAG/metadata.yaml — это не запись ROS 2." >&2; exit 1; }
NAME=$(basename "$BAG")
OUT=$(realpath -m "${FOD_RESULTS:-$PWD/results}")
mkdir -p "$OUT"

if [ -n "${FOD_NO_DOCKER:-}" ]; then
    cd "$(dirname "$(realpath "$0")")"
    exec python3 -m fod_ros.run_bag "$BAG" --out "$OUT" "$@"
fi
exec docker run --rm --gpus all -u "$(id -u):$(id -g)" \
    -v "$BAG:/data/$NAME:ro" -v "$OUT:/results" \
    "${FOD_IMAGE:-fod}" python3 -m fod_ros.run_bag "/data/$NAME" --out /results "$@"
