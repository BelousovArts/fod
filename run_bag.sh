#!/usr/bin/env bash
# Прогнать бэг через детектор; результат — в results/<имя бэга>/.
#
#   ./run_bag.sh /путь/к/бэгу               # video.mp4, detections.csv, obstacles.csv, summary.json
#   ./run_bag.sh /путь/к/бэгу --no-video    # без видео (быстрее)
#   ./run_bag.sh /путь/к/бэгу --until 60    # только первые 60 секунд
#
# Бэг — папка с metadata.yaml (или файл .db3 / .mcap внутри неё).
# FOD_RESULTS — куда писать (по умолчанию ./results), FOD_NO_DOCKER=1 — без контейнера.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/docker/common.sh"

if [ $# -lt 1 ] || [ "$1" = -h ] || [ "$1" = --help ]; then
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
resolve_bag "$1"
shift
OUT=$(realpath -m "${FOD_RESULTS:-$PWD/results}")
mkdir -p "$OUT"

ARGS=() VIDEO=(--video)
for a in "$@"; do
    if [ "$a" = --no-video ]; then VIDEO=(); else ARGS+=("$a"); fi
done
ARGS+=("${VIDEO[@]}")

if [ -n "${FOD_NO_DOCKER:-}" ]; then
    cd "$HERE"
    exec python3 -m fod_ros.run_bag "$BAG" --out "$OUT" "${ARGS[@]}"
fi
ensure_image
exec docker run --rm --gpus all "${USER_FLAGS[@]}" \
    -v "$BAG:/data/$NAME:ro" -v "$OUT:/results" \
    "$IMAGE" python3 -m fod_ros.run_bag "/data/$NAME" --out /results "${ARGS[@]}"
