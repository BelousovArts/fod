#!/usr/bin/env bash
# Инструменты по записи в контейнере; результат в results/<имя записи>/.
#
#   ./run_tool.sh video /путь/к/записи [--start 20 --until 80]    # video.mp4 + CSV
#   ./run_tool.sh rerun /путь/к/записи [--until 60]                # rerun.rrd → rerun results/<запись>/rerun.rrd
#   ./run_tool.sh map   /путь/к/записи [--voxel 0.1]               # map/map.ply, map.pcd, map_top.png, tiles/
#   ./run_tool.sh stand /путь/к/записи --object person@80 [--video]
#   ./run_tool.sh stand /путь/к/записи --scenario мой_сценарий.yaml
#   ./run_tool.sh stand --list                                     # типы объектов
#
# У всех: --forward/--up/--rpy — установка лидара (по умолчанию как в записях хакатона).
# FOD_RESULTS — куда писать (./results), FOD_IMAGE — образ (fod), FOD_NO_DOCKER=1 — без контейнера.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[ $# -ge 1 ] || usage
case "$1" in
    video) SCRIPT=scripts/make_video.py ;;
    rerun) SCRIPT=scripts/rerun_bag.py ;;
    map) SCRIPT=scripts/make_map.py ;;
    stand) SCRIPT=scripts/stand.py ;;
    *) usage ;;
esac
TOOL=$1
shift
OUT=$(realpath -m "${FOD_RESULTS:-$PWD/results}")
mkdir -p "$OUT"

if [ "$TOOL" = stand ] && [ "${1:-}" = --list ]; then
    BAG="" NAME=""
else
    [ $# -ge 1 ] || usage
    BAG=$(realpath "$1")
    shift
    [ -f "$BAG" ] && BAG=$(dirname "$BAG")
    [ -f "$BAG/metadata.yaml" ] || { echo "Нет $BAG/metadata.yaml — это не запись ROS 2." >&2; exit 1; }
    NAME=$(basename "$BAG")
fi

if [ -n "${FOD_NO_DOCKER:-}" ]; then
    cd "$HERE"
    ARGS=(${BAG:+"$BAG"} --out "$OUT" "$@")
    [ "$TOOL" = rerun ] && [[ " $* " != *" --save "* ]] && ARGS+=(--save "$OUT/$NAME/rerun.rrd")
    [ "$TOOL" = stand ] && [ -z "$BAG" ] && ARGS=("$@")
    exec python3 "$SCRIPT" "${ARGS[@]}"
fi

RUN=(docker run --rm --gpus all -u "$(id -u):$(id -g)" -v "$OUT:/results")
[ -n "$BAG" ] && RUN+=(-v "$BAG:/data/$NAME:ro")
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --scenario)
            # Файл сценария с хоста — в контейнер.
            SC=$(realpath "$2")
            RUN+=(-v "$SC:/scenario/$(basename "$SC"):ro")
            ARGS+=(--scenario "/scenario/$(basename "$SC")")
            shift 2 ;;
        --save) ARGS+=(--save "/results/$NAME/$(basename "$2")"); shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
if [ -z "$BAG" ]; then
    exec "${RUN[@]}" "${FOD_IMAGE:-fod}" python3 "$SCRIPT" "${ARGS[@]}"
fi
[ "$TOOL" = rerun ] && [[ " ${ARGS[*]} " != *" --save "* ]] && ARGS+=(--save "/results/$NAME/rerun.rrd")
exec "${RUN[@]}" "${FOD_IMAGE:-fod}" python3 "$SCRIPT" "/data/$NAME" --out /results "${ARGS[@]}"
