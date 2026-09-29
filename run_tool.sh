#!/usr/bin/env bash
# Инструменты по бэгу; результат — в results/<имя бэга>/.
#
#   ./run_tool.sh rerun /путь/к/бэгу             # 3D-просмотр в браузере: http://localhost:9090
#   ./run_tool.sh video /путь/к/бэгу             # только видео (то же, что run_bag.sh)
#   ./run_tool.sh map   /путь/к/бэгу             # карта: облако map.ply, вид сверху, плитки по 100 м
#   ./run_tool.sh stand /путь/к/бэгу --object person@80 --video   # вставить объект и проверить детектор
#   ./run_tool.sh stand /путь/к/бэгу --scenario мой_сценарий.yaml
#   ./run_tool.sh stand --list                   # какие объекты можно вставить
#
# Общие ключи: --start 20 --until 80 (кусок бэга, секунды); rerun --save файл.rrd — в файл вместо браузера.
# FOD_RESULTS — куда писать (./results), FOD_NO_DOCKER=1 — без контейнера.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
source "$HERE/docker/common.sh"

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
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

BAG="" NAME=""
if ! { [ "$TOOL" = stand ] && [ "${1:-}" = --list ]; }; then
    [ $# -ge 1 ] || usage
    resolve_bag "$1"
    shift
fi

# rerun без --save — в браузер.
WEB=""
if [ "$TOOL" = rerun ] && [[ " $* " != *" --save "* ]]; then
    WEB=1
fi
WEB_URL="http://localhost:9090/?url=ws://localhost:9877"
SAVE_NAME=""

if [ -n "${FOD_NO_DOCKER:-}" ]; then
    cd "$HERE"
    ARGS=()
    [ -n "$BAG" ] && ARGS+=("$BAG")
    [ "$TOOL" = rerun ] || { [ -n "$BAG" ] && ARGS+=(--out "$OUT"); }
    ARGS+=("$@")
    [ -n "$WEB" ] && { ARGS+=(--web); open_when_ready "$WEB_URL" 9090; }
    exec python3 "$SCRIPT" "${ARGS[@]}"
fi

ensure_image
RUN=(docker run --rm --init --gpus all "${USER_FLAGS[@]}" -v "$OUT:/results")
[ -t 0 ] && RUN+=(-it)
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
        --save) SAVE_NAME=$(basename "$2"); ARGS+=(--save "/results/$NAME/$SAVE_NAME"); shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
if [ -z "$BAG" ]; then
    exec "${RUN[@]}" "$IMAGE" python3 "$SCRIPT" "${ARGS[@]}"
fi
if [ -n "$WEB" ]; then
    RUN+=(-p 9090:9090 -p 9877:9877)
    ARGS+=(--web)
    echo "Просмотр откроется в браузере: $WEB_URL (Ctrl+C — закончить)"
    open_when_ready "$WEB_URL" 9090
    exec "${RUN[@]}" "$IMAGE" python3 "$SCRIPT" "/data/$NAME" "${ARGS[@]}"
fi
if [ "$TOOL" = rerun ]; then
    "${RUN[@]}" "$IMAGE" python3 "$SCRIPT" "/data/$NAME" "${ARGS[@]}"
    echo "Файл: $OUT/$NAME/$SAVE_NAME (открыть: rerun $OUT/$NAME/$SAVE_NAME, нужен pip install rerun-sdk==0.22.1)"
    exit 0
fi
exec "${RUN[@]}" "$IMAGE" python3 "$SCRIPT" "/data/$NAME" --out /results "${ARGS[@]}"
