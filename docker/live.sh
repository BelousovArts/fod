#!/usr/bin/env bash
# Узел детектора + RViz2 (+ проигрывание записи по кругу). Запускается из run_live.sh в
# контейнере fod-viz или прямо в окружении с ROS 2 из корня репозитория.
#   docker/live.sh [ЗАПИСЬ] [--rate 0.5] [--topic /lidar_points]
# Без записи — живой лидар: топик из config/train.yaml или первый найденный PointCloud2.
# Ctrl+C или закрытие RViz — всё остановить.
set -u
cd "$(dirname "$0")/.."

BAG="" TOPIC="" PLAY_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --topic) TOPIC="$2"; shift 2 ;;
        --rate|--start-offset) PLAY_ARGS+=("$1" "$2"); shift 2 ;;
        -*) echo "Неизвестный ключ $1" >&2; exit 1 ;;
        *) BAG="$1"; shift ;;
    esac
done
PARAMS=${FOD_PARAMS:-config/train.yaml}

if [ -n "$BAG" ]; then
    IFS=$'\t' read -r BAG BAG_TOPIC FRAME < <(python3 - "$BAG" <<'E' 2>/dev/null | tail -1
import sys
from fod.bags import peek_frame_id, read_bag_topic, resolve_bag
d = resolve_bag(sys.argv[1]); t = read_bag_topic(d)
print(d, t, peek_frame_id(d, t), sep="\t")
E
)
    [ -n "${BAG:-}" ] || { echo "Не могу прочитать запись" >&2; exit 1; }
    TOPIC=${TOPIC:-$BAG_TOPIC}
    echo "Запись $BAG, топик $TOPIC, система координат $FRAME"
else
    if [ -z "$TOPIC" ]; then
        TOPIC=$(python3 -c "import yaml; print(yaml.safe_load(open('$PARAMS'))['fod']['ros__parameters'].get('topic') or '')")
    fi
    echo "Живой лидар, ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}. Жду облако…"
    while [ -z "$TOPIC" ]; do
        TOPIC=$(ros2 topic list -t 2>/dev/null | awk '/sensor_msgs\/msg\/PointCloud2/ && $1 !~ /^\/fod/ {print $1; exit}')
        [ -n "$TOPIC" ] || sleep 2
    done
    FRAME=$(ros2 topic echo --once "$TOPIC" --field header.frame_id 2>/dev/null | head -1)
    echo "Топик $TOPIC, система координат $FRAME"
fi

RVIZ_CFG=$(mktemp --suffix=.rviz)
sed -e "s#Fixed Frame: hesai_lidar#Fixed Frame: ${FRAME:-hesai_lidar}#" -e "s#Value: /lidar_points#Value: $TOPIC#" config/fod.rviz > "$RVIZ_CFG"

python3 -m fod_ros.node --ros-args --params-file "$PARAMS" -p topic:="$TOPIC" &
NODE=$!
PLAY=""
cleanup() {
    kill -INT $NODE ${PLAY:+$PLAY} 2>/dev/null
    sleep 2
    kill $NODE ${PLAY:+$PLAY} 2>/dev/null
    rm -f "$RVIZ_CFG"
}
trap cleanup EXIT INT TERM
if [ -n "$BAG" ]; then
    # Пока грузятся сети, запись не пускаем: первые кадры иначе уйдут в пустоту.
    sleep "${FOD_WARMUP:-10}"
    ros2 bag play "$BAG" --loop "${PLAY_ARGS[@]}" > /dev/null 2>&1 &
    PLAY=$!
fi
rviz2 -d "$RVIZ_CFG" > /dev/null 2>&1
