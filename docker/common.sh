# Общее для run_bag.sh, run_tool.sh, run_live.sh (подключается через source).

IMAGE=${FOD_IMAGE:-fod}

# Путь к бэгу → абсолютная папка бэга в BAG и её имя в NAME.
# Бэг — папка с metadata.yaml; можно указать и сам файл .db3 / .mcap внутри неё.
resolve_bag() {
    [ -e "$1" ] || { echo "Нет такого пути: $1" >&2; exit 1; }
    BAG=$(realpath "$1")
    [ -f "$BAG" ] && BAG=$(dirname "$BAG")
    [ -f "$BAG/metadata.yaml" ] || { echo "В $BAG нет metadata.yaml — это не бэг ROS 2." >&2; exit 1; }
    NAME=$(basename "$BAG")
}

# Собрать образ, если его ещё нет (один раз, ~10 минут).
ensure_image() {
    docker image inspect "$IMAGE" > /dev/null 2>&1 && return
    echo "Образа $IMAGE нет — собираю (один раз, ~10 минут)…"
    docker build -t "$IMAGE" -f "$HERE/docker/Dockerfile" "$HERE"
}

# Для docker run: результаты — от имени текущего пользователя, не root.
USER_FLAGS=(-u "$(id -u):$(id -g)")

# Открыть адрес в браузере, как только на localhost:$2 кто-то слушает (в фоне).
open_when_ready() {
    local url=$1 port=$2
    (
        for _ in $(seq 600); do
            (echo > "/dev/tcp/127.0.0.1/$port") 2> /dev/null && break
            sleep 1
        done
        sleep 2
        if grep -qi microsoft /proc/version 2> /dev/null; then
            command -v wslview > /dev/null && wslview "$url" || explorer.exe "$url"
        elif command -v xdg-open > /dev/null; then
            xdg-open "$url"
        elif command -v open > /dev/null; then
            open "$url"
        fi
    ) > /dev/null 2>&1 &
}
