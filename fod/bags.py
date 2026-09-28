"""Поиск и разбор ROS 2 bag с облаком лидара."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# Папка с записями (подпапки с metadata.yaml); FOD_DATA — если данные лежат в другом месте.
DATA_ROOT = Path(os.environ.get("FOD_DATA", ROOT / "Исходные данные" / "датасет" / "archive"))

BAG_FRAME_HINTS = {
    "doubleT_obstacle": "lidar_livox",
}


def find_bags() -> dict[str, Path]:
    bags: dict[str, Path] = {}
    for root in (DATA_ROOT / "for_hackathon", DATA_ROOT):
        if not root.is_dir():
            continue
        for metadata in sorted(root.glob("*/metadata.yaml")):
            bags[metadata.parent.name] = metadata.parent
    return bags


def resolve_bag(query: str | None) -> Path:
    bags = find_bags()
    if query is None:
        if "doubleT_platform" in bags:
            return bags["doubleT_platform"]
        if bags:
            return next(iter(bags.values()))
        raise SystemExit(f"ROS 2 bag не найдены в {DATA_ROOT}")

    candidate = Path(query).expanduser()
    if candidate.is_file() and candidate.suffix in (".db3", ".mcap"):
        candidate = candidate.parent
    if candidate.is_dir() and (candidate / "metadata.yaml").is_file():
        return candidate.resolve()
    if query in bags:
        return bags[query]
    known = ", ".join(bags) or "(пусто)"
    raise SystemExit(f"Неизвестный bag «{query}». Доступны: {known}")


def read_bag_topic(bag_dir: Path) -> str:
    metadata = yaml.safe_load((bag_dir / "metadata.yaml").read_text(encoding="utf-8"))
    topics = metadata["rosbag2_bagfile_information"]["topics_with_message_count"]
    for item in topics:
        topic_type = item["topic_metadata"]["type"]
        if topic_type.endswith("PointCloud2"):
            return item["topic_metadata"]["name"]
    raise SystemExit(f"В {bag_dir.name} нет топика PointCloud2")


def storage_id(bag_dir: Path) -> str:
    """Формат записи из metadata.yaml: sqlite3 или mcap."""
    info = yaml.safe_load((bag_dir / "metadata.yaml").read_text(encoding="utf-8"))["rosbag2_bagfile_information"]
    return str(info.get("storage_identifier") or "sqlite3")


def peek_frame_id(bag_dir: Path, topic: str) -> str:
    hint = BAG_FRAME_HINTS.get(bag_dir.name)
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        return hint or "hesai_lidar"

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id(bag_dir)),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    msg_type = get_message(topics[topic])
    while reader.has_next():
        name, data, _timestamp = reader.read_next()
        if name != topic:
            continue
        message = deserialize_message(data, msg_type)
        frame_id = message.header.frame_id or hint or "hesai_lidar"
        del reader
        return frame_id
    return hint or "hesai_lidar"


def bag_time_range(bag_dir: Path) -> tuple[int, int]:
    """Начало и конец записи, нс."""
    info = yaml.safe_load((bag_dir / "metadata.yaml").read_text(encoding="utf-8"))["rosbag2_bagfile_information"]
    start = int(info["starting_time"]["nanoseconds_since_epoch"])
    return start, start + int(info["duration"]["nanoseconds"])


def iter_pointclouds(
    bag_dir: Path,
    topic: str | None = None,
    start_ns: int | None = None,
    stop_ns: int | None = None,
):
    """Итератор PointCloud2 из ROS 2 bag. Нужны rosbag2_py и sensor_msgs.

    `start_ns` / `stop_ns` — кусок записи по времени бэга; индекс считается от начала куска.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    topic = topic or read_bag_topic(bag_dir)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id(bag_dir)),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic not in topics:
        raise SystemExit(f"В {bag_dir.name} нет топика {topic}")
    msg_type = get_message(topics[topic])
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    if start_ns is not None:
        reader.seek(int(start_ns))
    index = 0
    try:
        while reader.has_next():
            name, data, timestamp_ns = reader.read_next()
            if name != topic:
                continue
            if stop_ns is not None and timestamp_ns >= stop_ns:
                break
            message = deserialize_message(data, msg_type)
            yield index, timestamp_ns, message
            index += 1
    finally:
        del reader


def list_bags() -> None:
    bags = find_bags()
    if not bags:
        print(f"Bag-файлы не найдены в {DATA_ROOT}")
        return
    print("Доступные записи:")
    for name, path in bags.items():
        metadata = yaml.safe_load((path / "metadata.yaml").read_text(encoding="utf-8"))
        info = metadata["rosbag2_bagfile_information"]
        topic = read_bag_topic(path)
        duration_s = info["duration"]["nanoseconds"] / 1e9
        count = info["message_count"]
        print(f"  {name:42}  {topic:40}  {count:5} кадр.  {duration_s:7.1f} с")
