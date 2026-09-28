#!/usr/bin/env python3
"""Живой узел ROS 2: облако лидара → статус пути и препятствия.

Подписка — на топик PointCloud2 (параметр `topic`; пусто — первый найденный
PointCloud2). Колбэк только кладёт сообщение в ячейку последнего кадра
(`fod/latest.py`), тракт крутится в своём потоке и всегда берёт самый свежий
кадр: если он не успевает за лидаром, старые кадры пропускаются, а отставание
не копится.

Публикует:
- `/fod/status` (std_msgs/String, JSON): статус UNKNOWN / CLEAR / OBSTACLE,
  расстояние до ближайшего препятствия, список препятствий и подозрений, до куда
  известна ось, время обработки, задержка, сколько кадров пропущено; `telemetry` —
  уровень (с ATTENTION), скорость, рекомендуемая скорость, тормозной путь
  (`fod/telemetry.py`);
- `/fod/obstacles` (sensor_msgs/PointCloud2) — для систем поезда: по точке на объект
  (подтверждённые и подозрения), в системе координат облака и с его временем. Поля:
  `x, y, z` (центр, м), `distance` (вдоль пути, м), `offset` (от оси, + влево, м),
  `height` (над полотном, м), `track_id`, `confirmed` (1 — препятствие, 0 — подозрение),
  `first_seen` (с какого расстояния замечен, м). Пустое облако — путь свободен
  (или тракт ещё не готов — см. `/fod/status`);
- `/fod/markers` (visualization_msgs/MarkerArray): ось, границы габарита,
  препятствия (красные) и подозрения (оранжевые), табличка с уровнем и скоростями —
  для RViz, в системе координат облака.

  python3 -m fod_ros.node --ros-args -p topic:=/lidar_points -p device:=cuda
  python3 -m fod_ros.node --ros-args --params-file config/train.yaml

Параметры: `topic`, `device` (cuda / cpu), `markers` (публиковать маркеры),
`best_effort` (подписка BEST_EFFORT — для драйвера, который публикует только так);
установка лидара (`fod/mount.py`): `forward` (`-y`, `+x`, …), `up` (`+z` / `-z`),
`rpy` (поправки крен, тангаж, рыскание, градусы). Координаты в `/fod/status` —
тоже в системе облака.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rclpy  # noqa: E402
from builtin_interfaces.msg import Duration  # noqa: E402
from geometry_msgs.msg import Point  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402
from std_msgs.msg import ColorRGBA, String  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from fod.latest import Latest  # noqa: E402
from fod.mount import Mount  # noqa: E402
from fod.telemetry import telemetry  # noqa: E402

CLOUD_TYPE = "sensor_msgs/msg/PointCloud2"
COLORS = {
    "UNKNOWN": ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.9),
    "CLEAR": ColorRGBA(r=0.1, g=0.9, b=0.2, a=0.9),
    "ATTENTION": ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9),
    "OBSTACLE": ColorRGBA(r=1.0, g=0.1, b=0.1, a=0.9),
}
OBSTACLE_FIELDS = ("x", "y", "z", "distance", "offset", "height", "track_id", "confirmed", "first_seen")


class FodNode(Node):
    def __init__(self) -> None:
        super().__init__("fod")
        self.declare_parameter("topic", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("markers", True)
        self.declare_parameter("best_effort", False)
        self.declare_parameter("forward", "-y")
        self.declare_parameter("up", "+z")
        self.declare_parameter("rpy", [0.0, 0.0, 0.0])
        self.topic = str(self.get_parameter("topic").value)
        self.markers = bool(self.get_parameter("markers").value)
        self.best_effort = bool(self.get_parameter("best_effort").value)
        device = str(self.get_parameter("device").value)
        rpy = [float(v) for v in self.get_parameter("rpy").value]
        self.mount = Mount(forward=str(self.get_parameter("forward").value), up=str(self.get_parameter("up").value),
                           rpy=tuple(rpy))
        qx, qy, qz, qw = self.mount.quaternion()
        self.box_q = (qx, qy, qz, qw)

        self.pub_status = self.create_publisher(String, "/fod/status", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/fod/markers", 10)
        self.pub_obstacles = self.create_publisher(PointCloud2, "/fod/obstacles", 10)
        self.slot: Latest = Latest()
        self.index = 0
        self.sub = None
        self.stats = {"done": 0, "ms": [], "lat": []}

        self.get_logger().info(f"Установка лидара: {self.mount.describe()}. Загрузка сетей на {device}…")
        from fod.pipeline import Pipeline

        self.pipeline = Pipeline(device=device, mount=self.mount)
        self.worker = threading.Thread(target=self._work, name="fod", daemon=True)
        self.worker.start()
        if self.topic:
            self._subscribe(self.topic)
        else:
            self.find_timer = self.create_timer(1.0, self._find_topic)
            self.get_logger().info("Жду топик PointCloud2…")
        self.create_timer(5.0, self._report)

    # --- вход -----------------------------------------------------------------

    def _find_topic(self) -> None:
        for name, types in self.get_topic_names_and_types():
            if CLOUD_TYPE in types and not name.startswith("/fod"):
                self.find_timer.cancel()
                self._subscribe(name)
                return

    def _subscribe(self, topic: str) -> None:
        # RELIABLE: облако — десятки МБ, при BEST_EFFORT потеря одного фрагмента теряет весь кадр.
        # Глубина 1: старые кадры всё равно не нужны.
        reliability = ReliabilityPolicy.BEST_EFFORT if self.best_effort else ReliabilityPolicy.RELIABLE
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=reliability)
        self.sub = self.create_subscription(PointCloud2, topic, self._on_cloud, qos)
        self.topic = topic
        self.get_logger().info(f"Подписка на {topic}")

    def _on_cloud(self, msg: PointCloud2) -> None:
        self.slot.put((self.index, msg, time.perf_counter()))
        self.index += 1

    # --- обработка ------------------------------------------------------------

    def _work(self) -> None:
        try:
            for res in self.pipeline.run(self.slot, live=True):
                self.stats["done"] += 1
                self.stats["ms"].append(res.ms)
                self.stats["lat"].append(res.latency_ms)
                self._publish(res)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Тракт остановился: {exc!r}")
            raise

    def _publish(self, res) -> None:
        tel = telemetry(res)
        body = res.as_dict()
        body["telemetry"] = tel.as_dict()
        body["received"] = self.slot.received
        body["dropped"] = self.slot.dropped
        for key in ("obstacles", "suspects"):
            for o in body[key]:
                o["x"], o["y"], o["z"] = (round(float(v), 3) for v in self.mount.to_sensor([[o["x"], o["y"], o["z"]]])[0])
        self.pub_status.publish(String(data=json.dumps(body, ensure_ascii=False)))
        self.pub_obstacles.publish(self._obstacle_cloud(res))
        if self.markers:
            self.pub_markers.publish(self._marker_array(res, tel))

    def _obstacle_cloud(self, res) -> PointCloud2:
        items = [(o, 1.0) for o in res.obstacles] + [(o, 0.0) for o in res.suspects]
        data = np.zeros((len(items), len(OBSTACLE_FIELDS)), dtype=np.float32)
        for k, (o, confirmed) in enumerate(items):
            data[k, :3] = self.mount.to_sensor([[o.x, o.y, o.z]])[0]
            data[k, 3:] = (o.distance, o.offset, o.height, o.track_id, confirmed, o.first_seen)
        msg = PointCloud2()
        msg.header.frame_id, msg.header.stamp = res.frame_id, _stamp(res.stamp)
        msg.height, msg.width = 1, len(items)
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(OBSTACLE_FIELDS)]
        msg.is_bigendian, msg.is_dense = False, True
        msg.point_step = 4 * len(OBSTACLE_FIELDS)
        msg.row_step = msg.point_step * msg.width
        msg.data = data.tobytes()
        return msg

    def _points(self, xyz) -> list[Point]:
        q = self.mount.to_sensor(np.asarray(xyz, dtype=np.float64).reshape(-1, 3))
        return [Point(x=float(a), y=float(b), z=float(c)) for a, b, c in q]

    def _marker_array(self, res, tel) -> MarkerArray:
        stamp = _stamp(res.stamp)
        life = Duration(sec=0, nanosec=500_000_000)
        out = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        out.markers.append(clear)

        def marker(mid: int, kind: int, ns: str) -> Marker:
            m = Marker()
            m.header.frame_id, m.header.stamp = res.frame_id, stamp
            m.ns, m.id, m.type, m.action, m.lifetime = ns, mid, kind, Marker.ADD, life
            m.pose.orientation.w = 1.0
            return m

        color = COLORS.get(tel.level, COLORS["UNKNOWN"])
        if res.axis_n.size and res.reach > 0.0:
            keep = res.axis_s <= res.reach
            s, n = res.axis_s[keep], res.axis_n[keep]
            z = res.axis_z[keep] if res.axis_z.size else np.full(s.size, -1.2)
            for mid, off, ns in ((0, 0.0, "axis"), (1, -res.half_width, "gauge"), (2, res.half_width, "gauge")):
                m = marker(mid, Marker.LINE_STRIP, ns)
                m.scale.x = 0.05 if mid == 0 else 0.08
                m.color = color
                m.points = self._points(np.stack([n + off, -s, z], axis=1))
                out.markers.append(m)
        boxes = [(o, "obstacles", COLORS["OBSTACLE"], "") for o in res.obstacles]
        boxes += [(o, "suspects", COLORS["ATTENTION"], "?") for o in res.suspects]
        for k, (o, ns, c, mark) in enumerate(boxes):
            m = marker(100 + k, Marker.CUBE, ns)
            m.pose.position = self._points([o.x, o.y, o.z])[0]
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = self.box_q
            m.scale.x, m.scale.y, m.scale.z = 0.6, 0.6, max(o.height, 0.2)
            m.color = c
            out.markers.append(m)
            t = marker(300 + k, Marker.TEXT_VIEW_FACING, ns)
            t.pose.position = self._points([o.x, o.y, o.z + 0.5 * max(o.height, 0.2) + 0.5])[0]
            t.scale.z = 0.8
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text = f"{o.distance:.0f} m{mark}"
            if not mark and np.isfinite(o.first_seen):
                t.text += f" (seen {o.first_seen:.0f} m)"
            out.markers.append(t)
        board = marker(1, Marker.TEXT_VIEW_FACING, "telemetry")
        board.pose.position = self._points([0.0, -12.0, 3.0])[0]
        board.scale.z = 1.0
        board.color = color
        board.text = _board(tel)
        out.markers.append(board)
        return out

    def _report(self) -> None:
        ms, lat = self.stats["ms"], self.stats["lat"]
        if not ms:
            return
        self.get_logger().info(
            f"{self.topic}: принято {self.slot.received}, обработано {self.stats['done']}, пропущено {self.slot.dropped}; "
            f"за 5 с: {np.median(ms):.0f} мс на кадр, задержка {np.median(lat):.0f} мс"
        )
        self.stats["ms"], self.stats["lat"] = [], []

    def close(self) -> None:
        self.slot.close()
        self.worker.join(timeout=5.0)


def _stamp(t: float):
    sec = int(t)
    return Time(seconds=sec, nanoseconds=int(round((t - sec) * 1e9))).to_msg()


def _board(tel) -> str:
    head = tel.level
    if tel.level == "OBSTACLE" and np.isfinite(tel.distance):
        head += f" {tel.distance:.0f} m"
    elif tel.level == "ATTENTION" and np.isfinite(tel.suspect):
        head += f" {tel.suspect:.0f} m?"
    if not np.isfinite(tel.safe_kmh):
        return head
    over = " !" if tel.overspeed else ""
    return (f"{head}\nspeed {tel.speed_kmh:.0f} km/h{over}  recommended {tel.safe_kmh:.0f} km/h\n"
            f"gauge checked to {tel.reach:.0f} m  stopping {tel.stopping_m:.0f} m")


def main() -> int:
    rclpy.init()
    node = FodNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
