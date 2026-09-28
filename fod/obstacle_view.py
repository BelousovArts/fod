"""Кадр детектора препятствий: сверху range image переднего сектора, снизу вид сверху.

Вид сверху — облако в координатах сенсора, масштаб по обеим осям одинаковый,
поезд едет слева направо, до `BEV_S` м. Поперёк ±`BEV_HALF` м: на 150 м путь
уходит вбок на s²/2R — у кривой R = 300 м это ~38 м, влезает без искажений.
"""

from __future__ import annotations

import cv2
import numpy as np

from fod.cloud import MIN_RANGE
from fod.obstacles import gauge_floor
from fod.colormaps import colorize_intensity
from fod.rail_template_view import rail_tracks
from fod.rails_view import project_to_image

WIDTH = 1280
TITLE_H = 30
FRONT_H = 260
BEV_S = 150.0
BEV_BACK = 3.0
BEV_PX = WIDTH / (BEV_S + BEV_BACK)  # пикселей на метр по обеим осям, ~8.4
BEV_HALF = 40.0
BG = 20

COL_TEXT = (235, 235, 235)
COL_DIM = (140, 140, 140)
COL_GAUGE = (230, 230, 230)
COL_AXIS = (90, 220, 120)
COL_RAIL = (0, 230, 255)
COL_CR = (255, 140, 30)
COL_WALL = (70, 90, 150)
COL_CAND = (0, 165, 255)
COL_SUSPECT = (0, 130, 255)
COL_CONFIRMED = (60, 60, 255)
COL_OBJECT = (255, 80, 255)


def render_obstacle_frame(
    xyz, intensity, rail, front, det, detector, obj=None, *, title: str, cr=None, axis=None, head=None
) -> np.ndarray:
    """`front` — RangeImage переднего сектора, `obj` — (name, s, n) вставленного объекта или None.

    `cr` — ContactRailFrame (линии КР), `axis` / `head` — ось детектора `n(s)` и высота
    головок `z(s)`, если не ось кадра рельсов и не подгонка детектора.
    """
    if axis is None or not det.ready:
        axis = _axis(det, detector, rail)
    if head is None and detector.head_coeff is not None:
        head = detector.head_z
    s_gauge = det.s_max if np.isfinite(det.s_max) else detector.cfg.s_max
    top = _front(front, rail, det, axis, head, detector, obj, cr, s_gauge)
    bev = _bev(xyz, intensity, rail, det, axis, head, detector, obj, cr, s_gauge)
    out = np.full((TITLE_H + top.shape[0] + bev.shape[0], WIDTH, 3), BG, np.uint8)
    out[TITLE_H : TITLE_H + top.shape[0]] = top
    out[TITLE_H + top.shape[0] :] = bev
    color = COL_CONFIRMED if det.status == "OBSTACLE" else COL_AXIS if det.status == "CLEAR" else COL_DIM
    status = det.status + (f" {det.distance:.1f} m" if det.status == "OBSTACLE" else "")
    _put(out, title, (10, 20), 0.52, COL_TEXT)
    _put(out, status, (WIDTH - 230, 21), 0.62, color, 2)
    return out


def _axis(det, detector, rail):
    """Ось как функция `n(s)`; None, если фильтр не заперт."""
    if rail.axis_s is None or rail.axis_n is None or rail.axis_s.size < 2 or not det.ready:
        return None
    s, n = np.asarray(rail.axis_s, dtype=np.float64), np.asarray(rail.axis_n, dtype=np.float64)
    return lambda q: np.interp(q, s, n)


def _track_box(track, det, axis) -> tuple[float, float, float]:
    s_now = track.s_now(det.s_odom)
    n_abs = track.n + (float(axis(max(s_now, 0.0))) if axis is not None else 0.0)
    return s_now, n_abs, track.height


# --- range image ------------------------------------------------------------


def _found_tracks(rail):
    """Головки по фильтру только до последней найденной пары: дальше — экстраполяция кубики."""
    tracks = rail_tracks(rail)
    if tracks is None or not rail.marks:
        return None
    s_hi = max(float(m.s) for m in rail.marks)
    tracks = tuple(c[c[:, 0] <= s_hi] for c in tracks)
    return tracks if all(len(c) >= 2 for c in tracks) else None


def _front(front, rail, det, axis, head, detector, obj, cr, s_gauge) -> np.ndarray:
    vis = colorize_intensity(front.intensity)
    h, w = vis.shape[:2]
    sx = WIDTH / w
    sy = FRONT_H / h
    vis = cv2.resize(vis, (WIDTH, FRONT_H), interpolation=cv2.INTER_NEAREST)

    def proj(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols, rows = project_to_image(front, xyz, 1)
        return (cols + 0.5) * sx, (rows + 0.5) * sy

    def poly(xyz: np.ndarray, color, thickness: int = 2) -> None:
        cols, rows = proj(xyz)
        ok = np.isfinite(cols) & np.isfinite(rows)
        for a in range(len(cols) - 1):
            if ok[a] and ok[a + 1]:
                p0, p1 = (int(cols[a]), int(rows[a])), (int(cols[a + 1]), int(rows[a + 1]))
                cv2.line(vis, p0, p1, color, thickness, cv2.LINE_AA)

    tracks = _found_tracks(rail) if head is not None else None
    if tracks is not None:
        for curve in tracks:
            ok = curve[:, 0] > 2.0
            if int(ok.sum()) >= 2:
                c = curve[ok]
                poly(np.stack([c[:, 1], -c[:, 0], head(c[:, 0])], axis=1), COL_RAIL, 2)
    if cr is not None:
        for line in cr.lines.values():
            poly(np.stack([line[:, 1], -line[:, 0], line[:, 2]], axis=1), COL_CR, 2)
    if axis is not None and head is not None:
        cfg = detector.cfg
        s = np.linspace(cfg.s_min, s_gauge, 120)
        n = axis(s)
        z0 = head(s)
        for side in (-cfg.half_width, cfg.half_width):
            poly(np.stack([n + side, -s, z0 + cfg.floor_u], axis=1), COL_GAUGE, 1)
            poly(np.stack([n + side, -s, z0 + cfg.height_max], axis=1), COL_GAUGE, 1)
        for side in (-cfg.notch_half, cfg.notch_half):
            poly(np.stack([n + side, -s, z0 + cfg.notch_top], axis=1), COL_GAUGE, 1)
        n_rel = np.linspace(-cfg.half_width, cfg.half_width, 41)
        floor = gauge_floor(cfg, n_rel)
        poly(np.stack([n[-1] + n_rel, np.full_like(n_rel, -s_gauge), z0[-1] + floor], axis=1), COL_GAUGE, 1)
        poly(np.array([[n[-1] - cfg.half_width, -s_gauge, z0[-1] + cfg.height_max],
                       [n[-1] + cfg.half_width, -s_gauge, z0[-1] + cfg.height_max]]), COL_GAUGE, 1)
    if det.cand_s.size:
        cols, rows = proj(np.stack([det.cand_n, -det.cand_s, det.cand_z], axis=1))
        for c, r in zip(cols, rows):
            if np.isfinite(c) and np.isfinite(r):
                cv2.circle(vis, (int(c), int(r)), 2, COL_CAND, -1)
    if head is not None:
        for track in det.tracks:
            s_now, n_abs, height = _track_box(track, det, axis)
            if s_now < 1.0:
                continue
            z0 = float(head(s_now))
            half = max(0.3, 0.5 * detector.cfg.cell_n)
            corners = np.array([[n_abs + half, -s_now, z0 + height], [n_abs - half, -s_now, z0 - 0.05]])
            cols, rows = proj(corners)
            if not np.all(np.isfinite(cols) & np.isfinite(rows)):
                continue
            color = COL_CONFIRMED if track.confirmed else COL_SUSPECT
            p0 = (int(min(cols)) - 3, int(min(rows)) - 3)
            p1 = (int(max(cols)) + 3, int(max(rows)) + 3)
            cv2.rectangle(vis, p0, p1, color, 2 if track.confirmed else 1)
            if track.confirmed:
                _put(vis, f"{s_now:.1f}m", (p1[0] + 4, p0[1] + 12), 0.45, color)
        if obj is not None:
            name, s_o, n_o = obj
            cols, rows = proj(np.array([[n_o, -s_o, float(head(s_o))]]))
            if np.isfinite(cols[0]) and np.isfinite(rows[0]):
                cv2.circle(vis, (int(cols[0]), int(rows[0])), 14, COL_OBJECT, 1, cv2.LINE_AA)
    _put(vis, "FRONT VIEW: lidar range image, colour = return intensity", (8, 16), 0.42, COL_TEXT)
    x = 8
    for text, color in (("running rails", COL_RAIL), ("contact (third) rail", COL_CR),
                        ("gauge 2.1 x 3.0 m, notch between rails", COL_GAUGE)):
        cv2.line(vis, (x, 30), (x + 18, 30), color, 2)
        _put(vis, text, (x + 24, 34), 0.42, color)
        x += 34 + int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0])
    return vis


# --- вид сверху -----------------------------------------------------------------


def _bev(xyz, intensity, rail, det, axis, head, detector, obj, cr, s_gauge) -> np.ndarray:
    h = int(round(2.0 * BEV_HALF * BEV_PX))
    cy = h / 2.0

    def to_px(s, n) -> tuple[np.ndarray, np.ndarray]:
        # s > 0 — вперёд (вправо), n > 0 — влево от поезда (вверх).
        x = (np.asarray(s, dtype=np.float64) + BEV_BACK) * BEV_PX
        y = cy - np.asarray(n, dtype=np.float64) * BEV_PX
        return x, y

    def pts(s, n) -> np.ndarray:
        x, y = to_px(s, n)
        return np.round(np.stack([x, y], axis=1) * 4).astype(np.int32)

    img = np.full((h, WIDTH, 3), 12, np.uint8)
    for sv in range(0, int(BEV_S) + 1, 10):
        x = int(to_px(sv, 0.0)[0])
        cv2.line(img, (x, 0), (x, h - 1), (38, 38, 38) if sv % 50 else (60, 60, 60), 1)
    s = -xyz[:, 1]
    rng = (s > -BEV_BACK) & (s < BEV_S) & (np.einsum("ij,ij->i", xyz, xyz) > MIN_RANGE * MIN_RANGE)
    s, n = s[rng], xyz[rng, 0]
    if head is not None:
        ref = head(np.clip(s, 0.0, None))
    else:
        ref = np.full(s.shape, rail.z_floor if rail.z_floor is not None else -2.0)
    u = xyz[rng, 2] - ref
    keep = (np.abs(n) < BEV_HALF) & (u > -0.8) & (u < 4.5)
    if np.any(keep):
        xs, ys = to_px(s[keep], n[keep])
        xs, ys = xs.astype(np.int32), ys.astype(np.int32)
        ok = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < h)
        xs, ys, uu = xs[ok], ys[ok], u[keep][ok]
        ii = intensity[rng][keep].astype(np.float64)[ok]
        shade = np.clip(50.0 + 205.0 * np.clip(ii / 60.0, 0.0, 1.0) ** 0.6, 0, 255).astype(np.uint8)
        bed = uu < 0.6
        img[ys[~bed], xs[~bed]] = COL_WALL
        img[ys[bed], xs[bed]] = np.stack([shade[bed]] * 3, axis=1)

    def line(sv, nv, color, thickness=1) -> None:
        cv2.polylines(img, [pts(sv, nv)], False, color, thickness, cv2.LINE_AA, shift=2)

    tracks = _found_tracks(rail)
    if tracks is not None:
        for curve in tracks:
            line(curve[:, 0], curve[:, 1], COL_RAIL)
    if cr is not None:
        for cl in cr.lines.values():
            line(cl[:, 0], cl[:, 1], COL_CR, 2)
    if axis is not None:
        cfg = detector.cfg
        sg = np.linspace(cfg.s_min, s_gauge, 200)
        ng = np.asarray(axis(sg), dtype=np.float64)
        # Габарит перпендикулярно оси, а не вдоль n: на кривой иначе сужается.
        th = np.arctan(np.gradient(ng, sg))
        for side in (-cfg.half_width, cfg.half_width, -cfg.notch_half, cfg.notch_half):
            line(sg - side * np.sin(th), ng + side * np.cos(th), COL_GAUGE, 1)
        for i in range(0, len(sg) - 1, 4):
            line(sg[i : i + 2], ng[i : i + 2], COL_AXIS, 1)
        end = np.array([-1.0, 1.0]) * cfg.half_width
        line(sg[-1] - end * np.sin(th[-1]), ng[-1] + end * np.cos(th[-1]), COL_GAUGE, 1)
    if det.cand_s.size:
        x, y = to_px(det.cand_s, det.cand_n)
        for xi, yi in zip(x, y):
            cv2.circle(img, (int(xi), int(yi)), 2, COL_CAND, -1)
    for track in det.tracks:
        s_now, n_abs, height = _track_box(track, det, axis)
        x, y = to_px(s_now, n_abs)
        color = COL_CONFIRMED if track.confirmed else COL_SUSPECT
        r = 9 if track.confirmed else 6
        cv2.rectangle(img, (int(x) - r, int(y) - r), (int(x) + r, int(y) + r), color, 2 if track.confirmed else 1)
        if track.confirmed:
            _put(img, f"{s_now:.1f} m, h {height:.2f} m", (int(x) - 50, int(y) - r - 6), 0.45, color)
    if obj is not None:
        name, s_o, n_o = obj
        x, y = to_px(s_o, n_o)
        cv2.circle(img, (int(x), int(y)), 14, COL_OBJECT, 1, cv2.LINE_AA)
        _put(img, f"test object: {name} {s_o:.1f} m", (int(x) - 60, int(y) + 32), 0.45, COL_OBJECT)
    x0, y0 = to_px(0.0, 0.0)
    tri = np.array([[x0 + 10, y0], [x0 - 6, y0 - 7], [x0 - 6, y0 + 7]], np.int32)
    cv2.fillPoly(img, [tri], COL_TEXT)

    for sv in range(10, int(BEV_S) + 1, 10):
        x = int(to_px(sv, 0.0)[0])
        _put(img, f"{sv}", (x - 10, h - 6), 0.4, COL_DIM)
    for nv in range(-30, 31, 10):
        _put(img, f"{nv:+d}", (4, int(cy - nv * BEV_PX) + 4), 0.38, COL_DIM)
    x10 = int(WIDTH - 20 - 10 * BEV_PX)
    cv2.line(img, (x10, 14), (WIDTH - 20, 14), COL_TEXT, 2)
    _put(img, "10 m", (x10, 30), 0.4, COL_TEXT)
    _put(img, "TOP VIEW, 1:1 scale, train moves right; distance along track and offset in m", (40, 16), 0.45, COL_DIM)
    _legend(img, h, obj is not None)
    return img


LEGEND = (
    ("line", COL_RAIL, "running rails (found by the detector)"),
    ("line", COL_CR, "contact (third) rail"),
    ("line", COL_GAUGE, "clearance gauge 2.1 x 3.0 m; inner lines - floor raised between the rails"),
    ("dash", COL_AXIS, "track centreline"),
    ("dot", COL_CAND, "candidate points: above the normal track bed, inside the gauge"),
    ("box", COL_SUSPECT, "suspect: seen, not confirmed yet"),
    ("box", COL_CONFIRMED, "obstacle: confirmed (4 of 5 frames)"),
)


def _legend(img, h: int, with_object: bool) -> None:
    rows = LEGEND + ((("circle", COL_OBJECT, "test object inserted by the stand"),) if with_object else ())
    x0, y = 40, h - 24 - 20 * len(rows)
    cv2.rectangle(img, (x0 - 10, y - 18), (x0 + 700, h - 26), (28, 28, 28), -1)
    for kind, color, text in rows:
        c = (x0 + 10, y - 4)
        if kind == "line":
            cv2.line(img, (x0, c[1]), (x0 + 22, c[1]), color, 2)
        elif kind == "dash":
            for k in range(0, 22, 8):
                cv2.line(img, (x0 + k, c[1]), (x0 + k + 4, c[1]), color, 1)
        elif kind == "dot":
            cv2.circle(img, c, 3, color, -1)
        elif kind == "box":
            cv2.rectangle(img, (c[0] - 7, c[1] - 7), (c[0] + 7, c[1] + 7), color, 2)
        else:
            cv2.circle(img, c, 8, color, 1, cv2.LINE_AA)
        _put(img, text, (x0 + 32, y), 0.42, color if color != COL_GAUGE else COL_TEXT)
        y += 20


def _put(img, text, org, scale=0.4, color=COL_TEXT, thickness=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
