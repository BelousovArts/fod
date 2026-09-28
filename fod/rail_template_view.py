"""Кадр шаблонного детектора: BEV коридора и передний сектор.

Пары рисуются по нормали к полиному оси. Сама ось — пунктир этого полинома,
а не ломаная через центры пар. Масштаб по s и n одинаковый, иначе поворот
шаблона на картинке схлопывается.
"""

from __future__ import annotations

import cv2
import numpy as np

from fod.cloud import MIN_RANGE
from fod.colormaps import colorize_intensity
from fod.rail_template import S_MAX as DETECT_S
from fod.rail_template import RailFrame, RailMark

from fod.rails_view import project_to_image
from fod.range_image import RangeImage

VIEW_HALF = 12.0
# Детектор ищет пары только до DETECT_S. Кадр показывает облако дальше, до 150 м.
VIEW_S = 150.0
# Одинаковый масштаб по s и n, иначе шаблон, перпендикулярный оси в метрах,
# на картинке выглядит почти горизонтальным: поперечная ось была растянута.
PX_PER_M = 16.0
BEV_W = int(round(2.0 * VIEW_HALF * PX_PER_M))
BEV_H = int(round(VIEW_S * PX_PER_M))
PANEL_W = 460
TITLE_H = 34
AXIS_LEFT = 44
AXIS_BOTTOM = 26
BG = 20

COL_TEXT = (228, 228, 228)
COL_DIM = (150, 150, 150)
COL_WALL = (120, 70, 45)
COL_AXIS = (90, 220, 120)
COL_RAIL = (40, 210, 255)


def render_frame(
    xyz: np.ndarray,
    intensity: np.ndarray,
    detected: RailFrame,
    *,
    title: str,
    front: RangeImage | None = None,
    speed_kmh: float | None = None,
) -> np.ndarray:
    bev = _bev_block(xyz, intensity, detected)
    row = _hstack([bev, _panel(detected, bev.shape[0], speed_kmh)])
    if front is not None:
        top = _front_block(front, detected)
        top = cv2.resize(top, (row.shape[1], max(int(round(top.shape[0] * row.shape[1] / top.shape[1])), 2)))
        row = _vstack([top, row])
    out = np.full((TITLE_H + row.shape[0], row.shape[1], 3), BG, np.uint8)
    out[TITLE_H:] = row
    _put(out, title, (AXIS_LEFT, 22), 0.52, (245, 245, 245))
    return out


def _bev_block(xyz: np.ndarray, intensity: np.ndarray, detected: RailFrame) -> np.ndarray:
    bev = _bev(xyz, intensity, detected)
    left = [(float(_px_s(s)), f"{s}") for s in range(0, int(VIEW_S) + 1, 10)]
    bottom = [(float(_px_n(n)), f"{n:+d}") for n in range(-12, 13, 4)]
    return _with_axes(bev, left, bottom, "s, m", "n, m")


def _bev(xyz: np.ndarray, intensity: np.ndarray, detected: RailFrame) -> np.ndarray:
    img = np.full((BEV_H, BEV_W, 3), 12, np.uint8)
    s = -xyz[:, 1]
    n = xyz[:, 0]
    u = xyz[:, 2] - detected.z_floor
    keep = (
        (s > 0.5)
        & (s <= VIEW_S)
        & (np.abs(n) < VIEW_HALF)
        & (u > -0.6)
        & (u < 3.6)
        & (np.einsum("ij,ij->i", xyz, xyz) > MIN_RANGE * MIN_RANGE)
    )
    if np.any(keep):
        xs = _px_n(n[keep]).astype(np.int32)
        ys = _px_s(s[keep]).astype(np.int32)
        inside = (xs >= 0) & (xs < BEV_W) & (ys >= 0) & (ys < BEV_H)
        xs, ys = xs[inside], ys[inside]
        uu = u[keep][inside]
        ii = intensity[keep].astype(np.float64)[inside]
        hi = max(float(np.percentile(ii, 90)), 1.0) if ii.size else 1.0
        dark = np.clip(1.0 - ii / hi, 0.0, 1.0) ** 1.4
        bed = uu < 0.8
        shade = np.clip(60.0 + 195.0 * dark, 0, 255).astype(np.uint8)
        img[ys[bed], xs[bed]] = np.stack([shade[bed], shade[bed], shade[bed]], axis=1)
        wall = ~bed
        img[ys[wall], xs[wall]] = COL_WALL

    _score_overlay(img, detected)
    _axis_bev(img, detected)
    _rails_bev(img, detected)
    for mark in detected.marks:
        _gauge(img, mark, detected)
    y30 = int(_px_s(30.0))
    cv2.line(img, (0, y30), (BEV_W - 1, y30), (80, 80, 110), 1)
    _put(img, "30 m", (6, max(y30 - 4, 12)), 0.36, COL_DIM)
    y_det = int(_px_s(DETECT_S))
    cv2.line(img, (0, y_det), (BEV_W - 1, y_det), (80, 110, 80), 1)
    _put(img, f"{DETECT_S:.0f} m", (6, max(y_det - 4, 12)), 0.36, COL_DIM)
    caption = "relsy sploshnye, centry — izmereniya" if detected.axis_filtered else "os' kadra, shablon perpendikulyarno"
    _put(img, caption, (6, BEV_H - 8), 0.36, COL_DIM)
    return img


def _score_overlay(img: np.ndarray, detected: RailFrame) -> None:
    for band in detected.evidence:
        score = band.score
        if not np.any(np.isfinite(score)):
            continue
        y0 = int(np.clip(_px_s(band.s1), 0, BEV_H - 1))
        y1 = int(np.clip(_px_s(band.s0), 0, BEV_H - 1))
        if y1 <= y0:
            continue
        xs = _px_n(band.grid).astype(np.int32)
        ok = (xs >= 0) & (xs < BEV_W) & np.isfinite(score)
        if not np.any(ok):
            continue
        add = np.zeros(BEV_W, dtype=np.float64)
        np.maximum.at(add, xs[ok], np.clip(score[ok], 0.0, 1.0) * 90.0)
        strip = img[y0:y1, :, 2].astype(np.float64)
        img[y0:y1, :, 2] = np.clip(strip + add[None, :], 0, 255).astype(np.uint8)


def _axis_z(detected: RailFrame) -> np.ndarray:
    assert detected.axis_s is not None
    if detected.marks:
        return np.interp(detected.axis_s, [m.s for m in detected.marks], [m.z for m in detected.marks])
    return np.full(detected.axis_s.shape, detected.z_floor + 0.14)


def rail_tracks(detected: RailFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Две сплошные головки вдоль оси. Каждая — массив `(s, n)`."""
    if detected.axis_s is None or detected.axis_n is None or detected.axis_s.size < 2:
        return None
    s = np.asarray(detected.axis_s, dtype=np.float64)
    n = np.asarray(detected.axis_n, dtype=np.float64)
    slope = np.gradient(n, s)
    psi = np.arctan(np.clip(slope, -1.2, 1.2))
    if detected.marks:
        half = 0.5 * float(np.median([mark.spacing for mark in detected.marks]))
    else:
        half = 0.80
    left = np.column_stack([s - half * np.sin(psi), n + half * np.cos(psi)])
    right = np.column_stack([s + half * np.sin(psi), n - half * np.cos(psi)])
    return left, right


def _drawn_n(mark: RailMark, detected: RailFrame) -> float:
    """Запертый фильтр рисует шаблон на оси. Сырой центр пары остаётся точкой."""
    if (
        detected.axis_filtered
        and detected.axis_s is not None
        and detected.axis_n is not None
        and detected.axis_s.size >= 2
    ):
        return float(np.interp(mark.s, detected.axis_s, detected.axis_n))
    return float(mark.n)


def _heads(mark: RailMark, n: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Концы шаблона поперёк оси: (s, n) левой и правой головки."""
    half = 0.5 * mark.spacing
    center = float(mark.n if n is None else n)
    normal_s = -np.sin(mark.psi)
    normal_n = np.cos(mark.psi)
    left = np.array([mark.s + half * normal_s, center + half * normal_n], dtype=np.float64)
    right = np.array([mark.s - half * normal_s, center - half * normal_n], dtype=np.float64)
    return left, right


def _rails_bev(img: np.ndarray, detected: RailFrame) -> None:
    tracks = rail_tracks(detected)
    if tracks is None:
        return
    for curve in tracks:
        pts = np.stack([_px_n(curve[:, 1]), _px_s(curve[:, 0])], axis=1).astype(np.int32)
        cv2.polylines(img, [pts], False, COL_RAIL, 2, cv2.LINE_AA)


def _axis_bev(img: np.ndarray, detected: RailFrame) -> None:
    if detected.axis_s is None or detected.axis_n is None or detected.axis_s.size < 2:
        return
    pts = np.stack([_px_n(detected.axis_n), _px_s(detected.axis_s)], axis=1)
    _dashed(img, pts, COL_AXIS, dash=8, gap=6, width=2)


def _gauge(img: np.ndarray, mark: RailMark, detected: RailFrame) -> None:
    color = _score_color(mark.score)
    left, right = _heads(mark, _drawn_n(mark, detected))
    p0 = (int(_px_n(left[1])), int(_px_s(left[0])))
    p1 = (int(_px_n(right[1])), int(_px_s(right[0])))
    if not (0 <= p0[1] < BEV_H or 0 <= p1[1] < BEV_H):
        return
    cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)
    cv2.circle(img, p0, 4, color, -1, cv2.LINE_AA)
    cv2.circle(img, p1, 4, color, -1, cv2.LINE_AA)
    center = (int(_px_n(mark.n)), int(_px_s(mark.s)))
    if 0 <= center[0] < img.shape[1] and 0 <= center[1] < img.shape[0]:
        cv2.circle(img, center, 2, (255, 255, 255), -1, cv2.LINE_AA)


def _score_color(score: float) -> tuple[int, int, int]:
    t = float(np.clip((score - 0.34) / 0.7, 0.0, 1.0))
    return (int(50 + 40 * t), int(150 + 90 * t), int(210 + 45 * t))


def _front_block(image: RangeImage, detected: RailFrame) -> np.ndarray:
    vis = colorize_intensity(image.intensity)
    if detected.axis_s is not None and detected.axis_n is not None and detected.axis_s.size >= 2:
        z = _axis_z(detected)
        axis_xyz = np.stack([detected.axis_n, -detected.axis_s, z], axis=1)
        cols, rows = project_to_image(image, axis_xyz, 1)
        ok = np.isfinite(cols) & np.isfinite(rows)
        if int(ok.sum()) >= 2:
            _dashed(vis, np.stack([cols[ok], rows[ok]], axis=1), COL_AXIS, dash=7, gap=5, width=1)
    tracks = rail_tracks(detected)
    if tracks is not None:
        z = _axis_z(detected)
        for curve in tracks:
            xyz = np.stack([curve[:, 1], -curve[:, 0], z], axis=1)
            cols, rows = project_to_image(image, xyz, 1)
            ok = np.isfinite(cols) & np.isfinite(rows)
            if int(ok.sum()) < 2:
                continue
            pts = np.stack([cols[ok], rows[ok]], axis=1).astype(np.int32)
            cv2.polylines(vis, [pts], False, COL_RAIL, 1, cv2.LINE_AA)
    for mark in detected.marks:
        left, right = _heads(mark, _drawn_n(mark, detected))
        pts = np.array(
            [
                [left[1], -left[0], mark.z],
                [right[1], -right[0], mark.z],
            ],
            dtype=np.float64,
        )
        cols, rows = project_to_image(image, pts, 1)
        if not np.all(np.isfinite(cols) & np.isfinite(rows)):
            continue
        p0 = (int(cols[0]), int(rows[0]))
        p1 = (int(cols[1]), int(rows[1]))
        color = _score_color(mark.score)
        cv2.line(vis, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(vis, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(vis, p1, 3, color, -1, cv2.LINE_AA)
    _put(vis, f"FPV: shablon do {DETECT_S:.0f} m", (8, 18), 0.42, COL_DIM)
    return vis


def _panel(detected: RailFrame, height: int, speed_kmh: float | None) -> np.ndarray:
    img = np.full((height, PANEL_W, 3), 16, np.uint8)
    y = 18
    speed = "" if speed_kmh is None else f"   {speed_kmh:5.1f} km/h"
    _put(img, f"par {len(detected.marks):2d}   pol {detected.z_floor:+.2f} m   {detected.ms:.1f} ms{speed}", (8, y), 0.42)
    y += 20
    _put(img, "s     n_lev   n_prav   baza   ball  podjem", (8, y), 0.36, COL_DIM)
    y += 16
    for mark in detected.marks:
        if y > height - 8:
            break
        _put(
            img,
            f"{mark.s:5.0f}  {mark.n_left:+6.2f}  {mark.n_right:+6.2f}  {mark.spacing:4.2f}  {mark.score:4.2f}  {mark.rise:+.3f}",
            (8, y),
            0.36,
            _score_color(mark.score),
        )
        y += 14
    return img


def _dashed(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    *,
    dash: int,
    gap: int,
    width: int,
) -> None:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return
    hold = 0.0
    drawing = True
    for a, b in zip(pts[:-1], pts[1:]):
        delta = b - a
        length = float(np.hypot(delta[0], delta[1]))
        if length < 1.0:
            continue
        direction = delta / length
        pos = 0.0
        while pos < length:
            room = (dash if drawing else gap) - hold
            step = min(room, length - pos)
            if drawing and step > 0.5:
                p0 = a + direction * pos
                p1 = a + direction * (pos + step)
                cv2.line(
                    img,
                    (int(round(p0[0])), int(round(p0[1]))),
                    (int(round(p1[0])), int(round(p1[1]))),
                    color,
                    width,
                    cv2.LINE_AA,
                )
            pos += step
            hold += step
            if hold >= (dash if drawing else gap) - 1e-6:
                hold = 0.0
                drawing = not drawing


def _px_n(n: np.ndarray | float) -> np.ndarray:
    return (np.asarray(n) + VIEW_HALF) * PX_PER_M


def _px_s(s: np.ndarray | float) -> np.ndarray:
    return (VIEW_S - np.asarray(s)) * PX_PER_M


def _with_axes(body, left_ticks, bottom_ticks, left_label, bottom_label) -> np.ndarray:
    h, w = body.shape[:2]
    out = np.full((h + AXIS_BOTTOM, AXIS_LEFT + w, 3), BG, np.uint8)
    out[:h, AXIS_LEFT:] = body
    for pos, text in left_ticks:
        y = int(np.clip(pos, 6, h - 4))
        cv2.line(out, (AXIS_LEFT - 6, y), (AXIS_LEFT - 1, y), (120, 120, 120), 1)
        _put(out, text, (2, y + 4), 0.34, COL_DIM)
    for pos, text in bottom_ticks:
        x = int(AXIS_LEFT + np.clip(pos, 0, w - 1))
        cv2.line(out, (x, h + 1), (x, h + 6), (120, 120, 120), 1)
        _put(out, text, (x - 12, h + 20), 0.34, COL_DIM)
    _put(out, left_label, (2, 12), 0.34, COL_DIM)
    (label_w, _), _ = cv2.getTextSize(bottom_label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
    _put(out, bottom_label, (out.shape[1] - label_w - 4, h + 20), 0.34, COL_DIM)
    return out


def _vstack(blocks: list[np.ndarray]) -> np.ndarray:
    width = max(b.shape[1] for b in blocks)
    height = sum(b.shape[0] for b in blocks)
    out = np.full((height, width, 3), BG, np.uint8)
    y = 0
    for block in blocks:
        out[y : y + block.shape[0], : block.shape[1]] = block
        y += block.shape[0]
    return out


def _hstack(blocks: list[np.ndarray]) -> np.ndarray:
    height = max(b.shape[0] for b in blocks)
    width = sum(b.shape[1] for b in blocks)
    out = np.full((height, width, 3), BG, np.uint8)
    x = 0
    for block in blocks:
        out[: block.shape[0], x : x + block.shape[1]] = block
        x += block.shape[1]
    return out


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.4, color=COL_TEXT) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
