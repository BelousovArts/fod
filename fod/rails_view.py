"""Отрисовка результата трекера: вид от первого лица, BEV полотна, телеметрия.

Две проекции одного и того же прогноза:

* **FPV** — интенсивность в range image, то есть буквально «глазами сенсора»;
  ось и нитки рельсов проецируются в (азимут, элевация) и ложатся на дорожку.
  Здесь сразу видно, что головки рельсов тёмные и как далеко они читаются.
* **BEV** — вид сверху, где точки полотна покрашены по «темноте»
  (инвертированная интенсивность), поэтому рельсы — две светлые нитки.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from fod.cloud import MIN_RANGE
from fod.colormaps import colorize_intensity
from fod.odometry import MotionEstimate
from fod.rails import RAIL_BANDS
from fod.range_image import RangeImage
from fod.track_filter import TrackEstimate

BEV_W = 540
BEV_H = 780
PANEL_W = 430
TITLE_H = 34
AXIS_LEFT = 44
AXIS_BOTTOM = 26
BG = 20

COL_AXIS = (90, 245, 90)
COL_RAIL = (60, 215, 255)
COL_HIT = (60, 215, 255)
COL_MISS = (110, 110, 200)
COL_WALL = (120, 70, 45)
COL_TEXT = (228, 228, 228)
COL_DIM = (150, 150, 150)

FRONT_S_MIN = 2.0    # ближе рельсы уходят за край сектора


@dataclass
class ViewConfig:
    view: str = "both"          # front / bev / both
    s_max: float = 130.0
    n_half: float = 6.0
    front_scale: int = 1
    show_evidence: bool = True
    show_corridor: bool = False
    history: int = 400


class TrackView:
    """Держит историю оси, чтобы на кадре было видно, скачет она или нет."""

    def __init__(self, config: ViewConfig | None = None) -> None:
        self.cfg = config or ViewConfig()
        self.history: deque[tuple[float, float, bool]] = deque(maxlen=self.cfg.history)

    def push(self, est: TrackEstimate) -> None:
        self.history.append((float(est.coeff[0]), float(est.n_at(40.0)), est.locked))

    def render(
        self,
        xyz: np.ndarray,
        intensity: np.ndarray,
        est: TrackEstimate,
        *,
        title: str,
        front: RangeImage | None = None,
        motion: MotionEstimate | None = None,
    ) -> np.ndarray:
        self.push(est)
        blocks: list[np.ndarray] = []
        scale = self.cfg.front_scale
        if front is not None and self.cfg.view in {"front", "both"}:
            if self.cfg.view == "front":
                scale = max(scale, 2)
            blocks.append(self._front_block(front, est, scale))
        if self.cfg.view in {"bev", "both"}:
            blocks.append(self._bev_block(xyz, intensity, est))

        body = _vstack(blocks)
        canvas = _hstack([body, self._panel(est, body.shape[0], motion)])
        out = np.full((TITLE_H + canvas.shape[0], canvas.shape[1], 3), BG, np.uint8)
        out[TITLE_H:] = canvas
        _put(out, title, (AXIS_LEFT, 22), 0.52, (245, 245, 245))
        return out

    # --- вид от первого лица ------------------------------------------------

    def _front_block(self, image: RangeImage, est: TrackEstimate, scale: int) -> np.ndarray:
        vis = colorize_intensity(image.intensity)
        if scale != 1:
            vis = cv2.resize(vis, (vis.shape[1] * scale, vis.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

        s = np.arange(FRONT_S_MIN, min(self.cfg.s_max, 200.0), 0.5)
        n = est.n_at(s)
        z_rail = est.rail_z(s)
        half = 0.5 * est.spacing
        solid = s <= est.s_known
        for offset, z, color, width in (
            (-half, z_rail, COL_RAIL, 2),
            (+half, z_rail, COL_RAIL, 2),
            (0.0, np.full_like(s, est.z_floor + 0.02), COL_AXIS, 1),
        ):
            pts = np.stack([n + offset, -s, z], axis=1)
            _polyline(vis, project_to_image(image, pts, scale), color, width, solid)

        for report in est.bands:
            if not report.accepted or report.n_meas is None:
                continue
            point = np.array([[report.n_meas, -report.s_mid, float(est.rail_z(report.s_mid))]])
            cols, rows = project_to_image(image, point, scale)
            if np.isfinite(cols[0]) and np.isfinite(rows[0]):
                cv2.circle(vis, (int(cols[0]), int(rows[0])), 4, COL_HIT, 1)

        _put(vis, "FPV: range image, intensivnost'", (8, 18), 0.42, COL_DIM)
        return _with_axes(
            vis,
            left_ticks=_elevation_ticks(image, vis.shape[0]),
            bottom_ticks=_azimuth_ticks(image, vis.shape[1]),
            left_label="el, deg",
            bottom_label="azimut, deg",
        )

    # --- вид сверху ---------------------------------------------------------

    def _px_n(self, n: np.ndarray | float) -> np.ndarray:
        return (np.asarray(n) + self.cfg.n_half) / (2.0 * self.cfg.n_half) * (BEV_W - 1)

    def _px_s(self, s: np.ndarray | float) -> np.ndarray:
        return (BEV_H - 1) * (1.0 - np.asarray(s) / self.cfg.s_max)

    def _bev_block(self, xyz: np.ndarray, intensity: np.ndarray, est: TrackEstimate) -> np.ndarray:
        bev = self._bev(xyz, intensity, est)
        step = 20 if self.cfg.s_max > 60 else (10 if self.cfg.s_max > 30 else 5)
        left = [(float(self._px_s(s)), f"{s}") for s in range(0, int(self.cfg.s_max) + 1, step)]
        half = int(self.cfg.n_half)
        bottom = [(float(self._px_n(n)), f"{n:+d}") for n in range(-half, half + 1, 2)]
        return _with_axes(bev, left, bottom, "s, m", "n, m")

    def _bev(self, xyz: np.ndarray, intensity: np.ndarray, est: TrackEstimate) -> np.ndarray:
        cfg = self.cfg
        img = np.full((BEV_H, BEV_W, 3), 12, np.uint8)

        s = -xyz[:, 1]
        n = xyz[:, 0]
        u = xyz[:, 2] - est.z_floor
        keep = (
            (s > 0.5)
            & (s < cfg.s_max)
            & (np.abs(n) < cfg.n_half)
            & (u > -0.6)
            & (u < 3.6)
            & (np.einsum("ij,ij->i", xyz, xyz) > MIN_RANGE * MIN_RANGE)
        )
        if np.any(keep):
            xs = self._px_n(n[keep]).astype(np.int32)
            ys = self._px_s(s[keep]).astype(np.int32)
            uu = u[keep]
            ii = intensity[keep].astype(np.float64)
            hi = max(float(np.percentile(ii, 90)), 1.0)
            dark = np.clip(1.0 - ii / hi, 0.0, 1.0) ** 1.4
            bed = uu < 0.8
            shade = np.clip(60.0 + 195.0 * dark, 0, 255).astype(np.uint8)
            img[ys[bed], xs[bed]] = np.stack([shade[bed], shade[bed], shade[bed]], axis=1)
            wall = ~bed
            img[ys[wall], xs[wall]] = COL_WALL

        if cfg.show_evidence:
            self._evidence_overlay(img, est)

        s_line = np.arange(0.0, cfg.s_max, 0.5)
        n_axis = est.n_at(s_line)
        half_spacing = 0.5 * est.spacing
        solid = s_line <= est.s_known
        # За измеренной зоной честнее показать конус ±σ, чем одну линию: она там
        # и обязана гулять, потому что кривизна впереди не наблюдается.
        if np.any(~solid):
            self._uncertainty(img, s_line[~solid], n_axis[~solid], est.sigma_at(s_line[~solid]))
        self._polyline_bev(img, n_axis - half_spacing, s_line, COL_RAIL, 2, solid)
        self._polyline_bev(img, n_axis + half_spacing, s_line, COL_RAIL, 2, solid)
        self._polyline_bev(img, n_axis, s_line, COL_AXIS, 2, solid)
        if cfg.show_corridor:
            half = est.corridor_half(s_line)
            self._polyline_bev(img, n_axis - half, s_line, (170, 170, 90), 1, solid)
            self._polyline_bev(img, n_axis + half, s_line, (170, 170, 90), 1, solid)

        for report in est.bands:
            if report.n_meas is None:
                continue
            x = int(self._px_n(report.n_meas))
            y = int(self._px_s(report.s_mid))
            if report.accepted:
                cv2.circle(img, (x, y), 4, COL_HIT, -1)
            else:
                cv2.circle(img, (x, y), 4, COL_MISS, 1)

        if est.s_known < cfg.s_max:
            y_known = int(self._px_s(est.s_known))
            cv2.line(img, (0, y_known), (BEV_W - 1, y_known), (70, 130, 70), 1)
            _put(img, f"s_known {est.s_known:.0f} m", (6, y_known - 5), 0.36, (120, 200, 120))

        s_rail = RAIL_BANDS[-1][1]
        if s_rail < cfg.s_max:
            y_rail = int(self._px_s(s_rail))
            cv2.line(img, (0, y_rail), (BEV_W - 1, y_rail), (60, 60, 120), 1)
            _put(img, "zona 1: relsy", (6, y_rail + 14), 0.36, (130, 130, 220))
            _put(img, "zona 2: sechenie tonnelya", (6, y_rail - 6), 0.36, (130, 130, 220))
        _put(img, "punktir i konus +-sigma = ekstrapolyatsiya, ne izmerenie", (6, BEV_H - 8), 0.36, COL_DIM)
        return img

    def _evidence_overlay(self, img: np.ndarray, est: TrackEstimate) -> None:
        for band in est.evidence:
            score = band.score
            if not np.any(np.isfinite(score)):
                continue
            y0 = int(self._px_s(band.s1))
            y1 = int(self._px_s(band.s0))
            if y1 <= y0:
                continue
            xs = self._px_n(band.grid).astype(np.int32)
            ok = (xs >= 0) & (xs < BEV_W) & np.isfinite(score)
            if not np.any(ok):
                continue
            add = np.zeros(BEV_W, dtype=np.float64)
            np.maximum.at(add, xs[ok], np.clip(score[ok], 0.0, 1.0) * 120.0)
            strip = img[y0:y1, :, 2].astype(np.float64)
            img[y0:y1, :, 2] = np.clip(strip + add[None, :], 0, 255).astype(np.uint8)

    def _uncertainty(self, img: np.ndarray, s: np.ndarray, n: np.ndarray, sigma: np.ndarray) -> None:
        lo = np.stack([self._px_n(n - sigma), self._px_s(s)], axis=1)
        hi = np.stack([self._px_n(n + sigma), self._px_s(s)], axis=1)
        polygon = np.concatenate([lo, hi[::-1]]).astype(np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [polygon], (70, 120, 70))
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0.0, dst=img)

    def _polyline_bev(self, img, n, s, color, width: int, solid=None) -> None:
        _polyline(img, (self._px_n(n), self._px_s(s)), color, width, solid)

    # --- телеметрия ---------------------------------------------------------

    def _panel(self, est: TrackEstimate, height: int, motion: MotionEstimate | None = None) -> np.ndarray:
        img = np.full((height, PANEL_W, 3), 16, np.uint8)
        y = 18
        lock = "LOCK" if est.locked else f"SEARCH ({est.misses})"
        radius = "pryamaya" if not np.isfinite(est.radius_m) or est.radius_m > 9000 else f"R={est.radius_m:.0f} m"
        bind = "n/a" if est.binding is None else f"{est.binding:+.2f} m"
        _put(img, f"{lock}   {radius}   {est.ms:.1f} ms", (8, y), 0.44, COL_AXIS if est.locked else COL_MISS)
        y += 18
        if motion is not None:
            odom_lock = "LOCK" if motion.locked else "INIT"
            _put(
                img,
                f"v={motion.kmh:5.1f} km/h  s={motion.s:6.1f} m  {odom_lock} {motion.source}  {motion.ms:.1f} ms",
                (8, y),
                0.38,
                COL_AXIS if motion.locked else COL_MISS,
            )
            y += 16
        _put(
            img,
            f"os': n(0)={est.coeff[0]:+.3f}  n(40)={float(est.n_at(40.0)):+.2f}  n(100)={float(est.n_at(100.0)):+.2f}",
            (8, y),
            0.38,
        )
        y += 16
        _put(img, f"sigma: 40 m {float(est.sigma_at(40.0)):.2f}   100 m {float(est.sigma_at(100.0)):.2f}", (8, y), 0.38)
        y += 16
        _put(
            img,
            f"privyazka {bind}  baza {est.spacing:.2f} m  golovka {est.head_u0:+.2f} m"
            f"  uklon {1000.0 * est.grade:+.0f} promille",
            (8, y),
            0.36,
            COL_DIM,
        )
        y += 22

        _put(img, "s     istochnik  n_izm   podjem  temnota  ball", (8, y), 0.36, COL_DIM)
        y += 15
        for report in est.bands:
            color = COL_AXIS if report.accepted else COL_DIM
            meas = "  n/a" if report.n_meas is None else f"{report.n_meas:+.2f}"
            _put(
                img,
                f"{report.s_mid:5.0f}  {report.source:9s} {meas}   {report.rise:+.3f}   {report.dark:+.2f}   {report.score:.2f}",
                (8, y),
                0.35,
                color,
            )
            y += 14

        if y + 170 <= height:
            self._history_plot(img, top=y + 16)
        return img

    def _history_plot(self, img: np.ndarray, top: int) -> None:
        h, w = 150, PANEL_W - 24
        left = 12
        cv2.rectangle(img, (left, top), (left + w, top + h), (60, 60, 60), 1)
        cv2.line(img, (left, top + h // 2), (left + w, top + h // 2), (50, 50, 50), 1)
        if len(self.history) < 2:
            return
        n0 = np.array([float(v[0]) for v in self.history])
        n40 = np.array([float(v[1]) for v in self.history])
        lock = np.array([bool(v[2]) for v in self.history])
        span = max(float(np.abs(np.concatenate([n0, n40])).max()), 0.5)
        _put(img, f"istoriya osi: n(0) belyy, n(40) zheltyy, +-{span:.1f} m", (left, top - 6), 0.34, COL_DIM)
        xs = np.linspace(left, left + w, n0.size).astype(np.int32)

        def curve(values: np.ndarray, color) -> None:
            ys = (top + h / 2 - values / span * (h / 2 - 4)).astype(np.int32)
            cv2.polylines(img, [np.stack([xs, ys], axis=1)], False, color, 1, cv2.LINE_AA)

        curve(n40, (80, 220, 240))
        curve(n0, (235, 235, 235))
        for i in np.nonzero(~lock)[0]:
            cv2.line(img, (int(xs[i]), top + h - 4), (int(xs[i]), top + h - 1), COL_MISS, 1)


# --- проекция в range image -------------------------------------------------


def _grid_index(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Дробный индекс по монотонной (в любую сторону) угловой сетке."""
    idx = np.arange(grid.size, dtype=np.float64)
    if grid[0] <= grid[-1]:
        return np.interp(values, grid, idx, left=np.nan, right=np.nan)
    return np.interp(values, grid[::-1], idx[::-1], left=np.nan, right=np.nan)


def project_to_image(image: RangeImage, xyz: np.ndarray, scale: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Точки сцены → пиксели range image по азимуту и элевации."""
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    el = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
    cols = _grid_index(az, image.azimuth_deg) * scale
    rows = _grid_index(el, image.elevation_deg) * scale
    return cols, rows


def _elevation_ticks(image: RangeImage, height: int) -> list[tuple[float, str]]:
    ticks = []
    for deg in (10.0, 5.0, 0.0, -5.0, -10.0, -20.0):
        row = _grid_index(np.array([deg]), image.elevation_deg)[0]
        if np.isfinite(row):
            ticks.append((row / image.elevation_deg.size * height, f"{deg:+.0f}"))
    return ticks


def _azimuth_ticks(image: RangeImage, width: int) -> list[tuple[float, str]]:
    ticks = []
    for rel in (-20.0, -10.0, 0.0, 10.0, 20.0):
        col = _grid_index(np.array([-90.0 + rel]), image.azimuth_deg)[0]
        if np.isfinite(col):
            ticks.append((col / image.azimuth_deg.size * width, f"{rel:+.0f}"))
    return ticks


# --- компоновка -------------------------------------------------------------


def _with_axes(
    body: np.ndarray,
    left_ticks: list[tuple[float, str]],
    bottom_ticks: list[tuple[float, str]],
    left_label: str,
    bottom_label: str,
) -> np.ndarray:
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
        _put(out, text, (x - 10, h + 20), 0.34, COL_DIM)
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


def _polyline(
    img: np.ndarray,
    pixels: tuple[np.ndarray, np.ndarray],
    color,
    width: int,
    solid: np.ndarray | None = None,
) -> None:
    """`solid` — маска измеренной части; остальное рисуется пунктиром."""
    xs, ys = pixels
    ok = np.isfinite(xs) & np.isfinite(ys)
    if int(ok.sum()) < 2:
        return
    pts = np.stack([xs, ys], axis=1)
    if solid is None:
        solid = np.ones(xs.shape, dtype=bool)
    _segments(img, pts, ok & solid, color, width, dash=None)
    _segments(img, pts, ok & ~solid, color, max(width - 1, 1), dash=(6, 5))


def _segments(img, pts, mask, color, width, dash) -> None:
    if int(mask.sum()) < 2:
        return
    idx = np.nonzero(mask)[0]
    breaks = np.nonzero(np.diff(idx) > 1)[0] + 1
    for chunk in np.split(idx, breaks):
        if chunk.size < 2:
            continue
        if dash is None:
            cv2.polylines(img, [pts[chunk].astype(np.int32)], False, color, width, cv2.LINE_AA)
            continue
        on, off = dash
        for start in range(0, chunk.size, on + off):
            piece = chunk[start : start + on]
            if piece.size >= 2:
                cv2.polylines(img, [pts[piece].astype(np.int32)], False, color, width, cv2.LINE_AA)


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.4, color=COL_TEXT) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
