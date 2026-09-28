"""Пары головок по сегментации range image — замена `detect_rails` для `RailTracker`.

Сеть размечает пиксели кадра (левая / правая головка), их точки идут по тем же
полосам дальности и через то же окно вокруг оси, что и у шаблонного поиска.
В полосе — медиана бокового положения каждой головки, приведённая к середине
полосы по курсу оси. Видна одна головка — пара достраивается по колее.

Запасной путь — шаблонный поиск (`detect_rails`). Сеть видит в пикселе абсолютную
высоту точки, и при другой установке лидара (`doubleT_obstacle`: головки на 0.45 м
ниже, наклон 1.6 %) не находит головок совсем. Если сеть `FALLBACK_AFTER` кадров
подряд даёт меньше `FALLBACK_MARKS` пар, кадр ищется шаблоном, пока сеть снова
не найдёт пары.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from fod.rail_seg_data import CR, HALF, LEFT, RIGHT, ring_image
from fod.rail_template import (
    S_MIN, START_S_MAX, GAP_BUDGET, RailFrame, RailMark, _fit_axis, _gate, count_limits, detect_rails, template_bands,
)
from fod.rails import SPACING_LIMITS, SINGLE_HEAD_PENALTY

DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "models" / "rail_seg.pt"
FALLBACK_MARKS = 4
FALLBACK_AFTER = 3


class SegRailDetector:
    def __init__(self, ckpt: Path = DEFAULT_CKPT, device: str = "cuda", s_max: float = 80.0, fallback: bool = True) -> None:
        from fod.rail_seg_net import load

        self.net = load(ckpt, device)
        self.bands = template_bands(S_MIN, s_max)
        self.last = None
        self.fallback = fallback
        self.net_misses = 0            # кадров подряд, где сеть дала меньше FALLBACK_MARKS пар
        self.used_template = False     # последний кадр — по шаблону
        self.template_frames = 0

    def classify(self, xyz: np.ndarray, intensity: np.ndarray):
        """Изображение кадра и вероятности классов (4, 128, W)."""
        self.last = self.classify_only(xyz, intensity)
        return self.last

    def classify_only(self, xyz: np.ndarray, intensity: np.ndarray):
        """То же без сохранения в `last` — можно звать из другого потока заранее и отдать в `pre`."""
        from fod.rail_seg_net import predict

        img = ring_image(xyz, intensity)
        has = img.index >= 0
        z_cm = np.zeros(img.index.shape, np.int16)
        z_cm[has] = np.clip(np.rint(xyz[img.index[has], 2] * 100.0), -32000, 32000)
        rng_cm = np.clip(np.rint(img.range * 100.0), 0, 65535).astype(np.uint16)
        prob = predict(self.net, rng_cm, z_cm, img.intensity)
        return img, prob

    def __call__(self, xyz: np.ndarray, intensity: np.ndarray, prior=None, predict_m: float | None = None, pre=None) -> RailFrame:
        """`pre` — `(xyz, img, prob)` от `classify_only` для этого же массива `xyz`; иначе сеть считается здесь."""
        started = time.perf_counter()
        if pre is not None and pre[0] is xyz:
            self.last = (pre[1], pre[2])
        else:
            self.classify(xyz, intensity)
        frame = self._net_frame(xyz, intensity, prior, predict_m)
        self.net_misses = self.net_misses + 1 if len(frame.marks) < FALLBACK_MARKS else 0
        self.used_template = False
        if self.fallback and self.net_misses >= FALLBACK_AFTER:
            # Пустые лучи (0, 0, 0) шаблону не нужны, а у Hesai их больше половины кадра.
            valid = xyz.any(axis=1)
            backup = detect_rails(xyz[valid], np.asarray(intensity)[valid], prior=prior, predict_m=predict_m)
            if len(backup.marks) > len(frame.marks):
                frame, self.used_template = backup, True
                self.template_frames += 1
        frame.ms = (time.perf_counter() - started) * 1e3
        return frame

    def _net_frame(self, xyz: np.ndarray, intensity: np.ndarray, prior, predict_m: float | None) -> RailFrame:
        started = time.perf_counter()
        xyz = np.asarray(xyz)
        img, prob = self.last
        cls = prob.argmax(axis=0)
        heads = {}
        for c in (LEFT, RIGHT):
            m = (cls == c) & (img.index >= 0)
            p = xyz[img.index[m]].astype(np.float64)
            heads[c] = (-p[:, 1], p[:, 0], p[:, 2], prob[c][m])
        locked = prior is not None and bool(getattr(prior, "locked", False))
        marks: list[RailMark] = []
        gap = 0.0
        for s0, s1, step in self.bands:
            s_mid = 0.5 * (s0 + s1)
            if not marks and s_mid > START_S_MAX and not locked:
                break
            axis, psi, half = _gate(marks, s_mid, gap, prior if locked else None)
            mark = self._band(heads, s0, s1, s_mid, axis, psi, half, bool(marks) or locked)
            if mark is None:
                if marks or locked:
                    gap += step
                    if gap > GAP_BUDGET:
                        break
                continue
            gap = 0.0
            marks.append(mark)
        axis_s = axis_n = None
        if len(marks) >= 2:
            axis = _fit_axis(marks)
            for mark in marks:
                mark.psi = axis.psi(mark.s)
            s_lo, s_hi = float(marks[0].s), float(marks[-1].s)
            if predict_m is not None and np.isfinite(predict_m):
                s_hi = max(float(predict_m), s_lo)
            axis_s = np.linspace(s_lo, s_hi, max(int(round((s_hi - s_lo) / 0.5)), 2))
            axis_n = np.asarray(axis.n(axis_s), dtype=np.float64)
        z_floor = float(np.median(heads[LEFT][2])) - 0.2 if heads[LEFT][2].size else -1.4
        return RailFrame(z_floor=z_floor, marks=marks, evidence=[], ms=(time.perf_counter() - started) * 1e3,
                         axis_s=axis_s, axis_n=axis_n)

    def contact_rail(self, xyz: np.ndarray, axis, head):
        """Линия КР по классу сети этого кадра: бины вдоль оси путевых рельсов, без геометрического трекера."""
        from fod.contact_rail import OFFSET, ContactRailFrame, offset_axis

        frame = ContactRailFrame()
        if self.last is None or axis is None:
            return frame
        img, prob = self.last
        m = (prob.argmax(axis=0) == CR) & (img.index >= 0)
        if int(m.sum()) < 30:
            return frame
        p = xyz[img.index[m]].astype(np.float64)
        s, n, z = -p[:, 1], p[:, 0], p[:, 2]
        ok = (s > 4.0) & (s < 150.0) & np.isfinite(s + n + z)
        s, n, z = s[ok], n[ok], z[ok]
        if s.size < 20:
            return frame
        n_rel = n - np.asarray(axis.n(s), dtype=np.float64)
        if head is not None and getattr(head, "head_coeff", None) is not None:
            u = z - np.asarray(head.head_z(s), dtype=np.float64)
            band = (u > 0.05) & (u < 0.95)
        else:
            band = np.ones(s.shape, bool)
        left, right = band & (n_rel > 0.95) & (n_rel < 2.05), band & (n_rel < -0.95) & (n_rel > -2.05)
        side = 1 if int(left.sum()) >= int(right.sum()) else -1
        sel = left if side == 1 else right
        if int(sel.sum()) < 20:
            return frame
        s, n, z, n_rel = s[sel], n[sel], z[sel], n_rel[sel]
        rows = []
        b = 4.0
        while b < 150.0:
            w = max(2.0, 0.04 * b)
            k = (s >= b) & (s < b + w)
            if int(k.sum()) >= 4 and abs(float(np.median(n_rel[k])) - side * OFFSET) < 0.35:
                rows.append((float(np.median(s[k])), float(np.median(n[k])), float(np.median(z[k]))))
            b += w
        if len(rows) < 4:
            return frame
        kept = [rows[0]]
        for row in rows[1:]:
            if row[0] - kept[-1][0] > 12.0:
                break
            kept.append(row)
        if len(kept) < 4:
            return frame
        line = np.array(kept, dtype=np.float64)
        frame.lines[side] = line
        frame.side = side
        frame.axis_s, frame.axis_n = offset_axis(line, side)
        return frame

    @staticmethod
    def _band(heads, s0, s1, s_mid, axis, psi, half, chained) -> RailMark | None:
        tan, cos = np.tan(psi), max(np.cos(psi), 0.75)
        need = count_limits(s_mid)[0]
        found = {}
        for c, sign in ((LEFT, 1.0), (RIGHT, -1.0)):
            s, n, z, w = heads[c]
            k = (s >= s0) & (s < s1)
            if not k.any():
                continue
            n_mid = n[k] - tan * (s[k] - s_mid)
            centre = float(axis.n(s_mid)) + sign * HALF / cos
            g = np.abs(n_mid - centre) < half
            if g.sum() >= need:
                found[c] = (float(np.median(n_mid[g])), float(np.median(z[k][g])), float(np.mean(w[k][g])))
        if LEFT in found and RIGHT in found:
            (nl, zl, wl), (nr, zr, wr) = found[LEFT], found[RIGHT]
            spacing = (nl - nr) * cos
            if not SPACING_LIMITS[0] <= spacing <= SPACING_LIMITS[1]:
                return None
            score, z = 0.5 * (wl + wr), 0.5 * (zl + zr)
        elif chained and found:
            c, (nh, z, score) = next(iter(found.items()))
            nl, nr = (nh, nh - 2 * HALF / cos) if c == LEFT else (nh + 2 * HALF / cos, nh)
            spacing, score = 2 * HALF, score * SINGLE_HEAD_PENALTY
        else:
            return None
        return RailMark(s0=s0, s1=s1, s=s_mid, n=0.5 * (nl + nr), n_left=nl, n_right=nr, z=z, score=score, rise=0.0, dark=0.0,
                        prominence=0.0, spacing=spacing, psi=psi)
