"""Раскраска range image: дальность (лог-шкала) и интенсивность."""

from __future__ import annotations

import cv2
import numpy as np

NO_DATA = (18, 18, 18)


def colorize_range(rng: np.ndarray, vmax: float) -> np.ndarray:
    if rng.size == 0:
        return np.zeros(rng.shape + (3,), np.uint8)
    t = np.zeros(rng.shape, dtype=np.float32)
    ok = np.isfinite(rng) & (rng > 0)
    t[ok] = np.clip(np.log1p(rng[ok]) / np.log1p(vmax), 0.0, 1.0)
    u8 = np.clip(t * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    bgr[~ok] = NO_DATA
    return bgr


def colorize_intensity(intensity: np.ndarray, vmax: float = 80.0, gamma: float = 0.6) -> np.ndarray:
    """Фиксированная шкала: интенсивность — физическая величина, а не контраст кадра.

    Нормировка на процентиль кадра выглядит заманчиво, но 98-й процентиль гуляет
    по сценам от 26 до 65 (измерено), поэтому в видео яркость «дышит» от кадра к
    кадру и сравнивать участки нельзя. Медиана в тоннеле 5…15, отсюда vmax = 80 и
    гамма 0.6: тёмный низ различим, яркие отражатели честно уходят в насыщение.
    """
    if intensity.size == 0:
        # cv2.applyColorMap на пустом массиве падает по SIGFPE.
        return np.zeros(intensity.shape + (3,), np.uint8)
    t = np.zeros(intensity.shape, dtype=np.float32)
    ok = np.isfinite(intensity)
    t[ok] = np.clip(intensity[ok] / max(vmax, 1.0), 0.0, 1.0) ** gamma
    u8 = np.clip(t * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    bgr[~ok] = NO_DATA
    return bgr
