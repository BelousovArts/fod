"""Запись кадров в mp4 через ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def even_dims(frame: np.ndarray) -> np.ndarray:
    """libx264 требует чётные стороны."""
    h, w = frame.shape[:2]
    nh, nw = h + (h % 2), w + (w % 2)
    if (nh, nw) == (h, w):
        return frame
    out = np.zeros((nh, nw, 3), dtype=np.uint8)
    out[:h, :w] = frame
    return out


def open_ffmpeg(path: Path, width: int, height: int, fps: float) -> subprocess.Popen[bytes]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.4f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "fast",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
