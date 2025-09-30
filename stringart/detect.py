from __future__ import annotations
import numpy as np
import cv2
from typing import Tuple, Optional

__all__ = ["auto_detect", "sample_rim_signal", "detect_board_circle", "estimate_nails_by_fft"]

def _ensure_gray01(path_or_img, target_size: Optional[int] = None) -> np.ndarray:
    """Load BGR or accept np.ndarray, return grayscale float32 in [0,1]."""
    if isinstance(path_or_img, str):
        img = cv2.imread(path_or_img, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path_or_img)
    else:
        img = path_or_img
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if target_size is not None:
        g = cv2.resize(g, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return g

def detect_board_circle(gray01: np.ndarray) -> Tuple[int, int, int]:
    """Hough circle; fallback to centered circle."""
    H, W = gray01.shape
    g8 = (gray01 * 255).astype(np.uint8)
    g_blur = cv2.GaussianBlur(g8, (0, 0), 2.0)
    m = min(H, W)
    circles = cv2.HoughCircles(
        g_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(m * 0.25),
        param1=120, param2=50, minRadius=int(m * 0.33), maxRadius=int(m * 0.50)
    )
    if circles is not None and len(circles) > 0:
        x, y, r = circles[0][0]
        return int(round(x)), int(round(y)), int(round(r))
    c = m // 2
    return c, c, int(m * 0.45)

def sample_rim_signal(
    gray01: np.ndarray,
    cx: int, cy: int, R: int,
    invert: bool,
    theta_samples: int = 4096,
    r_in_frac: float = 0.90,
    r_out_frac: float = 0.98,
) -> np.ndarray:
    """Angular max-intensity signal on a thin perimeter annulus."""
    H, W = gray01.shape
    r_in = max(5, int(R * r_in_frac))
    r_out = min(R, int(R * r_out_frac))
    thetas = np.linspace(-np.pi/2, 3*np.pi/2, theta_samples, endpoint=False)
    vals = np.zeros(theta_samples, dtype=np.float32)
    rr = np.arange(r_in, r_out, 1, dtype=np.float32)
    for i, th in enumerate(thetas):
        ct, st = np.cos(th), np.sin(th)
        xs = cx + ct * rr
        ys = cy + st * rr
        xs = np.clip(xs, 0, W - 1 - 1e-3)
        ys = np.clip(ys, 0, H - 1 - 1e-3)
        x0 = np.floor(xs).astype(np.int32); x1 = x0 + 1
        y0 = np.floor(ys).astype(np.int32); y1 = y0 + 1
        wx = xs - x0; wy = ys - y0
        Ia = gray01[y0, x0]; Ib = gray01[y0, x1]
        Ic = gray01[y1, x0]; Id = gray01[y1, x1]
        band = Ia*(1-wx)*(1-wy) + Ib*wx*(1-wy) + Ic*(1-wx)*wy + Id*wx*wy
        vals[i] = float(np.max(band))
    if invert:
        vals = 1.0 - vals
    vals = vals - float(np.mean(vals))
    return vals

def estimate_nails_by_fft(
    gray01: np.ndarray,
    cx: int, cy: int, R: int,
    invert: bool,
    n_min: int = 160,
    n_max: int = 300
) -> Tuple[int, float, float]:
    """Return (n_est, phase, peak_ratio)."""
    theta_samples = 4096
    vals = sample_rim_signal(gray01, cx, cy, R, invert, theta_samples=theta_samples)
    spec = np.abs(np.fft.rfft(vals))
    k_min = max(2, n_min); k_max = min(theta_samples//2 - 1, n_max)
    k = int(np.argmax(spec[k_min:k_max+1]) + k_min)
    n_est = int(k)

    # phase
    t = np.arange(theta_samples, dtype=np.float32)
    ang = 2*np.pi*k*t/theta_samples
    a = float(np.sum(vals * np.cos(ang)))
    b = float(np.sum(vals * np.sin(ang)))
    phi = float(np.arctan2(-b, a))

    # peak ratio
    nb = 6
    lo = max(k-nb, 1); hi = min(k+nb, len(spec)-1)
    neigh = np.r_[spec[lo:k], spec[k+1:hi+1]]
    peak_ratio = float(spec[k] / (np.median(neigh) + 1e-9))

    if n_est % 4 != 0:
        n4 = int(round(n_est / 4.0) * 4)
        n_est = max(n_min, min(n_max, n4))
    return n_est, phi, peak_ratio

def auto_detect(path_or_img, canvas: int, invert: bool) -> Tuple[int,int,int,int,float,float]:
    """Convenience wrapper used by the CLI."""
    g = _ensure_gray01(path_or_img, target_size=canvas)
    cx, cy, R = detect_board_circle(g)
    n, phi, pr = estimate_nails_by_fft(g, cx, cy, R, invert=invert)
    return n, cx, cy, R, phi, pr
