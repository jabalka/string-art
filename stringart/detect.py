from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2
import numpy.fft as fft


def _radial_edge_profile(gray01: np.ndarray, cx: int, cy: int, r_min: int, r_max: int) -> int:
    """
    Fallback: pick radius where the radial edge energy is maximized.
    Looks at gradient magnitude sampled on circles of increasing radius.
    """
    H, W = gray01.shape
    gx = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    thetas = np.linspace(-np.pi / 2, 3 * np.pi / 2, 2048, endpoint=False).astype(np.float32)
    ct = np.cos(thetas)
    st = np.sin(thetas)

    best_r, best_s = r_min, -1.0
    for r in range(r_min, r_max + 1):
        xs = cx + r * ct
        ys = cy + r * st
        xs = np.clip(xs, 0, W - 1 - 1e-3)
        ys = np.clip(ys, 0, H - 1 - 1e-3)
        x0 = np.floor(xs).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(ys).astype(np.int32)
        y1 = y0 + 1
        wx = xs - x0
        wy = ys - y0
        Ia = mag[y0, x0]
        Ib = mag[y0, x1]
        Ic = mag[y1, x0]
        Id = mag[y1, x1]
        ring = Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy) + Ic * (1 - wx) * wy + Id * wx * wy
        s = float(ring.mean())
        if s > best_s:
            best_s = s
            best_r = r
    return best_r


def detect_board_circle(gray01: np.ndarray) -> Tuple[int, int, int]:
    """
    Try Hough first; if weak, use a radial edge profile.
    Then return a *slightly larger* radius to be generous (avoid clipping threads).
    """
    H, W = gray01.shape
    g8 = (gray01 * 255).astype(np.uint8)
    g_blur = cv2.GaussianBlur(g8, (0, 0), 2.0)
    m = min(H, W)

    # Hough with wider search
    circles = cv2.HoughCircles(
        g_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=int(m * 0.5),
        param1=120,
        param2=40,
        minRadius=int(m * 0.35),
        maxRadius=int(m * 0.60),
    )
    if circles is not None and len(circles) > 0:
        x, y, r = circles[0][0]
        cx, cy, R = int(round(x)), int(round(y)), int(round(r))
    else:
        # center fallback
        cx = cy = m // 2
        R = _radial_edge_profile(gray01, cx, cy, int(m * 0.35), int(m * 0.60))

    # Be generous by a few percent so we don't trim outer strings
    R = int(round(R * 1.06))  # +6%
    # Clamp inside image
    R = max(10, min(R, (m // 2) - 2))
    return cx, cy, R


def detect_nail_count(
    gray01: np.ndarray,
    cx: int,
    cy: int,
    R: int,
    n_min: int = 20,
    n_max: int = 400,
    smooth_sigma: float = 2.0,
    suppress_harmonics: bool = True,
    debug_dir: Optional[str] = None,
) -> int:
    """Estimate nail count using a robust ring signal analysis.

    Improvements over the previous simplistic FFT pick:
      * Adaptive min/max range (n_min..n_max)
      * Gaussian smoothing of the angular signal to reduce spurious high freq noise
      * Optional harmonic suppression: divide spectrum by smoothed version and penalize multiples
      * Peak prominence filtering to avoid picking double counts (e.g., 36 -> 72)
      * Fallback to nearest multiple of 4 if within tolerance (common board design)
    """
    samples = 4096
    thetas = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    ring_r = max(5, R - 5)
    xs = cx + ring_r * np.cos(thetas)
    ys = cy + ring_r * np.sin(thetas)
    xs = np.clip(xs, 0, gray01.shape[1] - 1)
    ys = np.clip(ys, 0, gray01.shape[0] - 1)
    vals = gray01[ys.astype(int), xs.astype(int)].astype(np.float32)
    # emphasize nail heads as dark peaks by inverting if background brighter
    # decide simple polarity: compare mean of darkest 5% vs brightest 5%
    v_sorted = np.sort(vals)
    dark_mean = float(v_sorted[: int(0.05 * len(v_sorted))].mean())
    bright_mean = float(v_sorted[int(0.95 * len(v_sorted)) :].mean())
    if dark_mean > bright_mean:  # invert so nails become peaks after processing
        vals = 1.0 - vals
    # remove low-frequency bias
    vals = vals - vals.mean()
    # smooth angular noise
    if smooth_sigma > 0:
        k = int(smooth_sigma * 6) | 1
        g = cv2.getGaussianKernel(k, smooth_sigma)
        g = (g / g.sum()).astype(np.float32).ravel()
        vals = np.convolve(vals, g, mode="same")
    spectrum = np.abs(fft.rfft(vals))
    freqs = np.arange(spectrum.size)  # since d=1, index==frequency bin
    # restrict plausible range
    lo = max(2, n_min)
    hi = min(n_max, freqs[-1])
    if lo >= hi:
        return max(lo, 2)
    spec_range = spectrum.copy()
    # harmonic suppression: divide by a local median-smoothed spectrum
    if suppress_harmonics:
        win = 9
        half = win // 2
        med = np.zeros_like(spec_range)
        for i in range(spec_range.size):
            a = max(0, i - half)
            b = min(spec_range.size, i + half + 1)
            med[i] = np.median(spec_range[a:b])
        med = np.maximum(med, 1e-6)
        spec_norm = spec_range / med
    else:
        spec_norm = spec_range
    search = spec_norm.copy()
    search[: lo] = 0.0
    search[hi + 1 :] = 0.0
    # penalize obvious multiples to avoid picking 2x actual nails
    for k in range(lo, hi + 1):
        if 2 * k < search.size:
            if search[2 * k] > search[k]:
                search[2 * k] *= 0.75  # damp second harmonic
        if 3 * k < search.size:
            search[3 * k] *= 0.85
    k_best = int(np.argmax(search))
    peak_val = search[k_best]
    # Consider half-frequency if even (possible double counting)
    if k_best % 2 == 0 and (k_best // 2) >= lo:
        k_half = k_best // 2
        # heuristic: if half bin has at least 55% of amplitude after normalization, prefer half
        if search[k_half] >= 0.55 * peak_val:
            k_best = k_half
            peak_val = search[k_best]
    # peak prominence check: ensure local dominance
    neigh = search[max(lo, k_best - 5) : min(hi, k_best + 6)]
    if peak_val < 1.12 * (np.median(neigh) + 1e-9):
        top_idx = np.argsort(search[lo : hi + 1])[-5:] + lo
        weights = search[top_idx]
        k_weighted = int(np.round(np.sum(top_idx * weights) / (np.sum(weights) + 1e-9)))
        if abs(k_weighted - k_best) > 0:  # adopt weighted if different and plausible
            k_best = k_weighted
    # snap to nearest multiple of 4 if within 2%
    k4 = int(round(k_best / 4.0) * 4)
    if k4 >= lo and k4 <= hi and abs(k4 - k_best) / max(1, k_best) < 0.02:
        k_best = k4
    k_best = int(np.clip(k_best, lo, hi))
    if debug_dir:
        try:
            import matplotlib.pyplot as plt
            import os
            os.makedirs(debug_dir, exist_ok=True)
            fig, axs = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
            axs[0].plot(vals, lw=0.8)
            axs[0].set_title("Angular ring signal (smoothed)")
            axs[1].plot(spectrum, lw=0.8)
            axs[1].axvline(k_best, color='r', ls='--'); axs[1].set_title(f"Raw spectrum | chosen k={k_best}")
            axs[2].plot(search, lw=0.8)
            axs[2].axvline(k_best, color='r', ls='--'); axs[2].set_title("Search (norm + suppression)")
            for ax in axs: ax.set_xlim(lo - 5, hi + 5)
            fig.savefig(os.path.join(debug_dir, "nail_detect_debug.png"), dpi=140)
            plt.close(fig)
        except Exception:
            pass
    return k_best
