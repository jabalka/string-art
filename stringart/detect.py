from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2
import numpy.fft as fft


def _radial_edge_profile(image_gray: np.ndarray, center_x: int, center_y: int, min_radius: int, max_radius: int) -> int:
    """Fallback radius finder using radial gradient energy.

    For each candidate radius we sample points uniformly on the circle, bilinearly
    interpolate the gradient magnitude image, and keep the radius with maximum
    average response. This helps when HoughCircles fails.
    """
    height, width = image_gray.shape
    grad_x = cv2.Sobel(image_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    thetas = np.linspace(-np.pi / 2, 3 * np.pi / 2, 2048, endpoint=False).astype(np.float32)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    best_radius, best_score = min_radius, -1.0
    for radius in range(min_radius, max_radius + 1):
        sample_x = center_x + radius * cos_t
        sample_y = center_y + radius * sin_t
        sample_x = np.clip(sample_x, 0, width - 1 - 1e-3)
        sample_y = np.clip(sample_y, 0, height - 1 - 1e-3)
        x0 = np.floor(sample_x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(sample_y).astype(np.int32)
        y1 = y0 + 1
        wx = sample_x - x0
        wy = sample_y - y0
        Ia = grad_mag[y0, x0]
        Ib = grad_mag[y0, x1]
        Ic = grad_mag[y1, x0]
        Id = grad_mag[y1, x1]
        ring_samples = Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy) + Ic * (1 - wx) * wy + Id * wx * wy
        score = float(ring_samples.mean())
        if score > best_score:
            best_score = score
            best_radius = radius
    return best_radius


def detect_board_circle(image_gray: np.ndarray) -> Tuple[int, int, int]:
    """Detect (center_x, center_y, radius) of the circular board.

    Strategy:
      1. Attempt Hough circle detection on a blurred grayscale copy.
      2. If that fails, fall back to radial gradient energy scanning.
      3. Expand the detected radius slightly (6%) to avoid clipping outer threads.
    Returns integer pixel coordinates and radius.
    """
    height, width = image_gray.shape
    uint8_img = (image_gray * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(uint8_img, (0, 0), 2.0)
    min_side = min(height, width)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=int(min_side * 0.5),
        param1=120,
        param2=40,
        minRadius=int(min_side * 0.35),
        maxRadius=int(min_side * 0.60),
    )
    if circles is not None and len(circles) > 0:
        x_center, y_center, radius = circles[0][0]
        center_x, center_y, board_radius = int(round(x_center)), int(round(y_center)), int(round(radius))
    else:
        center_x = center_y = min_side // 2
        board_radius = _radial_edge_profile(image_gray, center_x, center_y, int(min_side * 0.35), int(min_side * 0.60))

    board_radius = int(round(board_radius * 1.06))  # +6% generosity
    board_radius = max(10, min(board_radius, (min_side // 2) - 2))
    return center_x, center_y, board_radius


def detect_nail_count(
    image_gray: np.ndarray,
    center_x: int,
    center_y: int,
    board_radius: int,
    min_nails: int = 20,
    max_nails: int = 400,
    smooth_sigma: float = 2.0,
    suppress_harmonics: bool = True,
    debug_dir: Optional[str] = None,
) -> int:
    """Estimate the number of nails by analyzing intensity variation along the rim.

    Method summary:
      * Sample grayscale values on a ring slightly inside the detected board radius.
      * Normalize polarity so nail heads become peaks (invert if needed).
      * Remove DC component, optionally smooth.
      * FFT -> pick strongest frequency after harmonic suppression & prominence checks.
      * Guard against double-frequency (e.g., 36 misread as 72) by checking half bin.
      * Snap to nearest multiple of 4 if close (typical board layout).
    """
    angular_samples = 4096
    thetas = np.linspace(0, 2 * np.pi, angular_samples, endpoint=False)
    sampling_radius = max(5, board_radius - 5)
    sample_x = center_x + sampling_radius * np.cos(thetas)
    sample_y = center_y + sampling_radius * np.sin(thetas)
    sample_x = np.clip(sample_x, 0, image_gray.shape[1] - 1)
    sample_y = np.clip(sample_y, 0, image_gray.shape[0] - 1)
    ring_values = image_gray[sample_y.astype(int), sample_x.astype(int)].astype(np.float32)

    sorted_vals = np.sort(ring_values)
    darkest_mean = float(sorted_vals[: int(0.05 * len(sorted_vals))].mean())
    brightest_mean = float(sorted_vals[int(0.95 * len(sorted_vals)) :].mean())
    if darkest_mean > brightest_mean:  # invert so nail heads are peaks
        ring_values = 1.0 - ring_values
    ring_values = ring_values - ring_values.mean()

    if smooth_sigma > 0:
        kernel = int(smooth_sigma * 6) | 1
        gk = cv2.getGaussianKernel(kernel, smooth_sigma)
        gk = (gk / gk.sum()).astype(np.float32).ravel()
        ring_values = np.convolve(ring_values, gk, mode="same")

    spectrum = np.abs(fft.rfft(ring_values))
    frequency_bins = np.arange(spectrum.size)
    low = max(2, min_nails)
    high = min(max_nails, frequency_bins[-1])
    if low >= high:
        return max(low, 2)
    raw_spectrum = spectrum.copy()
    if suppress_harmonics:
        window = 9
        half_window = window // 2
        local_median = np.zeros_like(raw_spectrum)
        for idx in range(raw_spectrum.size):
            a = max(0, idx - half_window)
            b = min(raw_spectrum.size, idx + half_window + 1)
            local_median[idx] = np.median(raw_spectrum[a:b])
        local_median = np.maximum(local_median, 1e-6)
        spectrum_norm = raw_spectrum / local_median
    else:
        spectrum_norm = raw_spectrum

    search_space = spectrum_norm.copy()
    search_space[: low] = 0.0
    search_space[high + 1 :] = 0.0
    for k in range(low, high + 1):
        if 2 * k < search_space.size and search_space[2 * k] > search_space[k]:
            search_space[2 * k] *= 0.75
        if 3 * k < search_space.size:
            search_space[3 * k] *= 0.85
    best_k = int(np.argmax(search_space))
    best_amp = search_space[best_k]
    if best_k % 2 == 0 and (best_k // 2) >= low:
        half_k = best_k // 2
        if search_space[half_k] >= 0.55 * best_amp:
            best_k = half_k
            best_amp = search_space[best_k]
    neighborhood = search_space[max(low, best_k - 5) : min(high, best_k + 6)]
    if best_amp < 1.12 * (np.median(neighborhood) + 1e-9):
        top_indices = np.argsort(search_space[low : high + 1])[-5:] + low
        weights = search_space[top_indices]
        weighted = int(np.round(np.sum(top_indices * weights) / (np.sum(weights) + 1e-9)))
        if abs(weighted - best_k) > 0:
            best_k = weighted
    nearest_multiple_4 = int(round(best_k / 4.0) * 4)
    if nearest_multiple_4 >= low and nearest_multiple_4 <= high and abs(nearest_multiple_4 - best_k) / max(1, best_k) < 0.02:
        best_k = nearest_multiple_4
    best_k = int(np.clip(best_k, low, high))
    if debug_dir:
        try:
            import matplotlib.pyplot as plt
            import os
            os.makedirs(debug_dir, exist_ok=True)
            fig, axs = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
            axs[0].plot(ring_values, lw=0.8); axs[0].set_title("Angular ring signal (smoothed)")
            axs[1].plot(spectrum, lw=0.8); axs[1].axvline(best_k, color='r', ls='--'); axs[1].set_title(f"Raw spectrum | chosen k={best_k}")
            axs[2].plot(search_space, lw=0.8); axs[2].axvline(best_k, color='r', ls='--'); axs[2].set_title("Search (norm + suppression)")
            for ax in axs: ax.set_xlim(low - 5, high + 5)
            fig.savefig(os.path.join(debug_dir, "nail_detect_debug.png"), dpi=140)
            plt.close(fig)
        except Exception:
            pass
    return best_k
