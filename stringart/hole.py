from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class HoleEstimate:
    """Detected inner void information.

    Attributes:
        radius_fraction:   Estimated inner hole radius divided by board radius.
        mode:              Suggested handling strategy: 'skip' or 'dampen'.
        confidence:        Heuristic [0,1] confidence score in the detection.
    """
    radius_fraction: float
    mode: str
    confidence: float

def estimate_inner_hole(
    target_image: np.ndarray,
    outer_radius_px: int,
    center: tuple[int, int] | None = None,
    min_radius_fraction: float = 0.03,
    max_radius_fraction: float = 0.35,
    smooth_window: int = 5,
) -> HoleEstimate:
    """Estimate the central void (if any) using a radial intensity profile.

    Heuristic:
      1. Compute radial mean intensity from center to outer radius.
      2. Search a radius band [min_radius_fraction, max_radius_fraction].
      3. Pick the lowest valley; measure depth relative to mid-ring baseline.
      4. Derive confidence from depth; choose mode 'skip' (fully exclude) if deep
         or 'dampen' (partial penalty) if shallow but present.
    Returns a HoleEstimate with radius_fraction=0 when no convincing hole is found.
    """
    height, width = target_image.shape
    if center is None:
        center_x = center_y = height // 2
    else:
        center_x, center_y = center

    y_coords, x_coords = np.indices(target_image.shape)
    radial_dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    outer_radius_f = float(outer_radius_px)
    inside_mask = radial_dist <= outer_radius_f
    pixel_values = target_image[inside_mask]
    pixel_radii = radial_dist[inside_mask]

    radial_bins = np.linspace(0, outer_radius_f, 512)
    bin_index = np.digitize(pixel_radii, radial_bins) - 1
    radial_profile = np.zeros(len(radial_bins), dtype=np.float32)
    for i in range(len(radial_bins)):
        sel = bin_index == i
        if sel.any():
            radial_profile[i] = float(pixel_values[sel].mean())
        elif i > 0:
            radial_profile[i] = radial_profile[i - 1]

    if smooth_window > 1:
        gaussian_size = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        radial_profile = cv2.GaussianBlur(radial_profile.reshape(-1, 1), (0, 0), gaussian_size * 0.15).ravel()

    prof_min, prof_max = radial_profile.min(), radial_profile.max()
    profile_range = prof_max - prof_min + 1e-6
    normalized_profile = (radial_profile - prof_min) / profile_range

    lo = int(min_radius_fraction * len(radial_bins))
    hi = int(max_radius_fraction * len(radial_bins))
    if hi - lo < 5:
        return HoleEstimate(0.0, 'skip', 0.0)
    search_segment = normalized_profile[lo:hi]
    valley_rel_index = int(np.argmin(search_segment))
    valley_value = float(search_segment[valley_rel_index])
    valley_radius_px = radial_bins[lo + valley_rel_index]

    mid_ring_baseline = normalized_profile[int(0.45 * len(radial_bins)) : int(0.55 * len(radial_bins))].mean()
    depth = mid_ring_baseline - valley_value
    confidence = max(0.0, min(1.0, depth * 2.0))
    if depth < 0.08:  # shallow -> no reliable hole
        return HoleEstimate(0.0, 'skip', confidence)

    radius_fraction = float(valley_radius_px / outer_radius_f)
    mode = 'skip' if depth > 0.15 else 'dampen'
    return HoleEstimate(radius_fraction=radius_fraction, mode=mode, confidence=confidence)

def auto_adjust_hole(params, estimate: HoleEstimate) -> bool:
    """Apply a HoleEstimate to solver params if a hole was confidently found.

    Returns True if parameters were modified.
    """
    # backward compatibility: params.inner_hole_frac expected by solver
    if estimate.radius_fraction <= 0.0:
        return False
    params.inner_hole_frac = estimate.radius_fraction
    params.inner_hole_mode = estimate.mode
    if estimate.mode == 'dampen' and params.inner_hole_dampen >= 0.3:
        params.inner_hole_dampen = 0.4
    return True
