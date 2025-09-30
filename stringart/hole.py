from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class HoleEstimate:
    frac: float  # inner hole radius / outer radius
    mode: str    # 'skip' or 'dampen'
    confidence: float

def estimate_inner_hole(target: np.ndarray, outer_radius: int, center: tuple[int,int] | None = None,
                        min_frac: float = 0.03, max_frac: float = 0.35,
                        smooth: int = 5) -> HoleEstimate:
    """Estimate inner hole radius by radial profile.

    Steps:
      1. Compute radial mean intensity.
      2. Find first significant valley (low mean region) followed by rise.
      3. Validate contrast ratio and set mode based on depth.
    Returns a HoleEstimate; if no hole, frac=0.0.
    """
    h, w = target.shape
    if center is None:
        cx = cy = h // 2
    else:
        cx, cy = center

    yy, xx = np.indices(target.shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = float(outer_radius)
    mask_outer = rr <= rmax
    vals = target[mask_outer]
    rvals = rr[mask_outer]

    bins = np.linspace(0, rmax, 512)
    idx = np.digitize(rvals, bins) - 1
    prof = np.zeros(len(bins), dtype=np.float32)
    counts = np.bincount(idx, minlength=len(bins)) + 1e-6
    for i in range(len(bins)):
        prof[i] = float(vals[idx == i].mean()) if (idx == i).any() else prof[i-1] if i>0 else 0.0
    if smooth > 1:
        k = smooth if smooth % 2 == 1 else smooth + 1
        prof = cv2.GaussianBlur(prof.reshape(-1,1), (0,0), k*0.15).ravel()

    # Normalize profile
    pmin, pmax = prof.min(), prof.max()
    rng = pmax - pmin + 1e-6
    nprof = (prof - pmin) / rng

    # Search in allowed fraction window
    lo = int(min_frac * len(bins))
    hi = int(max_frac * len(bins))
    if hi - lo < 5:
        return HoleEstimate(0.0, 'skip', 0.0)
    segment = nprof[lo:hi]
    rel_idx = np.argmin(segment)
    min_val = float(segment[rel_idx])
    radius_est = bins[lo + rel_idx]

    # Check valley depth vs outer ring average (take region near 0.5*rmax as baseline)
    mid_band = nprof[int(0.45*len(bins)):int(0.55*len(bins))].mean()
    depth = mid_band - min_val
    confidence = max(0.0, min(1.0, depth * 2.0))  # simple scaling
    if depth < 0.08:  # too shallow
        return HoleEstimate(0.0, 'skip', confidence)

    frac = float(radius_est / rmax)
    mode = 'skip' if depth > 0.15 else 'dampen'
    return HoleEstimate(frac=frac, mode=mode, confidence=confidence)

def auto_adjust_hole(params, estimate: HoleEstimate):
    if estimate.frac <= 0.0:
        return False
    params.inner_hole_frac = estimate.frac
    params.inner_hole_mode = estimate.mode
    if estimate.mode == 'dampen' and params.inner_hole_dampen >= 0.3:
        params.inner_hole_dampen = 0.4
    return True
