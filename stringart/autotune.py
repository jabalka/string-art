from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

from .detect import auto_detect, sample_rim_signal
from .geometry import generate_nails
from .pattern_init import choose_best_k_by_correlation

__all__ = ["AutoConfig", "auto_config"]

@dataclass
class AutoConfig:
    nails: int
    radius: int
    invert: bool
    rim: str                 # "keep" | "feather" | "erode"
    edge_margin: int
    feather: int
    suppress_nail_heads: int
    min_chord_px: float
    candidate_stride: int
    coarse_scale: int
    sample_stride: int
    topk_refine: int
    bootstrap_k: int
    bootstrap_steps: int
    coverage: float
    rel_improve: float
    abs_improve: float
    patience: int
    window: int

def _decide_invert(image_path: str, canvas: int, nails: int, radius: int) -> bool:
    """Pick invert/no-invert by which correlates better with a star lattice."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    # letterbox to square
    h, w = g.shape
    m = max(h, w)
    pad = np.zeros((m, m), dtype=np.float32)
    y0 = (m - h)//2; x0 = (m - w)//2
    pad[y0:y0+h, x0:x0+w] = g
    g = cv2.resize(pad, (canvas, canvas), interpolation=cv2.INTER_AREA)

    small = max(256, canvas // 3)
    nails_xy_small = generate_nails(n=nails, center=(small//2, small//2), radius=int(small*0.45))
    ks = list(range(2, max(3, min(80, nails//3))))

    def score(inv: bool) -> float:
        tgt = (1.0 - g) if inv else g
        tgt_small = cv2.resize(tgt, (small, small), interpolation=cv2.INTER_AREA)
        _, s = choose_best_k_by_correlation(nails, ks, boots=min(nails*2, 800),
                                            target_small=tgt_small, nails_xy_small=nails_xy_small)
        return float(s)

    return score(True) > score(False)

def auto_config(image_path: str, canvas: int) -> AutoConfig:
    """Return fully automatic configuration derived from the image."""
    n0, cx, cy, R, _phi, _pr0 = auto_detect(image_path, canvas=canvas, invert=False)
    inv = _decide_invert(image_path, canvas=canvas, nails=n0, radius=R)

    # Rim shininess with chosen invert
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    h, w = g.shape; m = max(h, w)
    pad = np.zeros((m, m), dtype=np.float32)
    y0 = (m - h)//2; x0 = (m - w)//2
    pad[y0:y0+h, x0:x0+w] = g
    g_sq = cv2.resize(pad, (canvas, canvas), interpolation=cv2.INTER_AREA)

    vals = sample_rim_signal(g_sq, cx, cy, R, invert=inv)
    spec = np.abs(np.fft.rfft(vals))
    k = int(round(n0)); nb = 6
    lo = max(k-nb, 1); hi = min(k+nb, len(spec)-1)
    neigh = np.r_[spec[lo:k], spec[k+1:hi+1]]
    peak_ratio = float(spec[k] / (np.median(neigh) + 1e-9))

    if peak_ratio > 4.0:
        rim = "feather"; edge_margin = max(8, int(canvas*0.012)); feather = max(8, int(canvas*0.010))
        suppress = max(4, min(10, int(canvas*0.008)))
    elif peak_ratio > 2.5:
        rim = "erode"; edge_margin = max(6, int(canvas*0.010)); feather = 0
        suppress = max(3, min(8, int(canvas*0.006)))
    else:
        rim = "keep"; edge_margin = 0; feather = 0; suppress = 0

    min_chord_px = float(max(12, int(canvas * 0.035)))
    candidate_stride = 3 if n0 >= 200 else 2
    coarse_scale = 6 if canvas >= 900 else 4
    sample_stride = 2
    topk_refine = 24 if n0 >= 200 else 16

    # Bootstrap k / steps
    small = max(256, canvas // 3)
    nails_xy_small = generate_nails(n=n0, center=(small//2, small//2), radius=int(small*0.45))
    tgt = (1.0 - g_sq) if inv else g_sq
    tgt_small = cv2.resize(tgt, (small, small), interpolation=cv2.INTER_AREA)
    ks = list(range(2, max(3, min(80, n0//3))))
    best_k, _ = choose_best_k_by_correlation(n0, ks, boots=min(n0*2, 800),
                                             target_small=tgt_small, nails_xy_small=nails_xy_small)
    boots = int(min(800, max(200, n0 * 2)))

    # Auto-stop thresholds by inner contrast
    yy, xx = np.mgrid[0:canvas, 0:canvas]
    r2 = (xx - canvas//2)**2 + (yy - canvas//2)**2
    inner = tgt[r2 <= (R*0.8)**2]
    std = float(np.std(inner))
    if std < 0.10:
        coverage = 0.82; rel_improve = 1e-4; abs_improve = 8e-7
    else:
        coverage = 0.87; rel_improve = 1e-4; abs_improve = 8e-7

    return AutoConfig(
        nails=n0, radius=R, invert=inv, rim=rim,
        edge_margin=edge_margin, feather=feather, suppress_nail_heads=suppress,
        min_chord_px=min_chord_px,
        candidate_stride=candidate_stride, coarse_scale=coarse_scale,
        sample_stride=sample_stride, topk_refine=topk_refine,
        bootstrap_k=best_k, bootstrap_steps=boots,
        coverage=coverage, rel_improve=rel_improve, abs_improve=abs_improve,
        patience=60, window=20
    )
