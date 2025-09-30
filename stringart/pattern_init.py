from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple

def render_star_pattern(n: int, k: int, steps: int, canvas: int, nails_xy) -> np.ndarray:
    """
    Draw a star polygon i -> i+k for 'steps' hops on a blank canvas. Returns float32 [0..1].
    """
    img = np.zeros((canvas, canvas), dtype=np.float32)
    i = 0
    for _ in range(steps):
        j = (i + k) % n
        x1, y1 = nails_xy[i]; x2, y2 = nails_xy[j]
        cv2.line(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 1.0, thickness=1, lineType=cv2.LINE_AA)
        i = j
    img = cv2.GaussianBlur(img, (0,0), 0.8)  # soften
    img = np.clip(img, 0.0, 1.0)
    return img

def choose_best_k_by_correlation(n: int, ks: List[int], boots: int, target_small: np.ndarray, nails_xy_small) -> Tuple[int, float]:
    best_k, best_score = ks[0], -1.0
    t = target_small / (np.linalg.norm(target_small)+1e-6)
    for k in ks:
        pat = render_star_pattern(n, k, steps=boots, canvas=target_small.shape[0], nails_xy=nails_xy_small)
        p = pat / (np.linalg.norm(pat)+1e-6)
        score = float((t * p).sum())   # cosine similarity
        if score > best_score:
            best_score = score; best_k = k
    return best_k, best_score

def bootstrap_steps(n: int, k: int, boots: int) -> List[Tuple[int,int]]:
    """
    Produce an initial (i->i+k) step list of length 'boots'.
    """
    steps_out: List[Tuple[int,int]] = []
    i = 0
    for _ in range(boots):
        j = (i + k) % n
        steps_out.append((i, j))
        i = j
    return steps_out
