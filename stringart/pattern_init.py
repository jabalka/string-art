from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple

def render_star_pattern(
    nail_count: int,
    hop_k: int,
    num_hops: int,
    canvas_size: int,
    nail_positions,
) -> np.ndarray:
    """Draw a simple star polygon pattern i -> (i + hop_k) over num_hops steps.

    Returns a float32 image in [0,1] useful for correlation-based bootstrapping.
    """
    raster = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    current_index = 0
    for _ in range(num_hops):
        next_index = (current_index + hop_k) % nail_count
        x1, y1 = nail_positions[current_index]
        x2, y2 = nail_positions[next_index]
        cv2.line(
            raster,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            1.0,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        current_index = next_index
    raster = cv2.GaussianBlur(raster, (0, 0), 0.8)  # soften edges
    return np.clip(raster, 0.0, 1.0)

def choose_best_k_by_correlation(
    nail_count: int,
    hop_candidates: List[int],
    bootstrap_steps_count: int,
    target_small: np.ndarray,
    nail_positions_small,
) -> Tuple[int, float]:
    """Pick hop value (k) giving star pattern most correlated with target.

    Uses cosine similarity on a downscaled target to select a promising initial
    star polygon stride for bootstrap instructions.
    """
    best_hop_k, best_score = hop_candidates[0], -1.0
    target_unit = target_small / (np.linalg.norm(target_small) + 1e-6)
    for hop_k in hop_candidates:
        pattern = render_star_pattern(
            nail_count,
            hop_k,
            num_hops=bootstrap_steps_count,
            canvas_size=target_small.shape[0],
            nail_positions=nail_positions_small,
        )
        pattern_unit = pattern / (np.linalg.norm(pattern) + 1e-6)
        score = float((target_unit * pattern_unit).sum())
        if score > best_score:
            best_score = score
            best_hop_k = hop_k
    return best_hop_k, best_score

def bootstrap_steps(nail_count: int, hop_k: int, num_steps: int) -> List[Tuple[int, int]]:
    """Create an initial step sequence (i -> i + hop_k) of given length."""
    steps_out: List[Tuple[int, int]] = []
    current_index = 0
    for _ in range(num_steps):
        next_index = (current_index + hop_k) % nail_count
        steps_out.append((current_index, next_index))
        current_index = next_index
    return steps_out
