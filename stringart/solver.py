from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from .geometry import generate_nails, index_to_label
from .visualize import render_nail_map
from .metrics import save_metrics
from .chords import get_chord_mask

@dataclass
class SolverParams:
    """Configuration for the greedy string art solver and refinement loop.

    Public fields retained for backward compatibility with existing CLI & code.
    """
    # Drawing / rendering model
    alpha: float = 0.25                       # per-chord additive intensity factor
    line_thickness_px: int = 1                # chord thickness in pixels
    blur_sigma: float = 0.0                   # optional Gaussian blur for chord mask
    min_chord_px: float = 15.0                # minimum Euclidean nail distance to consider
    per_nail_max_hits: int = 40               # cap on how many times a nail can be used

    # Canvas & nail layout
    canvas: int = 1200                        # square canvas size (pixels)
    radius: int = 540                         # circle radius used for nails
    nails: int = 240                          # number of nails
    start_label: str = "A1"                   # starting nail label
    invert: bool = False                      # whether the input target was inverted
    apply_mask: bool = True                   # whether circular mask applied in preprocessing
    seed: int = 1                             # RNG seed for any stochastic aspects

    # Progress / preview
    save_every: int = 200                     # save preview every N steps (0 disables)

    # Performance tuning (coarse to fine candidate scoring)
    candidate_stride: int = 2                 # stride over candidate nails for coarse scoring
    coarse_scale: int = 4                     # downscale factor for coarse residual map
    sample_stride: int = 2                    # stride when sampling along chord at coarse level
    topk_refine: int = 20                     # number of coarse top candidates to refine full-res

    # Automatic stopping controls
    auto_steps: bool = True
    max_steps: int = 4000
    coverage: float = 0.85                    # stop once this residual coverage reached
    rel_improve: float = 1e-4                 # relative improvement threshold
    abs_improve: float = 1e-6                 # absolute improvement threshold
    patience: int = 50                        # consecutive low-improve windows before stopping
    window: int = 20                          # length of recent gains window

    # Path aesthetics & smoothing
    endpoint_taper: float = 0.20              # taper chord ends (blend near nails)
    angle_smooth: float = 0.25                # weight encouraging smoother direction changes

    # Optional: stop if average nail usage reaches this value
    avg_hits_stop: float | None = None

    # Inner hole handling (auto-detected or user-provided)
    inner_hole_frac: float = 0.0              # fraction of radius considered a void
    inner_hole_mode: str = "skip"             # 'skip' chords or 'dampen' their score
    inner_hole_dampen: float = 0.3            # multiplier if mode == 'dampen'

    # Refinement loop controls
    refine_rounds: int = 0                    # number of refinement passes after baseline
    refine_mse_thresh: float = 0.012          # stop refining once MSE <= threshold
    refine_alpha_scale: float = 0.7           # alpha decay per refinement round


def _euclidean_distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    """Return Euclidean distance between two 2D points."""
    return float(((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5)


def _coarse_residual_score(
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    residual_coarse: np.ndarray,
    downscale: float,
    stride: int,
) -> float:
    """Approximate chord benefit by sampling along a downscaled residual image."""
    x1, y1 = start_xy
    x2, y2 = end_xy
    x1c, y1c = (x1 / downscale), (y1 / downscale)
    x2c, y2c = (x2 / downscale), (y2 / downscale)
    dx, dy = (x2c - x1c), (y2c - y1c)
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    sample_count = int(length / stride) + 1
    if sample_count <= 1:
        return 0.0
    ts = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    xs = x1c + ts * dx
    ys = y1c + ts * dy
    height, width = residual_coarse.shape
    xs = np.clip(xs, 0, width - 1 - 1e-3)
    ys = np.clip(ys, 0, height - 1 - 1e-3)
    x0 = np.floor(xs).astype(np.int32)
    x1i = x0 + 1
    y0 = np.floor(ys).astype(np.int32)
    y1i = y0 + 1
    wx = xs - x0
    wy = ys - y0
    Ia = residual_coarse[y0, x0]
    Ib = residual_coarse[y0, x1i]
    Ic = residual_coarse[y1i, x0]
    Id = residual_coarse[y1i, x1i]
    interpolated = Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy) + Ic * (1 - wx) * wy + Id * wx * wy
    return float(np.sum(interpolated))


def save_overlay(outdir: str, target_image: np.ndarray, result_image: np.ndarray):
    """Save side-by-side (target, reconstruction, residual) preview."""
    canvas_size = target_image.shape[0]
    residual_image = np.clip(target_image - result_image, 0.0, 1.0)
    target_rgb = Image.fromarray(np.uint8(target_image * 255), mode="L").convert("RGB")
    result_rgb = Image.fromarray(np.uint8(result_image * 255), mode="L").convert("RGB")
    residual_rgb = Image.fromarray(np.uint8(residual_image * 255), mode="L").convert("RGB")
    combined = Image.new("RGB", (canvas_size * 3, canvas_size), (255, 255, 255))
    combined.paste(target_rgb, (0, 0))
    combined.paste(result_rgb, (canvas_size, 0))
    combined.paste(residual_rgb, (canvas_size * 2, 0))
    combined.save(os.path.join(outdir, "preview_overlay.png"))


def save_result_preview(outdir: str, result_image: np.ndarray, invert_preview: bool = False):
    """Save the current reconstruction image (optionally inverted for viewing)."""
    display = 1.0 - result_image if invert_preview else result_image
    Image.fromarray(np.uint8(np.clip(display, 0, 1) * 255), mode="L").convert("RGB").save(
        os.path.join(outdir, "preview_final.png")
    )


def write_instructions(outdir: str, chord_steps: List[Tuple[int, int]], nail_count: int):
    """Persist chord sequence as CSV (with labels) and plain text."""
    import csv
    csv_path = os.path.join(outdir, "instructions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "from_label", "to_label", "from_index", "to_index"])
        for step_number, (nail_a, nail_b) in enumerate(chord_steps, start=1):
            writer.writerow([
                step_number,
                index_to_label(nail_a, nail_count),
                index_to_label(nail_b, nail_count),
                nail_a,
                nail_b,
            ])
    txt_path = os.path.join(outdir, "instructions.txt")
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        for nail_a, nail_b in chord_steps:
            txt_file.write(f"{index_to_label(nail_a, nail_count)}-{index_to_label(nail_b, nail_count)}\n")
    return csv_path, txt_path


def greedy_solver(
    target_image: np.ndarray,
    params: SolverParams,
    outdir: str,
    init_steps: Optional[List[Tuple[int, int]]] = None,
    initial_result: Optional[np.ndarray] = None,
    append: bool = False,
):
    """Greedy residual-reduction loop selecting chords that most reduce remaining error.

    Returns (steps, result_image, residual_image).
    """
    np.random.seed(params.seed)
    nail_count = params.nails
    canvas_size = params.canvas
    board_radius = params.radius
    center_x = center_y = canvas_size // 2
    nail_positions = tuple(generate_nails(n=nail_count, center=(center_x, center_y), radius=board_radius))

    render_nail_map(
        os.path.join(outdir, "legend_nails.png"),
        canvas=canvas_size,
        radius=board_radius,
        n=nail_count,
        show_labels=True,
        show_indices=False,
    )

    if initial_result is not None:
        result_image = np.clip(initial_result.copy().astype(np.float32), 0.0, 1.0)
        residual_image = np.clip(target_image - result_image, 0.0, 1.0)
        chord_steps: List[Tuple[int, int]] = [] if not append else (init_steps or []).copy()
    else:
        result_image = np.zeros_like(target_image, dtype=np.float32)
        residual_image = target_image.copy()
        chord_steps = []
    nail_usage_counts = np.zeros((nail_count,), dtype=np.int32)

    residual_flat = residual_image.ravel()
    residual_initial_sum = float(residual_flat.sum() + 1e-12)

    current_nail = 0
    last_chosen_pair = (-1, -1)
    previous_direction: Optional[Tuple[float, float]] = None

    stride = params.candidate_stride
    downscale = float(params.coarse_scale)
    residual_coarse = cv2.resize(
        residual_image,
        (canvas_size // params.coarse_scale, canvas_size // params.coarse_scale),
        interpolation=cv2.INTER_AREA,
    )
    min_distance = params.min_chord_px

    recent_gains: List[float] = []
    plateau_streak = 0
    step_limit = params.max_steps
    steps_executed = 0
    with tqdm(total=step_limit, desc="[stringart] greedy-fast") as progress_bar:
        while True:
            if not params.auto_steps and steps_executed >= params.max_steps:
                break

            coarse_scores: List[float] = []
            candidate_indices: List[int] = []
            current_xy = nail_positions[current_nail]
            for candidate in range(0, nail_count, stride):
                if candidate == current_nail:
                    continue
                if nail_usage_counts[current_nail] >= params.per_nail_max_hits or nail_usage_counts[candidate] >= params.per_nail_max_hits:
                    continue
                a_pair = (current_nail, candidate) if current_nail < candidate else (candidate, current_nail)
                if last_chosen_pair == a_pair:
                    continue
                if _euclidean_distance(current_xy, nail_positions[candidate]) < min_distance:
                    continue
                candidate_indices.append(candidate)
            if not candidate_indices:
                break

            for candidate in candidate_indices:
                coarse_scores.append(
                    _coarse_residual_score(
                        current_xy,
                        nail_positions[candidate],
                        residual_coarse,
                        downscale=downscale,
                        stride=params.sample_stride,
                    )
                )
            if len(candidate_indices) > params.topk_refine:
                top_indices = np.argpartition(np.array(coarse_scores), -params.topk_refine)[-params.topk_refine:]
                candidate_indices = [candidate_indices[i] for i in top_indices]

            best_candidate = None
            best_score = -1.0
            best_mask = None
            residual_flat = residual_image.ravel()
            for candidate in candidate_indices:
                flat_indices, weights = get_chord_mask(
                    current_nail,
                    candidate,
                    nail_positions,
                    canvas_size,
                    params.line_thickness_px,
                    params.blur_sigma,
                    params.endpoint_taper,
                )
                if flat_indices.size == 0:
                    continue
                score = float(np.dot(residual_flat[flat_indices], weights))

                if params.inner_hole_frac > 0.0:
                    mid_x = 0.5 * (nail_positions[current_nail][0] + nail_positions[candidate][0]) - center_x
                    mid_y = 0.5 * (nail_positions[current_nail][1] + nail_positions[candidate][1]) - center_y
                    midpoint_radius = (mid_x * mid_x + mid_y * mid_y) ** 0.5
                    if midpoint_radius < params.inner_hole_frac * board_radius:
                        if params.inner_hole_mode == "skip":
                            continue
                        else:
                            score *= params.inner_hole_dampen

                if previous_direction is not None and params.angle_smooth > 0.0:
                    vx_prev, vy_prev = previous_direction
                    vx_new = nail_positions[candidate][0] - nail_positions[current_nail][0]
                    vy_new = nail_positions[candidate][1] - nail_positions[current_nail][1]
                    n_prev = (vx_prev * vx_prev + vy_prev * vy_prev) ** 0.5 + 1e-9
                    n_new = (vx_new * vx_new + vy_new * vy_new) ** 0.5 + 1e-9
                    cos_angle = (vx_prev * vx_new + vy_prev * vy_new) / (n_prev * n_new)
                    smoothing = 0.5 * (1.0 + cos_angle)
                    score *= (1.0 - params.angle_smooth) + params.angle_smooth * smoothing

                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_mask = (flat_indices, weights)

            if best_candidate is None or best_score <= 1e-10:
                break
            if best_mask is None:
                break

            flat_indices, weights = best_mask
            result_flat = result_image.ravel()
            before_sum = float(residual_flat.sum())
            result_flat[flat_indices] = np.clip(result_flat[flat_indices] + params.alpha * weights, 0.0, 1.0)
            result_image = result_flat.reshape(result_image.shape)
            residual_image = np.clip(target_image - result_image, 0.0, 1.0)
            residual_flat = residual_image.ravel()
            after_sum = float(residual_flat.sum())
            gain = max(0.0, before_sum - after_sum)

            chord_steps.append((current_nail, best_candidate))
            nail_usage_counts[current_nail] += 1
            nail_usage_counts[best_candidate] += 1
            last_chosen_pair = (min(current_nail, best_candidate), max(current_nail, best_candidate))
            previous_direction = (
                nail_positions[best_candidate][0] - nail_positions[current_nail][0],
                nail_positions[best_candidate][1] - nail_positions[current_nail][1],
            )
            current_nail = best_candidate
            steps_executed += 1
            progress_bar.update(1)

            if (steps_executed % 10) == 0:
                residual_coarse = cv2.resize(
                    residual_image,
                    (canvas_size // params.coarse_scale, canvas_size // params.coarse_scale),
                    interpolation=cv2.INTER_AREA,
                )

            if params.save_every and (steps_executed % params.save_every == 0):
                save_result_preview(outdir, result_image, invert_preview=False)

            if params.auto_steps:
                coverage_now = 1.0 - (after_sum / residual_initial_sum)
                if coverage_now >= params.coverage:
                    progress_bar.set_postfix_str(f"stop: coverage {coverage_now:.3f} ≥ {params.coverage:.3f}")
                    break
                recent_gains.append(gain)
                if len(recent_gains) > params.window:
                    recent_gains = recent_gains[-params.window:]
                average_gain = float(np.mean(recent_gains))
                relative_threshold = params.rel_improve * residual_initial_sum
                if average_gain <= params.abs_improve and average_gain <= relative_threshold:
                    plateau_streak += 1
                else:
                    plateau_streak = 0
                if plateau_streak >= params.patience:
                    progress_bar.set_postfix_str(f"stop: plateau avg_gain={average_gain:.3e}")
                    break
                if params.avg_hits_stop is not None and np.mean(nail_usage_counts) >= params.avg_hits_stop:
                    progress_bar.set_postfix_str(f"stop: average nail used ≥ {params.avg_hits_stop}")
                    break
                if steps_executed >= params.max_steps:
                    progress_bar.set_postfix_str("stop: max_steps cap")
                    break

    return chord_steps, result_image, residual_image


def solve_with_refinement(target_image: np.ndarray, params: SolverParams, outdir: str):
    """Run baseline greedy solve plus optional refinement passes.

    Returns (all_steps, metrics_dict).
    """
    base_steps, result_image, residual_image = greedy_solver(target_image, params, outdir)
    from .metrics import mse_psnr

    mse_value, _psnr = mse_psnr(target_image, result_image)
    all_steps = base_steps.copy()
    current_alpha = params.alpha
    refinement_index = 0
    while (
        params.refine_rounds > 0
        and refinement_index < params.refine_rounds
        and mse_value > params.refine_mse_thresh
    ):
        refinement_index += 1
        current_alpha *= params.refine_alpha_scale
        refine_params = params
        refine_params.alpha = current_alpha
        refine_params.max_steps = int(params.max_steps * 1.3)
        refine_params.auto_steps = True
        refine_steps, result_image, residual_image = greedy_solver(
            target_image,
            refine_params,
            outdir,
            initial_result=result_image,
            append=True,
        )
        all_steps.extend(refine_steps)
        mse_value, _ = mse_psnr(target_image, result_image)

    write_instructions(outdir, all_steps, nail_count=params.nails)
    save_result_preview(outdir, result_image, invert_preview=False)
    save_overlay(outdir, target_image, result_image)
    metrics_dict = save_metrics(os.path.join(outdir, "metrics.json"), target_image, result_image)
    dump = asdict(params)
    dump["actual_steps"] = len(all_steps)
    with open(os.path.join(outdir, "params.json"), "w", encoding="utf-8") as params_file:
        json.dump(dump, params_file, indent=2)
    return all_steps, metrics_dict
