from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from .geometry import generate_nails, index_to_label, label_to_index
from .visualize import render_nail_map
from .metrics import save_metrics
from .chords import get_chord_mask

# ---------- Params with auto-stopping ----------
@dataclass
class SolverParams:
    # drawing model
    alpha: float = 0.25
    line_thickness_px: int = 1
    blur_sigma: float = 0.0
    min_chord_px: float = 15.0
    per_nail_max_hits: int = 40     # cap to avoid grainy overuse

    # canvas + nails
    canvas: int = 1200
    radius: int = 540
    nails: int = 240
    start_label: str = "A1"
    invert: bool = False
    apply_mask: bool = True
    seed: int = 1

    # progress previews
    save_every: int = 200

    # FAST mode
    candidate_stride: int = 2
    coarse_scale: int = 4
    sample_stride: int = 2
    topk_refine: int = 20

    # --- AUTO-STOPPING ---
    auto_steps: bool = True
    max_steps: int = 4000
    coverage: float = 0.85          # explain this fraction of residual energy
    rel_improve: float = 1e-4
    abs_improve: float = 1e-6
    patience: int = 50
    window: int = 20

    # NEW: endpoint taper & smoothness
    endpoint_taper: float = 0.20    # 0..1, fades each chord end
    angle_smooth: float = 0.25      # 0..1, prefer small direction changes

# ---------- helpers ----------
def _dist(p1: Tuple[float,float], p2: Tuple[float,float]) -> float:
    return float(((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5)

def _score_line_coarse(cur_xy, j_xy, res_coarse: np.ndarray, scale: float, stride: int) -> float:
    x1, y1 = cur_xy; x2, y2 = j_xy
    x1c, y1c = (x1/scale), (y1/scale); x2c, y2c = (x2/scale), (y2/scale)
    dx, dy = (x2c - x1c), (y2c - y1c)
    length = max(1.0, (dx*dx + dy*dy) ** 0.5)
    steps = int(length / stride) + 1
    if steps <= 1: return 0.0
    ts = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    xs = x1c + ts * dx; ys = y1c + ts * dy
    h, w = res_coarse.shape
    xs = np.clip(xs, 0, w - 1 - 1e-3); ys = np.clip(ys, 0, h - 1 - 1e-3)
    x0 = np.floor(xs).astype(np.int32); x1i = x0 + 1
    y0 = np.floor(ys).astype(np.int32); y1i = y0 + 1
    wx = xs - x0; wy = ys - y0
    Ia = res_coarse[y0, x0]; Ib = res_coarse[y0, x1i]
    Ic = res_coarse[y1i, x0]; Id = res_coarse[y1i, x1i]
    vals = (Ia*(1-wx)*(1-wy) + Ib*wx*(1-wy) + Ic*(1-wx)*wy + Id*wx*wy)
    return float(np.sum(vals))

def save_overlay(outdir: str, target: np.ndarray, result: np.ndarray):
    canvas = target.shape[0]
    residual = np.clip(target - result, 0.0, 1.0)
    A = Image.fromarray(np.uint8(target*255), mode="L").convert("RGB")
    B = Image.fromarray(np.uint8(result*255), mode="L").convert("RGB")
    C = Image.fromarray(np.uint8(residual*255), mode="L").convert("RGB")
    combo = Image.new("RGB", (canvas*3, canvas), (255,255,255))
    combo.paste(A, (0,0)); combo.paste(B, (canvas,0)); combo.paste(C, (canvas*2,0))
    combo.save(os.path.join(outdir, "preview_overlay.png"))

def save_result_preview(outdir: str, result: np.ndarray, invert_preview: bool = False):
    arr = 1.0 - result if invert_preview else result
    Image.fromarray(np.uint8(np.clip(arr,0,1)*255), mode="L").convert("RGB").save(
        os.path.join(outdir, "preview_final.png")
    )

def write_instructions(outdir: str, steps_list: List[Tuple[int,int]], n: int):
    import csv
    csv_path = os.path.join(outdir, "instructions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["step","from_label","to_label","from_index","to_index"])
        for k,(a,b) in enumerate(steps_list, start=1):
            w.writerow([k, index_to_label(a, n), index_to_label(b, n), a, b])
    txt_path = os.path.join(outdir, "instructions.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for (a,b) in steps_list:
            f.write(f"{index_to_label(a, n)}-{index_to_label(b, n)}\n")
    return csv_path, txt_path

# ---------- main solver ----------
def greedy_solver(
    target: np.ndarray,
    params: SolverParams,
    outdir: str,
    init_steps: Optional[List[Tuple[int,int]]] = None
):
    np.random.seed(params.seed)
    n = params.nails
    canvas = params.canvas; radius = params.radius; cx = cy = canvas // 2
    nails_xy = generate_nails(n=n, center=(cx,cy), radius=radius)

    render_nail_map(os.path.join(outdir, "legend_nails.png"),
                    canvas=canvas, radius=radius, n=n,
                    show_labels=True, show_indices=False)

    # state
    result = np.zeros_like(target, dtype=np.float32)
    residual = target.copy()
    per_hits = np.zeros((n,), dtype=np.int32)
    steps_out: List[Tuple[int,int]] = []

    # initial residual energy (L1) — matches greedy score
    flat_res = residual.ravel()
    R0 = float(flat_res.sum() + 1e-12)

    # apply bootstrap if given
    cur = 0
    last_pair = (-1,-1)
    prev_vec: Optional[Tuple[float,float]] = None  # ---------- keep track of last direction
    if init_steps:
        flat = result.ravel()
        for (a,b) in init_steps:
            idx, w = get_chord_mask(a, b, tuple(nails_xy), canvas,
                                    params.line_thickness_px, params.blur_sigma,
                                    params.endpoint_taper)
            if idx.size:
                flat[idx] = np.clip(flat[idx] + params.alpha * w, 0.0, 1.0)
                steps_out.append((a,b))
                per_hits[a] += 1; per_hits[b] += 1
                last_pair = (min(a,b), max(a,b))
        result = flat.reshape(result.shape)
        residual = np.clip(target - result, 0.0, 1.0)
        cur = init_steps[-1][1]
        a, b = init_steps[-1]
        x0, y0 = nails_xy[a]; x1, y1 = nails_xy[b]
        prev_vec = (x1 - x0, y1 - y0)                      # ---------- initial prev_vec
        flat_res = residual.ravel()

    # fast-mode helpers
    s = params.candidate_stride
    scale = float(params.coarse_scale)
    res_coarse = cv2.resize(residual, (canvas//params.coarse_scale, canvas//params.coarse_scale),
                            interpolation=cv2.INTER_AREA)
    min_d = params.min_chord_px

    # auto-stop trackers
    recent_gains: List[float] = []
    low_improve_streak = 0
    step_limit = params.max_steps

    # main loop
    num_steps_run = 0
    with tqdm(total=step_limit, desc="[stringart] greedy-fast") as pbar:
        while True:
            if not params.auto_steps and num_steps_run >= params.max_steps:
                break

            # --- candidate list
            coarse_scores = []
            cand_js = []
            cur_xy = nails_xy[cur]
            for j in range(0, n, s):
                if j == cur: continue
                if per_hits[cur] >= params.per_nail_max_hits or per_hits[j] >= params.per_nail_max_hits: continue
                a,b = (cur,j) if cur<j else (j,cur)
                if last_pair == (a,b): continue
                if _dist(cur_xy, nails_xy[j]) < min_d: continue
                cand_js.append(j)
            if not cand_js:
                break

            # --- coarse screening
            for j in cand_js:
                coarse_scores.append(
                    _score_line_coarse(cur_xy, nails_xy[j], res_coarse, scale=scale, stride=params.sample_stride)
                )
            if len(cand_js) > params.topk_refine:
                idxs = np.argpartition(np.array(coarse_scores), -params.topk_refine)[-params.topk_refine:]
                cand_js = [cand_js[i] for i in idxs]

            # --- refine on full-res (with endpoint taper & angle smoothness)
            best_j = None; best_score = -1.0; best_mask = None
            flat_res = residual.ravel()
            for j in cand_js:
                idx, w = get_chord_mask(cur, j, tuple(nails_xy), canvas,
                                        params.line_thickness_px, params.blur_sigma,
                                        params.endpoint_taper)
                if idx.size == 0:
                    continue
                score = float(np.dot(flat_res[idx], w))

                # angle smoothness (prefer small change in direction)
                if prev_vec is not None and params.angle_smooth > 0.0:
                    vx0, vy0 = prev_vec
                    vx1 = nails_xy[j][0] - nails_xy[cur][0]
                    vy1 = nails_xy[j][1] - nails_xy[cur][1]
                    n0 = (vx0*vx0 + vy0*vy0) ** 0.5 + 1e-9
                    n1 = (vx1*vx1 + vy1*vy1) ** 0.5 + 1e-9
                    cosang = (vx0*vx1 + vy0*vy1) / (n0 * n1)  # [-1,1]
                    smooth = 0.5 * (1.0 + cosang)            # [0..1]
                    score *= (1.0 - params.angle_smooth) + params.angle_smooth * smooth

                if score > best_score:
                    best_score = score; best_j = j; best_mask = (idx, w)

            if best_j is None or best_score <= 1e-10:
                break

            # --- apply best
            if best_mask is None:
                break
            idx, w = best_mask
            flat = result.ravel()
            before_sum = float(flat_res.sum())
            flat[idx] = np.clip(flat[idx] + params.alpha * w, 0.0, 1.0)
            result = flat.reshape(result.shape)
            residual = np.clip(target - result, 0.0, 1.0)
            flat_res = residual.ravel()
            after_sum = float(flat_res.sum())
            gain = max(0.0, before_sum - after_sum)

            # update state
            steps_out.append((cur, best_j))
            per_hits[cur] += 1; per_hits[best_j] += 1
            last_pair = (min(cur,best_j), max(cur,best_j))
            # ---------- update prev_vec AFTER using cur→best_j
            prev_vec = (
                nails_xy[best_j][0] - nails_xy[cur][0],
                nails_xy[best_j][1] - nails_xy[cur][1],
            )
            cur = best_j
            num_steps_run += 1
            pbar.update(1)

            # refresh coarse residual occasionally
            if (num_steps_run % 10) == 0:
                res_coarse = cv2.resize(residual, (canvas//params.coarse_scale, canvas//params.coarse_scale),
                                        interpolation=cv2.INTER_AREA)

            if params.save_every and (num_steps_run % params.save_every == 0):
                save_result_preview(outdir, result, invert_preview=False)

            # ---------- AUTO-STOP ----------
            if params.auto_steps:
                coverage_now = 1.0 - (after_sum / R0)
                if coverage_now >= params.coverage:
                    pbar.set_postfix_str(f"stop: coverage {coverage_now:.3f} ≥ {params.coverage:.3f}")
                    break

                # plateau detection
                recent_gains.append(gain)
                if len(recent_gains) > params.window:
                    recent_gains = recent_gains[-params.window:]
                avg_gain = float(np.mean(recent_gains))
                rel_thresh = params.rel_improve * R0
                if avg_gain <= params.abs_improve and avg_gain <= rel_thresh:
                    low_improve_streak += 1
                else:
                    low_improve_streak = 0
                if low_improve_streak >= params.patience:
                    pbar.set_postfix_str(f"stop: plateau avg_gain={avg_gain:.3e}")
                    break

                if num_steps_run >= params.max_steps:
                    pbar.set_postfix_str("stop: max_steps cap")
                    break

    # outputs
    write_instructions(outdir, steps_out, n=n)
    save_result_preview(outdir, result, invert_preview=False)
    save_overlay(outdir, target, result)
    m = save_metrics(os.path.join(outdir, "metrics.json"), target, result)

    dump = asdict(params)
    dump["actual_steps"] = len(steps_out)
    with open(os.path.join(outdir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)

    return steps_out, m
