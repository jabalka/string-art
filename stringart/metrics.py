from __future__ import annotations
import json, os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse_psnr(a: np.ndarray, b: np.ndarray) -> tuple[float,float]:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = float(np.mean((a-b)**2))
    if mse <= 1e-12:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10(1.0 / mse)
    return float(mse), float(psnr)

def save_metrics(path: str, target: np.ndarray, result: np.ndarray):
    m, p = mse_psnr(target, result)
    # skimage.metrics.ssim returns a scalar float for 2D arrays; cast for type checker
    ssim_val = ssim(target, result, data_range=1.0)  # type: ignore[arg-type]
    if isinstance(ssim_val, tuple):  # defensive: unpack first element if legacy tuple
        ssim_val = ssim_val[0]
    s = float(ssim_val)
    data = {"mse": m, "psnr": p, "ssim": s}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data
