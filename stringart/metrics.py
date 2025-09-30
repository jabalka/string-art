from __future__ import annotations
import json, os
import numpy as np
from skimage.metrics import structural_similarity as structural_ssim

def mse_psnr(reference: np.ndarray, reconstruction: np.ndarray) -> tuple[float, float]:
    """Compute mean squared error and PSNR between two normalized images.

    Both inputs are expected to be in [0,1]. PSNR is infinite when images are
    identical (within floating epsilon).
    """
    reference = reference.astype(np.float32)
    reconstruction = reconstruction.astype(np.float32)
    mse_value = float(np.mean((reference - reconstruction) ** 2))
    if mse_value <= 1e-12:
        psnr_value = float("inf")
    else:
        psnr_value = 10.0 * np.log10(1.0 / mse_value)
    return float(mse_value), float(psnr_value)

def save_metrics(path: str, target_image: np.ndarray, result_image: np.ndarray):
    """Compute MSE, PSNR, SSIM and persist them to JSON.

    Returns a dict: {"mse": float, "psnr": float, "ssim": float}.
    """
    mse_value, psnr_value = mse_psnr(target_image, result_image)
    ssim_value = structural_ssim(target_image, result_image, data_range=1.0)  # type: ignore[arg-type]
    if isinstance(ssim_value, tuple):  # extremely old versions may return (value, _)
        ssim_value = ssim_value[0]
    metrics_dict = {"mse": float(mse_value), "psnr": float(psnr_value), "ssim": float(ssim_value)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(metrics_dict, json_file, indent=2)
    return metrics_dict
