"""
Create a brain mask for aging_exp from the MT reference image MRIm02.dcm.

This version is fixed to the parameters that worked best for you:
  thr_frac     = 0.07
  bridge_break = 1
  close_ksize  = 9

Usage (from repo root):
  python aging_exp/create_mask.py

Outputs:
  aging_exp/data/mask_aging.npy
  aging_exp/data/mask_preview.png
"""

import os
import numpy as np
import cv2
import pydicom as dcm


# ---- Fixed parameters (best found) ----
THR_FRAC = 0.07          # threshold = THR_FRAC * max(M0)
BRIDGE_BREAK = 1         # erosion iterations to break thin bridges
BRIDGE_KERNEL = 3        # erosion kernel size
CLOSE_KSIZE = 9          # morphological closing kernel size
BLUR_SIGMA = 1.0         # small Gaussian blur (denoise)
FILL_HOLES = True        # fill holes inside the brain mask
KEEP_LARGEST_CC = True   # keep only the largest connected component


def keep_largest_component(mask01: np.ndarray) -> np.ndarray:
    mask01 = mask01.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if n <= 1:
        return mask01
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # skip background
    return (labels == largest).astype(np.uint8)


def fill_holes(mask01: np.ndarray) -> np.ndarray:
    mask01 = mask01.astype(np.uint8)
    h, w = mask01.shape
    flood = mask01.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=1)
    holes = (1 - flood) & (1 - mask01)
    return (mask01 | holes).astype(np.uint8)


def build_mask_from_m0(m0: np.ndarray) -> np.ndarray:
    m0 = m0.astype(np.float32)

    # Smooth to reduce speckle
    if BLUR_SIGMA and BLUR_SIGMA > 0:
        m0 = cv2.GaussianBlur(m0, (0, 0), sigmaX=BLUR_SIGMA)

    # Initial threshold
    thr = float(THR_FRAC) * float(np.max(m0))
    mask = (m0 > thr).astype(np.uint8)

    # Break thin bridges to background (erode slightly)
    if BRIDGE_BREAK and BRIDGE_BREAK > 0:
        k = np.ones((BRIDGE_KERNEL, BRIDGE_KERNEL), np.uint8)
        mask = cv2.erode(mask, k, iterations=int(BRIDGE_BREAK))

    # Keep only the brain component after bridges are broken
    if KEEP_LARGEST_CC:
        mask = keep_largest_component(mask)

    # Re-grow and smooth boundary (close)
    if CLOSE_KSIZE and CLOSE_KSIZE > 1:
        k = np.ones((CLOSE_KSIZE, CLOSE_KSIZE), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Closing can reattach junk; keep largest again
    if KEEP_LARGEST_CC:
        mask = keep_largest_component(mask)

    if FILL_HOLES:
        mask = fill_holes(mask)

    if KEEP_LARGEST_CC:
        mask = keep_largest_component(mask)

    return mask.astype(np.uint8)


def save_preview_png(m0: np.ndarray, mask01: np.ndarray, out_png: str) -> None:
    m0n = m0.astype(np.float32)
    m0n = m0n - m0n.min()
    m0n = m0n / (m0n.max() + 1e-8)
    base = (255 * m0n).astype(np.uint8)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(mask01.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (0, 0, 255), 1)  # red contour
    cv2.imwrite(out_png, rgb)


def main():
    base_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(base_dir, "data")

    mt_folder = os.path.join(data_dir, "mt_52")
    m0_path = os.path.join(mt_folder, "MRIm02.dcm")

    out_mask = os.path.join(data_dir, "mask_aging.npy")
    out_preview = os.path.join(data_dir, "mask_preview.png")

    if not os.path.exists(m0_path):
        raise FileNotFoundError(f"Could not find M0 DICOM at: {m0_path}")

    m0 = dcm.dcmread(m0_path).pixel_array.astype(np.float32)
    mask = build_mask_from_m0(m0)

    np.save(out_mask, mask)
    save_preview_png(m0, mask, out_preview)

    print(f"Saved mask to: {out_mask}")
    print(f"Saved preview to: {out_preview}")
    print(f"Mask shape={mask.shape}, foreground pixels={int(mask.sum())}")
    print(f"Params: thr_frac={THR_FRAC}, bridge_break={BRIDGE_BREAK}, close_ksize={CLOSE_KSIZE}")


if __name__ == "__main__":
    main()