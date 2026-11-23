# SPDX-License-Identifier: AGPL-3.0-only

"""
Shared radiometric helpers for extract_traits.py and validate_ndvi.py.

- Camera tone curve: Raspberry Pi libcamera imx708 gamma curve (pi_imx708_gamma.json).
- Registration: SIFT + RANSAC homography.
- Time-series standardisation with pseudo-invariant features (PIF; Schott et
  al. 1988): per-channel linear gain/offset (in linear light) mapping an image
  onto its series' reference image, fitted robustly on non-plant, unclipped,
  low-texture pixels; pixels that changed are dropped iteratively (residual >
  2.5 robust SD).
"""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PIF_SCALE = 0.25
PIF_BLUR_SIZE = (9, 9)


def load_lut(path=HERE / "pi_imx708_gamma.json"):
    with path.open(encoding="utf-8") as file:
        g = np.array(json.load(file)["gamma_curve"], float).reshape(-1, 2) / 65535.0
    return np.interp(np.arange(256) / 255.0, g[:, 1], g[:, 0]).astype(np.float32)


def homography(src_bgr, dst_bgr, sc=0.5, min_inliers=200):
    """Homography mapping src onto dst, and the RANSAC inlier count."""
    def prep(im):
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    a = cv2.resize(prep(dst_bgr), None, fx=sc, fy=sc)
    b = cv2.resize(prep(src_bgr), None, fx=sc, fy=sc)
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        return None, 0
    good = [m[0] for m in cv2.BFMatcher().knnMatch(db, da, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 8:
        return None, 0
    H, inl = cv2.findHomography(np.float32([kb[g.queryIdx].pt for g in good]) / sc,
                                np.float32([ka[g.trainIdx].pt for g in good]) / sc, cv2.RANSAC, 8.0)
    n = int(inl.sum()) if inl is not None else 0
    return (H if H is not None and n >= min_inliers else None), n


def small(img, nearest=False):
    if nearest:
        return cv2.resize(img, None, fx=PIF_SCALE, fy=PIF_SCALE, interpolation=cv2.INTER_NEAREST)
    r = cv2.resize(img, None, fx=PIF_SCALE, fy=PIF_SCALE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(r, PIF_BLUR_SIZE, 3)


def warp_image_and_coverage(image, transform, size):
    """Warp an image and record which samples are fully inside its original frame."""
    warped = cv2.warpPerspective(image, transform, size)
    coverage = cv2.warpPerspective(np.ones(image.shape[:2], np.float32), transform, size)
    return warped, coverage == 1.0


def small_coverage(coverage):
    """Identify PIF samples made entirely from covered pixels, including the blur window."""
    resized = cv2.resize(coverage, None, fx=PIF_SCALE, fy=PIF_SCALE, interpolation=cv2.INTER_AREA)
    covered = (resized == 1.0).astype(np.uint8)
    return cv2.erode(covered, np.ones(PIF_BLUR_SIZE, np.uint8)) > 0


def pif_candidates(cur8_s, ref8_s, plant_s, extra_ok=None):
    """Non-plant, unclipped, not-too-dark, low-texture pixels (all at PIF scale)."""
    plant = cv2.dilate(plant_s.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(ref8_s, cv2.COLOR_BGR2GRAY), cv2.CV_32F))
    ok = (~plant & (cur8_s.max(2) < 245) & (ref8_s.max(2) < 245) & (cur8_s.mean(2) > 15)
          & (ref8_s.mean(2) > 15) & (cur8_s.sum(2) > 0) & (lap < 12))
    if extra_ok is not None:
        ok &= extra_ok
    return ok


def pif_fit(cur_lin_s, ref_lin_s, ok, fit_sel=None):
    """Robust per-channel fit ref = a*cur + b. Returns (a[3], b[3], invariant mask over ok pixels)."""
    a_out, b_out = np.ones(3), np.zeros(3)
    inv_all = np.zeros(ok.sum(), bool)
    fsel = np.ones(ok.sum(), bool) if fit_sel is None else fit_sel[ok]
    for c in range(3):
        x = cur_lin_s[:, :, c][ok].astype(np.float64)
        y = ref_lin_s[:, :, c][ok].astype(np.float64)
        keep = fsel.copy()
        a, b, thr = 1.0, 0.0, None
        for _ in range(6):
            if keep.sum() < 500:
                break
            a, b = np.polyfit(x[keep], y[keep], 1)
            r = y - (a * x + b)
            mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
            thr = 2.5 * 1.4826 * mad + 0.005
            keep = fsel & (np.abs(r) < thr)
        a_out[c], b_out[c] = a, b
        inv_all |= (np.abs(y - (a * x + b)) < thr) if thr is not None else False
    return a_out, b_out, inv_all
