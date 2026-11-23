# SPDX-License-Identifier: AGPL-3.0-only

"""
Validation of the radiometric chain and relative NDVI without ground truth.

  1. Standardisation: pseudo-invariant features (PIF; Schott et al. 1988).
     Each day is registered to the tray's first day; on unchanged, unclipped,
     non-plant pixels the median relative difference to the reference is
     reported for raw linear values vs white-standardised values.
  2. Known contrast: within the same plant, NDVI of green tissue vs
     chill-damaged (non-green) tissue (paired Wilcoxon test).
  3. Physical sanity: white reference NDVI (should be ~0), leaf vs soil.
  4. Repeatability: same plant on consecutive days (Spearman rho, median
     absolute change).
  5. Convergent validity: Spearman correlation of NDVI with GLI, VARI, NGRDI.

Usage:
    python validate_ndvi.py --run "chill_low|traits.csv|images_qc.csv|images_dir|masks_dir" \
                            --run ... --out validation_report.md
    (fields separated by "|" because Windows paths contain ":")
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from image_io import read_image
from inputs import PHOTO_NAME, photo_files, photo_key, read_rows
from radiometry import (homography, load_lut, pif_candidates, pif_fit, small, small_coverage,
                        warp_image_and_coverage)


def num(x):
    try:
        value = float(x)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def leaf_union(md, shape):
    m = np.zeros(shape, bool)
    masks = list(md.glob("pot_*_mask.png"))
    if not masks:
        raise FileNotFoundError(f"No plant masks in {md}")
    for p in masks:
        mask = read_image(p, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape:
            raise ValueError(f"Missing or incompatible leaf mask: {p}")
        m |= mask > 0
    return m


def pif_check(images_dir, masks_dir, names, lut, nir_dir=None):
    """Held-out test of the PIF standardisation used in extract_traits.py.
    names: one series sorted by day; first = reference. Gains are fitted on one
    half of the candidate pixels (20-px checkerboard at PIF scale) and the
    error is measured on the other half's invariant pixels."""
    nir_paths = {photo_key(PHOTO_NAME.fullmatch(p.name)): p for p in photo_files(nir_dir, "NIR")} if nir_dir else {}

    def load(nm):
        rgb = read_image(str(images_dir / nm))
        if rgb is None:
            raise OSError(f"Cannot read RGB photo: {images_dir / nm}")
        h, w = rgb.shape[:2]
        leaf = leaf_union(masks_dir / Path(nm).stem, (h, w))
        nir = coverage = None
        path = nir_paths.get(photo_key(PHOTO_NAME.fullmatch(nm)))
        if path is not None:
            raw = read_image(str(path))
            if raw is not None:
                Hn, _ = homography(raw, rgb)
                if Hn is not None:
                    nir, coverage = warp_image_and_coverage(raw, Hn, (w, h))
        return rgb, leaf, nir, coverage

    ref, ref_leaf, ref_nir, ref_coverage = load(names[0])
    h, w = ref.shape[:2]
    ref_s8 = small(ref)
    ref_leaf_s = small(ref_leaf.astype(np.uint8), nearest=True) > 0
    yy, xx = np.mgrid[0:ref_s8.shape[0], 0:ref_s8.shape[1]]
    half = ((yy // 20 + xx // 20) % 2 == 0)
    out = []
    for nm in names[1:]:
        im, leaf, nir, coverage = load(nm)
        H, _ = homography(im, ref)
        if H is None:
            continue
        warp = lambda a, fl=cv2.INTER_LINEAR: cv2.warpPerspective(a, H, (w, h), flags=fl)
        leaf_s = small(warp(leaf.astype(np.uint8), cv2.INTER_NEAREST), nearest=True) > 0
        res = {}
        pairs = [("RGB", warp(im), ref, None)]
        if nir is not None and ref_nir is not None:
            covered = small_coverage(warp(coverage.astype(np.float32))) & small_coverage(ref_coverage.astype(np.float32))
            pairs.append(("NIR", warp(nir), ref_nir, covered))
        ok_rgb = None
        for cam, cur8, r8, covered in pairs:
            cur_s8, r_s8 = small(cur8), small(r8)
            extra = None if covered is None else covered & ok_rgb
            ok = pif_candidates(cur_s8, r_s8, leaf_s | ref_leaf_s, extra_ok=extra)
            if cam == "RGB":
                ok_rgb = ok
            cur_lin, r_lin = small(lut[cur8]), small(lut[r8])
            a, b, inv = pif_fit(cur_lin, r_lin, ok, fit_sel=half)
            test = (~half[ok]) & inv
            x, y = cur_lin[ok][test], r_lin[ok][test]
            before = np.median(np.abs(x - y) / (y + 1e-9)) if x.size else np.nan
            after = np.median(np.abs((a * x + b) - y) / (y + 1e-9)) if x.size else np.nan
            res[cam] = (100 * float(before), 100 * float(after), int(test.sum()))
        out.append((int(PHOTO_NAME.fullmatch(nm)["day"]), res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name|traits.csv|images_qc.csv|images_dir|masks_dir[|nir_dir]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-pif", action="store_true")
    a = ap.parse_args()
    lut = load_lut()
    md = ["# Relative NDVI validation report", ""]
    for spec in a.run:
        parts = spec.split("|")
        if len(parts) not in (5, 6):
            ap.error("--run expects name|traits.csv|images_qc.csv|images_dir|masks_dir[|nir_dir]")
        name, traits, qc, imgdir, mdir = parts[:5]
        nirdir = Path(parts[5]) if len(parts) > 5 else Path(imgdir)
        rows = read_rows(traits)
        qcs = read_rows(qc)
        md += [f"## {name}", ""]

        # 1. standardisation
        if not a.skip_pif:
            # same references / segments as extract_traits.py (from images_qc.csv)
            by_tag = defaultdict(list)
            for r in qcs:
                ref = r.get("series_ref") or r["image"]
                by_tag[(r["treatment"], ref)].append(r["image"])
            for k, names in by_tag.items():
                names.sort(key=lambda n: (n != k[1], int(PHOTO_NAME.fullmatch(n)["day"])))  # reference first
            md += ["**1. Standardisation: held-out unchanged pixels vs reference day (median relative difference)**", "",
                   "| tray | day | RGB before | RGB after | NIR before | NIR after | test px |", "|---|---|---|---|---|---|---|"]
            for (tag, refname), names in sorted(by_tag.items()):
                for d, res in pif_check(Path(imgdir), Path(mdir), names, lut, nir_dir=nirdir):
                    rb, ra, n = res.get("RGB", (np.nan, np.nan, 0))
                    nb, na, _ = res.get("NIR", (np.nan, np.nan, 0))
                    values = [f"{value:.1f}%" if np.isfinite(value) else "unavailable" for value in (rb, ra, nb, na)]
                    md.append("| " + " | ".join([tag, str(d), *values, str(n)]) + " |")
                    print(f"{name} {tag} day {d}: RGB {values[0]} → {values[1]}, NIR {values[2]} → {values[3]}")
            md.append("")

        # 2. green vs damaged
        pairs = [(num(r.get("NDVI_green_tissue")), num(r.get("NDVI_damaged_tissue"))) for r in rows]
        pairs = [(g, d) for g, d in pairs if g is not None and d is not None]
        if len(pairs) >= 10:
            g, d = np.array(pairs).T
            p = wilcoxon(g, d).pvalue if np.any(g != d) else 1.0
            md += ["**2. Green vs damaged tissue (same plant)**", "",
                   f"- plant-days with both: {len(pairs)}; green higher in {100 * np.mean(g > d):.0f}%",
                   f"- median difference {np.median(g - d):.3f} NDVI; Wilcoxon p = {p:.2g}", ""]
        else:
            md += ["**2. Green vs damaged tissue**: too few plants with both tissue types", ""]

        # 3. physical sanity
        md += ["**3. Physical sanity (median over images)**", ""]
        for column, label in (("ndvi_white_ref", "White reference"), ("ndvi_leaf", "Leaf"), ("ndvi_soil", "Soil cover")):
            values = [value for r in qcs if (value := num(r.get(column))) is not None]
            md.append(f"- {label} NDVI: {np.median(values):.3f}" if values else f"- {label} NDVI: unavailable")
        paired = [(num(r.get("ndvi_leaf")), num(r.get("ndvi_soil"))) for r in qcs]
        comparisons = [leaf > soil for leaf, soil in paired if leaf is not None and soil is not None]
        if comparisons:
            md.append(f"- Leaf above soil in {100 * np.mean(comparisons):.0f}% of {len(comparisons)} paired images")
        md.append("")

        # 4. repeatability
        per = defaultdict(dict)
        for r in rows:
            v = num(r.get("NDVI"))
            if v is not None:
                per[(r["treatment"], r["pot_id"])][int(r["day"])] = v
        x, y = [], []
        for (tag, pot), dd in per.items():
            ds = sorted(dd)
            for d0, d1 in zip(ds, ds[1:]):
                if d1 - d0 == 1:
                    x.append(dd[d0]); y.append(dd[d1])
        if len(x) > 10:
            rho = spearmanr(x, y).correlation
            md += ["**4. Repeatability (same plant, consecutive days)**", "",
                   f"- pairs: {len(x)}; Spearman rho = {rho:.2f}; median |change| = {np.median(np.abs(np.array(y) - np.array(x))):.3f} NDVI", ""]

        # 5. convergent validity
        md += ["**5. Agreement with colour-camera indices (all plant-days)**", ""]
        for idx in ("GLI", "VARI", "NGRDI", "ExG"):
            pr = [(num(r.get("NDVI")), num(r.get(idx))) for r in rows]
            pr = [(u, v) for u, v in pr if u is not None and v is not None]
            if len(pr) > 10:
                md.append(f"- NDVI vs {idx}: Spearman rho = {spearmanr(*np.array(pr).T).correlation:.2f} (n={len(pr)})")
        md.append("")
    Path(a.out).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
