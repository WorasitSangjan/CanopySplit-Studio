# SPDX-License-Identifier: AGPL-3.0-only

"""
Step 4: per-plant traits from the leaf masks of segment_leaves.py and the
paired RGB + NIR photos (dual Raspberry Pi Camera Module 3: imx708 RGB +
imx708_noir without filter).

Main traits (colour camera): leaf area (pixels and relative to pot box size,
which removes changes in camera distance), VARI, GLI, NGRDI, ExG, damaged
leaf share and damage colour classes (yellow, pink/red, brown) as share of
leaf and as area relative to pot. Supporting (NIR): NIR_rel, relative NDVI.

Radiometric chain (every image, both cameras, every channel):

  1. Linearise: undo the camera's own tone curve from Raspberry Pi's published
     libcamera tuning (pi_imx708_gamma.json), not an assumed sRGB curve.
  2. Align the NIR photo to the RGB photo (SIFT + RANSAC homography; Stamford
     et al. 2023 used the same feature-based approach with ORB).
  3. Standardise the time series: each group (one treatment tag) has a
     reference image (earliest day). Every image is registered to it and a
     per-channel linear gain/offset is fitted on pseudo-invariant pixels
     (non-plant, unclipped, unchanged background; Schott et al. 1988), for the
     RGB and the NIR camera separately. This removes day-to-day exposure /
     white-balance drift of the automatic cameras.
  4. Reflectance scale: the reference image's own white background (bright,
     unsaturated, unclipped) defines 1.0 for every channel, i.e. white-relative
     reflectance (Glenn & Tabb 2019: rel = I / W). Applied once per series, so
     all days of a series share the same scale.
  5. Clean NIR: the NoIR camera has no filter, so each channel = visible + NIR:
         noir_c = f_c * vis_c + (1 - f_c) * NIR
     f_c fitted per image; the median over the run is used (same hardware).
  6. NDVI = (NIR - Red) / (NIR + Red) with K = 1 (a flat white surface reads
     0). Reported as *relative* NDVI: the RGB camera's broad, processed red
     channel and clipped highlights keep values below sensor-grade NDVI.

Usage:
    python extract_traits.py --centers-csv pots.csv --masks-dir masks \
        --images <rgb dir> --nir-dir <nir dir> --out traits.csv --image-summary-out images_qc.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .image_io import read_image
    from .inputs import PHOTO_NAME, group_labels, photo_files, photo_key, read_rows
    from .qc_images import write_nir_check
    from .traits import AREA_UNITS, CM_PER_INCH
    from .radiometry import load_lut, homography, small, small_coverage, pif_candidates, pif_fit, warp_image_and_coverage
else:
    from image_io import read_image
    from inputs import PHOTO_NAME, group_labels, photo_files, photo_key, read_rows
    from qc_images import write_nir_check
    from traits import AREA_UNITS, CM_PER_INCH
    from radiometry import load_lut, homography, small, small_coverage, pif_candidates, pif_fit, warp_image_and_coverage

EPS = 1e-9
MIN_NIR_PIXELS = 50


def parse_args():
    p = argparse.ArgumentParser(description="Per-plant leaf area, colour indices and relative NDVI")
    p.add_argument("--centers-csv", required=True, help="Grid-labelled, pot-only centers CSV")
    p.add_argument("--masks-dir", required=True, help="Per-pot leaf masks from segment_leaves.py")
    p.add_argument("--images", required=True, help="Folder of RGB images")
    p.add_argument("--nir-dir", default=None, help="Folder with paired NIR images (_RGB_ -> _NIR_); omit to skip NDVI")
    p.add_argument("--leak", default=None, help="Fixed visible-leak fractions 'fB,fG,fR' (default: fitted, median over run)")
    p.add_argument("--exg-split", type=float, default=20.0, help="ExG (0-255 scale) above which leaf tissue counts as green, else damaged")
    p.add_argument("--pot-diameter-cm", type=float, default=None,
                   help="Outer top diameter of the pots in centimetres")
    p.add_argument("--area-unit", choices=AREA_UNITS, default="cm", help="Length unit for physical area (squared)")
    p.add_argument("--out", default="traits.csv", help="Per-plant output CSV")
    p.add_argument("--image-summary-out", default=None, help="Per-image QC output CSV")
    p.add_argument("--nir-check-dir", type=Path, help="Save aligned NoIR photos with RGB leaf outlines")
    return p.parse_args()


def white_mask(rgb, nir, leaf_any):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    away = ~cv2.dilate(leaf_any.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    m = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 170) & (rgb.max(axis=2) < 250) & away
    if nir is not None:
        m &= (nir.max(axis=2) < 250) & (nir.mean(axis=2) > 150) & (nir.sum(axis=2) > 0)
    return m


def fit_leak(V, Nn, sample_mask, leaf_mask, rng):
    il = np.flatnonzero((leaf_mask & sample_mask).ravel())
    ia = np.flatnonzero(sample_mask.ravel())
    if len(ia) < 1000:
        return None
    idx = np.concatenate([rng.choice(il, min(60000, len(il)), replace=False) if len(il) else np.array([], int),
                          rng.choice(ia, min(60000, len(ia)), replace=False)])
    v = V.reshape(-1, 3)[idx].astype(np.float64)
    n = Nn.reshape(-1, 3)[idx].astype(np.float64)
    f = np.array([0.3, 0.3, 0.3])
    for _ in range(300):
        N = np.mean((n - f * v) / (1 - f), axis=1)
        residual = v - N[:, None]
        denominator = np.sum(residual ** 2, axis=0)
        if np.any(denominator <= EPS):
            return None
        fn = np.clip(np.sum((n - N[:, None]) * residual, axis=0) / denominator, 0, 0.95)
        if np.abs(fn - f).max() < 1e-6:
            break
        f = fn
    return f


def noir_leaf_pixels(intensity, usable, plant, all_leaves, radius):
    """Keep a clearly brighter leaf sample inside one plant's RGB outline."""
    x, y, w, h = cv2.boundingRect(plant.astype(np.uint8))
    region = np.s_[max(0, y - radius):y + h + radius, max(0, x - radius):x + w + radius]
    sample = intensity[region]
    leaf = plant[region].astype(np.uint8)
    good = usable[region]
    window = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ring = (cv2.dilate(leaf, window) > 0) & ~all_leaves[region] & good
    nearby = sample[ring]
    foreground = sample[(leaf > 0) & good]
    if min(nearby.size, foreground.size) < MIN_NIR_PIXELS:
        return region, np.zeros(leaf.shape, dtype=bool)

    # Compare against the brighter part of nearby background to avoid including light soil.
    background = np.percentile(nearby, 85)
    bright_leaf = np.percentile(foreground, 75)
    contrast = bright_leaf - background
    if contrast < 20:
        return region, np.zeros(leaf.shape, dtype=bool)

    bright = ((leaf > 0) & good & (sample >= background + contrast / 2)).astype(np.uint8)
    # Stay one pixel inside bright regions to avoid averaging mixed leaf and soil at their edges.
    interior = cv2.erode(bright, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return region, interior > 0


def load_centers(path):
    per = defaultdict(list)
    for r in read_rows(path):
        per[r["image"]].append(r)
    return per


def load_masks(image_mask_dir, centers, shape):
    masks = {}
    for c in centers:
        m = read_image(str(image_mask_dir / f"pot_{c['pot_id']}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != shape:
            raise ValueError(f"Missing or incompatible mask for {c['pot_id']} in {image_mask_dir}")
        masks[c["pot_id"]] = m > 0
    return masks


def exif_exposure(path):
    try:
        from PIL import Image
        ex = Image.open(path)._getexif() or {}
        return ex.get(33434), ex.get(34855)  # ExposureTime, ISOSpeedRatings
    except Exception:
        return None, None


class Frame:
    """One image with its aligned NIR and union leaf mask (full resolution)."""
    def __init__(self, name, centers, image_dir, nir_paths, masks_root):
        self.name = name
        self.rgb = read_image(str(image_dir / name))
        if self.rgb is None:
            raise ValueError(f"Cannot read RGB photo: {image_dir / name}")
        h, w = self.rgb.shape[:2]
        self.masks = load_masks(masks_root / Path(name).stem, centers, (h, w))
        self.leaf = np.zeros((h, w), bool)
        for m in self.masks.values():
            self.leaf |= m
        self.nir, self.nir_matches = None, 0
        self.nir_footprint = None
        p = nir_paths.get(photo_key(PHOTO_NAME.fullmatch(name)))
        self.nir_status = "missing" if p is None else "failed"
        if p is not None:
            raw = read_image(str(p))
            if raw is not None:
                H, self.nir_matches = homography(raw, self.rgb)
                if H is not None:
                    self.nir, self.nir_footprint = warp_image_and_coverage(raw, H, (w, h))
                    self.nir_status = "aligned"


def main():
    args = parse_args()
    area_unit = args.area_unit
    leaf_area = f"leaf_area_{area_unit}2"
    damaged_area = f"damaged_area_{area_unit}2"
    leaf_dp = 2 if area_unit == "cm" else 3
    diameter = args.pot_diameter_cm
    if diameter is not None and area_unit == "in":
        diameter /= CM_PER_INCH
    lut = load_lut()
    per_image = load_centers(args.centers_csv)
    image_dir, masks_root = Path(args.images), Path(args.masks_dir)
    if args.nir_check_dir:
        args.nir_check_dir.mkdir(parents=True, exist_ok=True)
    nir_paths = {}
    if args.nir_dir:
        for path in photo_files(Path(args.nir_dir), "NIR"):
            key = photo_key(PHOTO_NAME.fullmatch(path.name))
            if key in nir_paths:
                raise SystemExit(f"Provide only one NIR photo per group and day: {path.name}")
            nir_paths[key] = path
    rng = np.random.default_rng(0)

    # series = treatment tag; reference = earliest day (prefer one with NIR)
    series = defaultdict(list)
    labels = group_labels({*per_image, *(path.name for path in photo_files(image_dir, "RGB"))})
    for name in per_image:
        match = PHOTO_NAME.fullmatch(name)
        if match is None or match["channel"].upper() != "RGB":
            raise SystemExit(f"Unrecognised RGB photo name: {name}")
        series[labels[match["tray"].casefold()]].append(name)
    for names in series.values():
        names.sort(key=lambda name: int(PHOTO_NAME.fullmatch(name)["day"]))

    def make_ref(fr):
        nir_white = white_mask(fr.rgb, fr.nir, fr.leaf) if fr.nir is not None else None
        has_nir_white = nir_white is not None and nir_white.any()
        rgb_white = nir_white if has_nir_white else white_mask(fr.rgb, None, fr.leaf)
        if nir_white is not None:
            nir_white = nir_white & fr.nir_footprint
        has_covered_white = nir_white is not None and nir_white.any()
        R = {"fr": fr, "W_rgb": np.median(lut[fr.rgb[rgb_white]], axis=0) if rgb_white.any() else None,
             "W_nir": np.median(lut[fr.nir[nir_white]], axis=0) if has_covered_white else None,
             "W_rgb_nir": np.median(lut[fr.rgb[nir_white]], axis=0) if has_covered_white else None,
             "s8": small(fr.rgb), "lin_s": small(lut[fr.rgb]),
             "nir_s8": small(fr.nir) if fr.nir is not None else None,
             "nir_lin_s": small(lut[fr.nir]) if fr.nir is not None else None,
             "nir_coverage_s": small_coverage(fr.nir_footprint.astype(np.float32)) if fr.nir is not None else None,
             "leaf_s": small(fr.leaf.astype(np.uint8), nearest=True) > 0}
        print(f"  reference {fr.name}: white reference px {int(rgb_white.sum())}")
        if R["W_rgb"] is None:
            print("  No usable white reference; calibrated colour indices are unavailable.")
        return R

    std, fits, qc_extra = {}, {}, {}
    for tag, names in series.items():
        # Prefer the earliest aligned NIR pair without retaining all earlier frames in memory.
        reference = None
        for name in names:
            if photo_key(PHOTO_NAME.fullmatch(name)) in nir_paths:
                candidate = Frame(name, per_image[name], image_dir, nir_paths, masks_root)
                if candidate.nir is not None:
                    reference = candidate
                    break
        if reference is None:
            reference = Frame(names[0], per_image[names[0]], image_dir, {}, masks_root)
            if args.nir_dir:
                print(f"series {tag}: no aligned NIR pair -> colour traits only")
        print(f"series {tag}:")
        start = reference.name
        R = make_ref(reference)
        segment = 1
        # Process the reference first, then the others in day order.
        for nm in [start] + [n for n in names if n != start]:
            fr = reference if nm == start else Frame(nm, per_image[nm], image_dir, nir_paths, masks_root)
            ref = R["fr"]
            h, w = ref.rgb.shape[:2]
            W_rgb, W_nir = R["W_rgb"], R["W_nir"]
            ref_s8, ref_lin_s, ref_nir_s8, ref_nir_lin_s, ref_leaf_s = R["s8"], R["lin_s"], R["nir_s8"], R["nir_lin_s"], R["leaf_s"]
            ident = (np.ones(3), np.zeros(3))
            g_rgb, g_nir, n_inv, reg = ident, ident, 0, 0
            if fr is not ref:
                Hd, reg = homography(fr.rgb, ref.rgb)
                if Hd is None and (ref.nir is None or fr.nir is not None):
                    # scene changed (tray/camera moved): start a new segment with this image as reference
                    segment += 1
                    print(f"  {nm}: cannot register to {ref.name} ({reg} matches) -> new segment {segment}")
                    R = make_ref(fr)
                    ref = fr
                    W_rgb, W_nir = R["W_rgb"], R["W_nir"]
                elif Hd is None:
                    print(f"  {nm}: could not register to reference and has no NIR -> left unstandardised")
                else:
                    warp = lambda a, fl=cv2.INTER_LINEAR: cv2.warpPerspective(a, Hd, (w, h), flags=fl)
                    cur_s8 = small(warp(fr.rgb))
                    leaf_s = small(warp(fr.leaf.astype(np.uint8), cv2.INTER_NEAREST), nearest=True) > 0
                    ok = pif_candidates(cur_s8, ref_s8, leaf_s | ref_leaf_s)
                    a, b, inv = pif_fit(small(warp(lut[fr.rgb])), ref_lin_s, ok)
                    g_rgb, n_inv = (a, b), int(inv.sum())
                    if fr.nir is not None and ref.nir is not None:
                        cur_n8 = small(warp(fr.nir))
                        cur_coverage = small_coverage(warp(fr.nir_footprint.astype(np.float32)))
                        covered = cur_coverage & R["nir_coverage_s"]
                        okn = pif_candidates(cur_n8, ref_nir_s8, leaf_s | ref_leaf_s, extra_ok=ok & covered)
                        an, bn, _ = pif_fit(small(warp(lut[fr.nir])), ref_nir_lin_s, okn)
                        g_nir = (an, bn)
            std[nm] = {"rgb": g_rgb, "nir": g_nir, "W_rgb": W_rgb, "W_nir": W_nir,
                       "W_rgb_nir": R["W_rgb_nir"], "ref": ref.name}
            qc_extra[nm] = {"series_ref": ref.name, "segment": segment, "reg_matches_to_ref": reg, "pif_px": n_inv, "nir_matches": fr.nir_matches,
                            "rgb_gain": ",".join(f"{x:.3f}" for x in g_rgb[0]), "nir_gain": ",".join(f"{x:.3f}" for x in g_nir[0])}
            # visible-leak fit on standardised half-resolution data
            if fr.nir is not None and W_nir is not None and not args.leak:
                sc = 0.5
                r_s, n_s = cv2.resize(fr.rgb, None, fx=sc, fy=sc), cv2.resize(fr.nir, None, fx=sc, fy=sc)
                coverage = cv2.resize(fr.nir_footprint.astype(np.float32), None, fx=sc, fy=sc)
                l_s = cv2.resize(fr.leaf.astype(np.uint8), None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST) > 0
                V = (g_rgb[0] * lut[r_s] + g_rgb[1]) / R["W_rgb_nir"]
                Nn = (g_nir[0] * lut[n_s] + g_nir[1]) / W_nir
                valid = (coverage == 1.0) & (n_s.sum(2) > 0) & (r_s.max(2) < 250) & (n_s.max(2) < 250)
                f = fit_leak(V, Nn, valid, l_s, rng)
                if f is not None:
                    fits[nm] = f
            print(f"  {nm}: reg {reg}, PIF px {n_inv}, RGB gain {qc_extra[nm]['rgb_gain']}, NIR gain {qc_extra[nm]['nir_gain']}")

    if args.leak:
        leak = np.array([float(x) for x in args.leak.split(",")])
    elif fits:
        leak = np.median(np.array(list(fits.values())), axis=0)
    else:
        leak = None
    if leak is not None:
        print(f"visible-leak fractions (B,G,R) used: {np.round(leak, 3).tolist()}"
              + (f"  per-image IQR: {np.round(np.percentile(np.array(list(fits.values())), [25, 75], axis=0), 3).tolist()}" if fits else ""))

    # ---- traits ----
    rows, img_rows = [], []
    for name, centers in per_image.items():
        if name not in std:
            continue
        fr = Frame(name, centers, image_dir, nir_paths, masks_root)
        s = std[name]
        match = PHOTO_NAME.fullmatch(name)
        day, treatment = int(match["day"]), labels[match["tray"].casefold()]
        V = ((s["rgb"][0] * lut[fr.rgb] + s["rgb"][1]) / s["W_rgb"]
             if s["W_rgb"] is not None else None)
        f8 = fr.rgb.astype(np.float32)
        exg255 = 2 * f8[:, :, 1] - f8[:, :, 2] - f8[:, :, 0]
        # damage colour classes (colour camera): yellow = chlorosis, pink/red =
        # anthocyanin, tan/brown = necrotic-looking. Brown only >= 2 px inside the
        # leaf so that leaf-edge pixels mixed with the dark soil are not counted.
        hsv8 = cv2.cvtColor(fr.rgb, cv2.COLOR_BGR2HSV)
        hh, ss, vv = hsv8[:, :, 0], hsv8[:, :, 1], hsv8[:, :, 2]
        inner = cv2.distanceTransform(fr.leaf.astype(np.uint8), cv2.DIST_L2, 5) >= 2
        dmg_any = exg255 <= args.exg_split
        dmg_cls = {
            "dmg_yellow_frac": dmg_any & (hh >= 18) & (hh <= 35) & (ss >= 60),
            "dmg_pinkred_frac": dmg_any & ((hh >= 150) | (hh < 8)) & (ss >= 45),
            "dmg_brown_frac": dmg_any & (hh >= 8) & (hh < 18) & (ss >= 45) & (vv < 200) & inner,
        }
        NIR = valid_n = nd = None
        if V is not None and fr.nir is not None and leak is not None and s["W_nir"] is not None:
            Nn = (s["nir"][0] * lut[fr.nir] + s["nir"][1]) / s["W_nir"]
            # Keep the original scale for RGB traits and use covered white pixels for paired traits.
            rgb_scale = s["W_rgb"] / s["W_rgb_nir"]
            NIR = np.mean((Nn - (leak * rgb_scale) * V) / (1 - leak), axis=2)
            red = V[:, :, 2] * rgb_scale[2]
            nir_intensity = cv2.cvtColor(fr.nir, cv2.COLOR_BGR2GRAY)
            nir_usable = fr.nir_footprint & np.any(fr.nir, axis=2) & (fr.nir.max(axis=2) < 250)
            valid_n = nir_usable & (fr.rgb.max(axis=2) < 250)
            denominator = NIR + red
            # Negative corrected signals are invalid reflectances for relative NDVI.
            valid_n &= (NIR >= 0) & (red >= 0) & np.isfinite(denominator) & (denominator > EPS)
            nd = np.divide(NIR - red, denominator, out=np.zeros_like(NIR), where=valid_n)
            rejected_nir = np.zeros(fr.leaf.shape, dtype=bool)
            used_nir = np.zeros(fr.leaf.shape, dtype=bool)

        # Physical area per pixel from the known pot diameter and the median pot size in this photo.
        area_per_px = None
        if args.pot_diameter_cm:
            pot_px = np.median([((float(c["x2"]) - float(c["x1"])) + (float(c["y2"]) - float(c["y1"]))) / 2
                                for c in centers])
            area_per_px = (diameter / pot_px) ** 2
        for c in centers:
            pot = c["pot_id"]
            sel = fr.masks[pot]
            npx = int(sel.sum())
            box_area = (float(c["x2"]) - float(c["x1"])) * (float(c["y2"]) - float(c["y1"]))
            row = {"image": name, "day": day, "treatment": treatment, "pot_id": pot,
                   "grid_row": c.get("grid_row", ""), "grid_col": c.get("grid_col", ""),
                   "leaf_area_px": npx, "leaf_area_pot_frac": round(npx / box_area, 5) if box_area > 0 else "",
                   leaf_area: round(npx * area_per_px, leaf_dp) if area_per_px is not None else "",
                   "edge_cut": "yes" if c.get("status") == "edge" else "no"}
            for k in ("R_rel", "G_rel", "B_rel", "ExG", "VARI", "GLI", "NGRDI", "NDVI", "NIR_rel", "nir_valid_frac",
                      "NDVI_green_tissue", "NDVI_damaged_tissue", "damaged_frac", damaged_area, *dmg_cls,
                      *[k.replace("_frac", "_area_pot") for k in dmg_cls]):
                row[k] = ""
            if npx > 0:
                if V is not None:
                    b, g, r = V[sel, 0], V[sel, 1], V[sel, 2]
                    row["R_rel"], row["G_rel"], row["B_rel"] = round(float(r.mean()), 4), round(float(g.mean()), 4), round(float(b.mean()), 4)
                    row["ExG"] = round(float((2 * g - r - b).mean()), 4)
                    row["VARI"] = round(float(((g - r) / (g + r - b + 1e-6)).mean()), 4)
                    row["GLI"] = round(float(((2 * g - r - b) / (2 * g + r + b + 1e-6)).mean()), 4)
                    row["NGRDI"] = round(float(((g - r) / (g + r + 1e-6)).mean()), 4)
                green_t = sel & (exg255 > args.exg_split)
                row["damaged_frac"] = round(1 - green_t.sum() / npx, 4)
                if area_per_px is not None:
                    row[damaged_area] = round((npx - int(green_t.sum())) * area_per_px, leaf_dp + 1)
                for k, cm in dmg_cls.items():
                    row[k] = round(float((sel & cm).sum()) / npx, 4)
                    # absolute area relative to pot size (not diluted by new green growth)
                    row[k.replace("_frac", "_area_pot")] = round(float((sel & cm).sum()) / box_area, 6) if box_area > 0 else ""
                if nd is not None:
                    pot_width = float(c["x2"]) - float(c["x1"])
                    # Scale the nearby background sample to the pot in this image.
                    radius = int(np.clip(round(0.03 * pot_width), 3, 32))
                    region, clear_leaf = noir_leaf_pixels(nir_intensity, nir_usable, sel, fr.leaf, radius)
                    vs = sel & valid_n
                    rejected_nir[region] |= vs[region] & ~clear_leaf
                    vs[region] &= clear_leaf
                    used_nir |= vs
                    valid_pixels = int(vs.sum())
                    row["nir_valid_frac"] = round(valid_pixels / npx, 5)
                    if valid_pixels >= MIN_NIR_PIXELS:
                        row["NDVI"] = round(float(nd[vs].mean()), 4)
                        row["NIR_rel"] = round(float(NIR[vs].mean()), 4)
                    gs, ds = vs & green_t, vs & ~green_t
                    if gs.sum() >= MIN_NIR_PIXELS:
                        row["NDVI_green_tissue"] = round(float(nd[gs].mean()), 4)
                    if ds.sum() >= MIN_NIR_PIXELS:
                        row["NDVI_damaged_tissue"] = round(float(nd[ds].mean()), 4)
            rows.append(row)

        filtered = rejected_nir & ~used_nir if nd is not None else None
        if args.nir_check_dir and fr.nir is not None:
            write_nir_check(args.nir_check_dir / f"{Path(name).stem}_check_noir.jpg", fr.nir, centers, fr.masks, filtered)
        hsv = cv2.cvtColor(fr.rgb, cv2.COLOR_BGR2HSV)
        away = ~cv2.dilate(fr.leaf.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
        soil = (hsv[:, :, 2] < 60) & away
        wm = white_mask(fr.rgb, fr.nir, fr.leaf)
        et, iso = exif_exposure(image_dir / name)
        q = {"image": name, "day": day, "treatment": treatment, **qc_extra[name],
             "nir_alignment": fr.nir_status if args.nir_dir else "disabled",
             "rgb_clipped_frac": round(float((fr.rgb.max(axis=2) >= 250)[away].mean()), 4) if away.any() else "",
             "exposure_s": et, "iso": iso,
             "leak_fit": "" if name not in fits else ",".join(f"{x:.3f}" for x in fits[name])}
        if nd is not None:
            q["nir_filtered_px"] = int(filtered.sum())
            for key, region in (("ndvi_white_ref", wm), ("ndvi_soil", soil), ("ndvi_leaf", used_nir)):
                values = nd[region & valid_n]
                q[key] = round(float(np.median(values)), 4) if values.size else ""
        img_rows.append(q)
        leaf_ndvi, soil_ndvi = q.get("ndvi_leaf"), q.get("ndvi_soil")
        print(f"{name}: NDVI leaf {leaf_ndvi if leaf_ndvi not in (None, '') else 'n/a'} "
              f"soil {soil_ndvi if soil_ndvi not in (None, '') else 'n/a'}")

    fields = ["image", "day", "treatment", "pot_id", "grid_row", "grid_col", "edge_cut", "leaf_area_px", "leaf_area_pot_frac",
              leaf_area, damaged_area,
              "R_rel", "G_rel", "B_rel", "ExG", "VARI", "GLI", "NGRDI", "NDVI", "NIR_rel", "nir_valid_frac",
              "NDVI_green_tissue", "NDVI_damaged_tissue", "damaged_frac",
              "dmg_yellow_frac", "dmg_pinkred_frac", "dmg_brown_frac",
              "dmg_yellow_area_pot", "dmg_pinkred_area_pot", "dmg_brown_area_pot"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    if args.image_summary_out and img_rows:
        keys = []
        for r in img_rows:
            keys += [k for k in r if k not in keys]
        with open(args.image_summary_out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(img_rows)
    print(f"\nDone. {len(rows)} plant rows -> {args.out}")


if __name__ == "__main__":
    main()
