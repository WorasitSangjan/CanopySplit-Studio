# SPDX-License-Identifier: AGPL-3.0-only

"""
Step 6: one check image per photo, showing the whole result so a person can
check pot detection, pot names and the leaves given to each pot at once.

  - coloured box     = detected pot; label = tray position (R2C5) and the
                       model's certainty (0-1)
  - same colour fill = the leaves given to that pot's plant
  - red label        = plant flagged reliable = no (leaf overlap, needs manual check)
  - "edge" in label  = pot partly outside the photo: measured, leaf area incomplete
  - grey box         = small sliver of a pot at the photo edge (not measured)
  - orange box       = duplicate position (not measured)
  - magenta box      = colour card

Usage:
    python qc_images.py --images photos/ --grid grid.csv --masks-dir masks/ \
        --overlap overlap.csv --out-dir qc_images/
"""

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
if __package__:
    from .image_io import read_image, write_image
    from .inputs import read_rows
else:
    from image_io import read_image, write_image
    from inputs import read_rows


CHECK_WIDTH = 1800


def parse_args():
    p = argparse.ArgumentParser(description="Per-photo check images")
    p.add_argument("--images", required=True)
    p.add_argument("--grid", required=True, help="grid.csv from assign_grid.py")
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--overlap", default=None, help="overlap.csv from overlap_score.py")
    p.add_argument("--out-dir", default="qc_images")
    p.add_argument("--width", type=int, default=CHECK_WIDTH, help="Width of the saved check image")
    return p.parse_args()


def colour(i):
    hue = int(180 * ((i * 0.381966) % 1))
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def label(img, text, x, y, fg, bg, scale):
    th = max(1, int(scale * 2))
    (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x = max(2, min(x, img.shape[1] - w - 4))
    y = max(h + 4, min(y, img.shape[0] - b - 2))
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + b + 2), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, th, cv2.LINE_AA)


def write_check_image(path, image):
    if not write_image(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise OSError(f"Could not write review image: {path}")


def write_nir_check(path, aligned, pots, masks, filtered):
    scale = CHECK_WIDTH / aligned.shape[1]
    view = cv2.resize(aligned, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if filtered is not None and filtered.any():
        tint = cv2.resize(filtered.astype(np.float32), view.shape[1::-1], interpolation=cv2.INTER_AREA) * 0.8
        tint = tint[:, :, None]
        view = np.round(view * (1 - tint) + np.float32([35, 155, 255]) * tint).astype(np.uint8)
    pots = sorted(pots, key=lambda row: (int(row["grid_row"]), int(row["grid_col"])))
    for i, pot in enumerate(pots):
        mask = cv2.resize(masks[pot["pot_id"]].astype(np.uint8), view.shape[1::-1],
                          interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (25, 25, 25), 4, cv2.LINE_AA)
        cv2.drawContours(view, contours, -1, colour(i), 2, cv2.LINE_AA)
    for i, pot in enumerate(pots):
        x, y = int(float(pot["cx"]) * scale), int(float(pot["cy"]) * scale)
        cv2.circle(view, (x, y), 4, colour(i), -1, cv2.LINE_AA)
        label(view, pot["pot_id"], x + 7, y + 18, (255, 255, 255), (40, 40, 40), 0.6)
    write_check_image(path, view)


def main():
    a = parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per = defaultdict(list)
    for r in read_rows(a.grid):
        per[r["image"]].append(r)
    unreliable, reviewed = set(), set()
    if a.overlap and Path(a.overlap).exists():
        for r in read_rows(a.overlap):
            if r["reliable"] == "no":
                unreliable.add((r["image"], r["pot_id"]))
            if r.get("reviewed") == "yes":
                reviewed.add(r["image"])

    for image, rows in per.items():
        img = read_image(str(Path(a.images) / image))
        if img is None:
            raise OSError(f"Cannot read RGB photo: {Path(a.images) / image}")
        s = a.width / img.shape[1]
        sc = max(0.4, a.width / 3000)
        md = Path(a.masks_dir) / Path(image).stem
        pots = sorted([r for r in rows if r["status"] in ("ok", "edge")], key=lambda r: (int(r["grid_row"]), int(r["grid_col"])))
        col = {r["pot_id"]: colour(i) for i, r in enumerate(pots)}

        over = img.copy()
        for r in pots:
            path = md / f"pot_{r['pot_id']}_mask.png"
            m = read_image(path, cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != img.shape[:2]:
                raise ValueError(f"Missing or incompatible plant mask: {path}")
            over[m > 0] = col[r["pot_id"]]
        vis = cv2.addWeighted(img, 0.45, over, 0.55, 0)
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        fixed = {"partial": (150, 150, 150), "duplicate": (0, 140, 255), "card": (255, 0, 255)}
        for r in rows:
            x1, y1, x2, y2 = (int(float(r[k]) * s) for k in ("x1", "y1", "x2", "y2"))
            c = col.get(r["pot_id"], fixed.get(r["status"], (200, 200, 200)))
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            if r["status"] in ("ok", "edge", "duplicate"):
                bad = (image, r["pot_id"]) in unreliable or r["status"] == "duplicate"
                cx, cy = int(float(r["cx"]) * s), int(float(r["cy"]) * s)
                cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1)
                tag = " edge" if r["status"] == "edge" else ""
                label(vis, f"{r['pot_id']} ({float(r['confidence']):.2f}){tag}", cx - 45, cy + 25,
                      (255, 255, 255), (0, 0, 200) if bad else (40, 40, 40), sc)

        n_bad = sum((image, r["pot_id"]) in unreliable for r in pots)
        n_part = sum(r["status"] == "partial" for r in rows)
        n_edge = sum(r["status"] == "edge" for r in rows)
        state = "CORRECTED BY HAND" if image in reviewed else f"need manual check: {n_bad}"
        label(vis, f"{Path(image).stem}   pots measured: {len(pots)} ({n_edge} at edge)   slivers left out: {n_part}   {state}",
              10, int(30 * sc / 0.6), (255, 255, 255), (0, 120, 0) if image in reviewed else (0, 0, 0), sc)
        path = out / f"{Path(image).stem}_check.jpg"
        write_check_image(path, vis)
    print(f"{len(per)} check images -> {out}")


if __name__ == "__main__":
    main()
