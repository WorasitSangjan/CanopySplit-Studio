# SPDX-License-Identifier: AGPL-3.0-only

"""
Step 5: overlap score -- how much of each plant's leaf area is part of a leaf
mat that also touches other plants.

A connected leaf shape that reaches the centre region (central 35% of the
pot box) of two or more pots contains leaves of different plants. The
plant's overlap_frac is the share of its leaf pixels inside such shapes.
A plant is flagged reliable = no if overlap_frac is above 10%, or if its own
pot centre lies in such a shared shape (its leaves may have been given to a
neighbour). On the Control tray this starts at about day 6; flagged plants
should be checked with the manual correction tool.

Usage:
    python overlap_score.py --centers-csv pots.csv --masks-dir masks/ --out overlap.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
if __package__:
    from .image_io import read_image
    from .inputs import read_rows
else:
    from image_io import read_image
    from inputs import read_rows



def parse_args():
    p = argparse.ArgumentParser(description="Per-plant leaf overlap score")
    p.add_argument("--centers-csv", required=True, help="Measurable pots (assign_grid.py --pots-out)")
    p.add_argument("--masks-dir", required=True, help="Masks from segment_leaves.py")
    p.add_argument("--out", default="overlap.csv")
    p.add_argument("--seed-scale", type=float, default=0.35, help="Centre region size as a fraction of the pot box")
    p.add_argument("--min-px", type=int, default=30, help="Leaf pixels needed in a centre region to count that pot")
    p.add_argument("--limit", type=float, default=0.10, help="overlap_frac above this -> reliable = no")
    return p.parse_args()


def main():
    a = parse_args()
    per = defaultdict(list)
    for r in read_rows(a.centers_csv):
        per[r["image"]].append(r)
    rows = []
    for image, pots in per.items():
        md = Path(a.masks_dir) / Path(image).stem
        masks = {}
        shape = None
        for r in pots:
            path = md / f"pot_{r['pot_id']}_mask.png"
            m = read_image(path, cv2.IMREAD_GRAYSCALE)
            if m is None or (shape is not None and m.shape != shape):
                raise ValueError(f"Missing or incompatible plant mask: {path}")
            shape = m.shape
            masks[r["pot_id"]] = m > 0
        union = np.zeros(shape, np.uint8)
        for m in masks.values():
            union |= m.astype(np.uint8)
        n, lab = cv2.connectedComponents(union, connectivity=8)
        hits = np.zeros(n, int)
        touched = {}
        for r in pots:
            x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hw, hh = (x2 - x1) * a.seed_scale / 2, (y2 - y1) * a.seed_scale / 2
            win = lab[max(int(cy - hh), 0):int(cy + hh), max(int(cx - hw), 0):int(cx + hw)]
            ids, cnt = np.unique(win[win > 0], return_counts=True)
            touched[r["pot_id"]] = ids[cnt >= a.min_px]
            hits[touched[r["pot_id"]]] += 1
        shared = hits >= 2
        shared[0] = False
        total_leaf = int(union.sum())
        img_shared = float(shared[lab].sum()) / total_leaf if total_leaf else 0.0
        for r in pots:
            m = masks.get(r["pot_id"])
            npx = int(m.sum()) if m is not None else 0
            frac = float(shared[lab][m].sum()) / npx if npx else 0.0
            # a plant whose own centre sits in a shared mat is unreliable even if
            # it received few pixels (its leaves may have gone to a neighbour)
            in_mat = bool(shared[touched.get(r["pot_id"], np.array([], int))].any())
            rows.append({"image": image, "pot_id": r["pot_id"], "overlap_frac": round(frac, 4),
                         "in_shared_mat": "yes" if in_mat else "no",
                         "image_overlap_frac": round(img_shared, 4),
                         "reliable": "no" if (frac > a.limit or in_mat) else "yes"})
        print(f"{image}: {100 * img_shared:.0f}% of leaf area shared between plants")
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "pot_id", "overlap_frac", "in_shared_mat", "image_overlap_frac", "reliable"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} plant rows -> {a.out}")


if __name__ == "__main__":
    main()
