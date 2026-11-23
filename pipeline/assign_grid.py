# SPDX-License-Identifier: AGPL-3.0-only

"""
Step 2: give every detected pot a fixed tray position (row, column).

Detection order from YOLO changes from photo to photo; a tray position such as
"R2C5" (row 2, column 5) does not, so the same plant can be followed over days.

Method (per image, no camera calibration needed):
  1. Pots cut off at the photo edge: boxes much smaller than a normal pot.
     If at least half of a pot is visible it is kept with status "edge": it
     gets a position (placed on the grid fitted to the full pots) and is
     measured, but its leaf area is incomplete. Smaller slivers get status
     "partial" and are not measured.
  2. The tray's tilt is measured from neighbouring pots in the same row
     (median direction of side-by-side pot pairs), so a slightly rotated
     tray is handled even when columns have different numbers of pots.
  3. Rows: optimal 1-D clustering (Fisher / Ckmeans) of the row coordinate
     into --max-rows groups.
  4. Columns: pot spacing (pitch) from neighbouring pots in the same row; the
     column grid offset is fitted to all pots at once (circular mean of
     position modulo pitch), so one odd box cannot shift every column number.
  5. The colour card is labelled "card", not given a pot position.
  6. If two pots still get the same position, the more confident one keeps
     it and the other is labelled with status "duplicate".

Outputs:
  --out       all detections with pot_id, grid_row, grid_col, status
  --pots-out  only measurable pots (status "ok" or "edge"), the input for step 3

Usage:
    python assign_grid.py --centers-csv detections.csv --out grid.csv --pots-out pots.csv
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Assign tray row/column positions to detected pots")
    p.add_argument("--centers-csv", required=True, help="Output of detect_pots.py (all classes)")
    p.add_argument("--out", default="grid.csv", help="All detections with positions and status")
    p.add_argument("--pots-out", default=None, help="Measurable pots only (input for segment_leaves.py)")
    p.add_argument("--max-rows", type=int, default=4, help="Number of pot rows in the tray")
    p.add_argument("--partial-ratio", type=float, default=0.85,
                   help="A pot box whose smaller side is below this fraction of the typical pot size is cut off")
    p.add_argument("--edge-min-visible", type=float, default=0.5,
                   help="A cut-off pot with at least this share of a normal pot visible is kept (status 'edge', "
                        "measured, leaf area incomplete); smaller slivers are left out (status 'partial')")
    return p.parse_args()


def optimal_1d_clusters(values_sorted, k):
    """Split sorted 1-D values into k contiguous groups with minimum total
    within-group variance (exact dynamic programme). Returns group index per value."""
    n = len(values_sorted)
    k = min(k, n)
    pre = np.concatenate([[0.0], np.cumsum(values_sorted)])
    pre2 = np.concatenate([[0.0], np.cumsum(values_sorted ** 2)])

    def cost(i, j):
        s, sq, c = pre[j] - pre[i], pre2[j] - pre2[i], j - i
        return sq - s * s / c

    D = np.full((k + 1, n + 1), np.inf)
    D[0, 0] = 0.0
    back = np.zeros((k + 1, n + 1), dtype=int)
    for c in range(1, k + 1):
        for j in range(c, n + 1):
            best, bi = np.inf, c - 1
            for i in range(c - 1, j):
                v = D[c - 1, i] + cost(i, j)
                if v < best:
                    best, bi = v, i
            D[c, j], back[c, j] = best, bi
    bounds, j = [], n
    for c in range(k, 0, -1):
        i = back[c, j]
        bounds.append(i)
        j = i
    bounds = sorted(bounds)[1:]
    return np.searchsorted(bounds, np.arange(n), side="right")


def assign(points, max_rows, extra=None):
    """points: (n, 2) centres of full pots; extra: (m, 2) estimated centres of
    edge-cut pots. Returns (row, col, extra_row, extra_col), all 1-based."""
    n = len(points)
    if n < 2:
        m = 0 if extra is None else len(extra)
        return np.ones(n, int), np.ones(n, int), np.ones(m, int), np.ones(m, int)
    centred = points - points.mean(axis=0)
    # tray tilt from neighbouring pots in the same row: vectors between pots at
    # about one pot spacing that point more sideways than up/down
    d = points[:, None, :] - points[None, :, :]
    dist = np.hypot(d[..., 0], d[..., 1])
    np.fill_diagonal(dist, np.inf)
    nn = np.median(dist.min(axis=1))
    i, j = np.where((dist > 0.7 * nn) & (dist < 1.3 * nn) & (np.abs(d[..., 0]) > np.abs(d[..., 1])) & (d[..., 0] > 0))
    theta = float(np.median(np.arctan2(d[i, j, 1], d[i, j, 0]))) if len(i) else 0.0
    col_dir = np.array([np.cos(theta), np.sin(theta)])     # along a row (left -> right)
    row_dir = np.array([-np.sin(theta), np.cos(theta)])    # down the tray (top -> bottom)
    u = centred @ col_dir
    v = centred @ row_dir

    ov = np.argsort(v)
    row = np.empty(n, int)
    row[ov] = optimal_1d_clusters(v[ov], max_rows)

    diffs = []
    for r in np.unique(row):
        ru = np.sort(u[row == r])
        diffs += np.diff(ru).tolist()
    diffs = np.array(diffs)
    positive = diffs[diffs > 0]
    pitch = float(np.median(positive[positive > 0.5 * np.median(positive)])) if positive.size else 1.0
    # grid offset fitted to all pots: circular mean of position modulo pitch
    ang = 2 * np.pi * u / pitch
    phase = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) / (2 * np.pi) * pitch
    col = np.round((u - phase) / pitch).astype(int)

    # pots cut off at the photo edge: placed on the grid fitted to the full pots
    erow = ecol = np.zeros(0, int)
    if extra is not None and len(extra):
        ec = np.asarray(extra, float) - points.mean(axis=0)
        eu, ev = ec @ col_dir, ec @ row_dir
        row_centres = np.array([v[row == r].mean() for r in range(row.max() + 1)])
        erow = np.abs(ev[:, None] - row_centres[None, :]).argmin(axis=1)
        ecol = np.round((eu - phase) / pitch).astype(int)
    shift = min(col.min(), ecol.min() if len(ecol) else col.min())
    return row + 1, col - shift + 1, erow + 1, ecol - shift + 1


def main():
    args = parse_args()
    per_image = defaultdict(list)
    with open(args.centers_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        in_fields = reader.fieldnames
        if not in_fields or not {"image", "pot_id", "class", "x1", "y1", "x2", "y2", "cx", "cy", "confidence"}.issubset(in_fields):
            raise SystemExit("The detection table is missing required columns. Run pot detection first.")
        for r in reader:
            per_image[r["image"]].append(r)

    out_rows, n_partial, n_dup, n_edge = [], 0, 0, 0
    for image, rows in per_image.items():
        pots = [r for r in rows if r["class"] == "pot"]
        wh = lambda r: (float(r["x2"]) - float(r["x1"]), float(r["y2"]) - float(r["y1"]))
        typical = np.median([min(wh(r)) for r in pots]) if pots else 0
        for r in rows:
            r["status"] = "ok" if r["class"] == "pot" else "card"
            if r["class"] == "pot" and min(wh(r)) < args.partial_ratio * typical:
                w, h = wh(r)
                r["status"] = "edge" if w * h / typical ** 2 >= args.edge_min_visible else "partial"
        good = [r for r in rows if r["status"] == "ok"]
        edge = [r for r in rows if r["status"] == "edge"]
        if good:
            pts = np.array([[float(r["cx"]), float(r["cy"])] for r in good])
            # estimated true centre of an edge-cut pot: one pot radius in from the visible side
            epts = []
            for r in edge:
                (x1, y1, x2, y2), (w, h) = (float(r[k]) for k in ("x1", "y1", "x2", "y2")), wh(r)
                cx = (x2 - typical / 2 if x1 < 5 else x1 + typical / 2) if w < args.partial_ratio * typical else (x1 + x2) / 2
                cy = (y2 - typical / 2 if y1 < 5 else y1 + typical / 2) if h < args.partial_ratio * typical else (y1 + y2) / 2
                epts.append([cx, cy])
            rr, cc, err, ecc = assign(pts, args.max_rows, np.array(epts) if epts else None)
            taken = {}
            ranked = sorted(zip(good, rr, cc), key=lambda t: -float(t[0]["confidence"])) + list(zip(edge, err, ecc))
            for r, a, b in ranked:
                lab = f"R{a}C{b}"
                if lab in taken:
                    r["status"] = "duplicate"
                    n_dup += 1
                    lab = f"{lab}_dup"
                elif r["status"] == "edge":
                    n_edge += 1
                taken[lab] = r
                r["pot_id"], r["grid_row"], r["grid_col"] = lab, a, b
        else:
            for r in edge:
                r["status"] = "partial"
        for r in rows:
            if r["status"] == "card":
                r["pot_id"], r["grid_row"], r["grid_col"] = "card", "", ""
            elif r["status"] == "partial":
                n_partial += 1
                r["pot_id"], r["grid_row"], r["grid_col"] = f"partial_{r['pot_id']}", "", ""
            out_rows.append(r)

    fields = ["image", "pot_id", "grid_row", "grid_col", "status"] + [
        k for k in in_fields if k not in ("image", "pot_id", "grid_row", "grid_col", "status")]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    if args.pots_out:
        with open(args.pots_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows([r for r in out_rows if r["status"] in ("ok", "edge")])
    print(f"{len(out_rows)} detections in {len(per_image)} images: {n_edge} edge-cut pots kept, "
          f"{n_partial} slivers excluded, {n_dup} duplicate positions -> {args.out}")


if __name__ == "__main__":
    main()
