# SPDX-License-Identifier: AGPL-3.0-only

"""
Step 3: find each plant's leaves and assign them to the plant they grow from.

  1. Leaf pixels: excess green (2G - R - B > threshold) plus chill-damaged
     tissue (yellow, cream, pink, red), see leaf_mask_with_damage().
     (A colour threshold was chosen over SAM, which at seedling stage tended
     to segment the pot instead of the thin leaves.)
  2. Connected leaf shapes: one physical leaf stays one piece even where it
     reaches into a neighbouring pot.
  3. Each shape is checked against the centre region of every pot box:
       - one pot (or none): the whole shape belongs to that pot's plant;
         shapes far from any pot are dropped as debris;
       - two or more pots: leaves of different plants touch, and the shape
         is split by distance measured *through the leaf* (geodesic) from
         each pot's centre region; a pot whose leaves meet at a real crown
         (skeleton branch point) gets a head start.
  Output: one mask PNG per pot per photo, plus segmentation_metadata.csv.

Limitation: when the canopy closes (Control tray from about day 6) leaves
of many plants form one connected mat and cannot be separated reliably;
use the overlap score (overlap_score.py) to flag those photos.

Usage:
    python segment_leaves.py --centers-csv pots.csv --images photos/ --out-dir masks/
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

SQRT2 = 2.0**0.5
if __package__:
    from .image_io import read_image, write_image
else:
    from image_io import read_image, write_image



def parse_args():
    p = argparse.ArgumentParser(description="Per-pot color-based leaf segmentation")
    p.add_argument("--centers-csv", required=True, help="Measurable pots from assign_grid.py (--pots-out)")
    p.add_argument("--images", required=True, help="Folder of RGB images")
    p.add_argument("--out-dir", default="plant_masks_color", help="Output directory for masks")
    p.add_argument(
        "--exg-threshold", type=float, default=20.0,
        help="Excess-green (2G-R-B) threshold; higher = stricter leaf detection",
    )
    p.add_argument(
        "--min-blob-area", type=int, default=25,
        help="Discard connected leaf blobs smaller than this many pixels (noise)",
    )
    p.add_argument(
        "--max-assign-dist", type=float, default=150.0,
        help="A solitary (non-touching) leaf shape is only assigned to a pot "
             "if the shape's closest point is within this many pixels of that "
             "pot's own center -- otherwise it's dropped instead of forced "
             "onto whichever pot happens to be nearest. Without this, a small "
             "disconnected fragment (a leaf tip separated by a mask gap, a "
             "stray bit of background clutter passing the color threshold) "
             "gets colored as if it belongs to a real but distant plant, "
             "which looks like a wrong touching-leaf split even though "
             "nothing is actually touching.",
    )
    p.add_argument(
        "--seed-box-scale", type=float, default=0.35,
        help="Shrink each pot's own detection box by this factor (around its "
             "center) before testing whether a leaf shape has real material "
             "there. In a tightly packed grid, full detection boxes overlap "
             "neighboring pots by default (box width can exceed the pot-to-pot "
             "pitch), so testing against the full box would flag most leaves "
             "as 'touching' even when only one plant is actually present.",
    )
    p.add_argument(
        "--no-damaged", action="store_true",
        help="Use green pixels only (old behavior). By default, yellow / cream / "
             "pink / red leaf parts (chill damage, senescence) are also counted "
             "as leaf -- see leaf_mask_with_damage().",
    )
    p.add_argument("--damaged-min-sat", type=int, default=45, help="Min HSV saturation (0-255) for a damaged-leaf pixel")
    p.add_argument("--damaged-min-val", type=int, default=90, help="Min HSV brightness (0-255) for a damaged-leaf pixel")
    p.add_argument("--damaged-near-green", type=int, default=12,
                   help="Outside pot interiors, a damaged-leaf pixel only counts within this many px of green leaf")
    p.add_argument("--damaged-min-piece", type=int, default=400,
                   help="A piece made only of damaged-colored pixels (no green at all) must be at least this many px")
    return p.parse_args()


def load_centers(csv_path):
    """pot_id is kept as a string (a tray position like 'R2C5' from
    assign_grid.py) and only used as a label.

    x1/y1/x2/y2 (the pot's own detection box) are used to decide whether a
    leaf shape actually belongs to this pot when resolving touching-leaf
    merges below; if a CSV lacks them (older output), fall back to a small
    box around the center so that check degrades gracefully instead of
    crashing.
    """
    per_image = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cx, cy = float(row["cx"]), float(row["cy"])
            has_box = all(k in row and row[k] != "" for k in ("x1", "y1", "x2", "y2"))
            if has_box:
                x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            else:
                x1, y1, x2, y2 = cx - 20, cy - 20, cx + 20, cy + 20
            per_image[row["image"]].append(
                {"pot_id": row["pot_id"], "cx": cx, "cy": cy, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            )
    return per_image


def leaf_mask_exg(image_rgb, threshold):
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    exg = 2 * g - r - b
    mask = (exg > threshold).astype(np.uint8)
    # Close first, small kernel: a leaf tip can dip a pixel or two below
    # threshold (JPEG noise, a highlight, slight discoloration) and come
    # back, splitting one leaf into two connected components with no
    # anatomical meaning. A disconnected tip like that has no seed-box
    # evidence of its own, so it gets assigned by raw nearest-pot-center --
    # visually indistinguishable from "this tip belongs to a different
    # plant" even though nothing is actually touching. Bridging gaps this
    # small first prevents that. Then open, same as before, to drop
    # leftover single-pixel noise specks.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask.astype(bool)


def leaf_mask_with_damage(image_rgb, centers, args):
    """Green leaf (ExG, as in leaf_mask_exg) PLUS chill-damaged / senescing
    leaf parts that are no longer green.

    Damaged parts in these photos are pale cream-yellow, pink or red
    (anthocyanin, necrosis). Green-only detection misses them, so leaf area
    of stressed plants was underestimated -- the plants a chill study cares
    most about. Those colors are 'warm' hues (OpenCV hue <= 35 or >= 160),
    clearly saturated and bright.

    Where a warm pixel counts:
      - inside a pot interior (ellipse at 0.42 x box size): the background
        there is dark soil, so any bright warm pixel is plant material;
      - outside pot interiors only within --damaged-near-green px of green
        leaf, since the tray has pink/beige areas of similar color.
    Pieces made only of warm pixels (no green anywhere in them) must reach
    --damaged-min-piece px, which drops seed coats and soil debris while
    keeping a fully discolored leaf.
    """
    green = leaf_mask_exg(image_rgb, args.exg_threshold)
    h, w = green.shape
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    warm = ((hue <= 35) | (hue >= 160)) & (sat >= args.damaged_min_sat) & (val >= args.damaged_min_val)

    interior = np.zeros((h, w), np.uint8)
    for c in centers:
        ax = int((c["x2"] - c["x1"]) * 0.42)
        ay = int((c["y2"] - c["y1"]) * 0.42)
        cv2.ellipse(interior, (int(c["cx"]), int(c["cy"])), (max(ax, 1), max(ay, 1)), 0, 0, 360, 1, -1)
    k = 2 * args.damaged_near_green + 1
    near_green = cv2.dilate(green.astype(np.uint8), np.ones((k, k), np.uint8)) > 0

    mask = (green | (warm & ((interior > 0) | near_green))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    has_green = np.bincount(lab[green], minlength=n) > 0
    drop = (~has_green) & (stats[:, cv2.CC_STAT_AREA] < args.damaged_min_piece)
    drop[0] = False
    return mask.astype(bool) & ~drop[lab]


def pot_seed_box(c, scale, w, h):
    """Pot i's own box, shrunk toward its center by `scale` -- used both to
    decide whether a leaf shape has real material rooted in this pot, and
    (below) as that pot's marker *region* for splitting, rather than a bare
    center point."""
    hw = (c["x2"] - c["x1"]) / 2 * scale
    hh = (c["y2"] - c["y1"]) / 2 * scale
    bx1, by1 = int(max(c["cx"] - hw, 0)), int(max(c["cy"] - hh, 0))
    bx2, by2 = int(min(c["cx"] + hw, w)), int(min(c["cy"] + hh, h))
    return bx1, by1, bx2, by2


def thin_skeleton(mask):
    """Zhang-Suen thinning: reduces a binary blob to its 1px-wide medial
    axis. Implemented by hand (vectorized over the whole image per
    sub-iteration) since neither cv2.ximgproc nor skimage is available in
    this environment. Padding avoids np.roll wrap-around at the border."""
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2 = np.roll(img, 1, axis=0)
            p3 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
            p4 = np.roll(img, -1, axis=1)
            p5 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)
            p6 = np.roll(img, -1, axis=0)
            p7 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
            p8 = np.roll(img, 1, axis=1)
            p9 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)

            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            b = sum(seq[:8])
            a = np.zeros_like(img)
            for k in range(8):
                a += ((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8)

            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            marker = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if marker.any():
                img[marker] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


def skeleton_degree(skeleton):
    """8-connected neighbor count for every skeleton pixel (0 elsewhere)."""
    padded = np.pad(skeleton.astype(np.uint8), 1)
    deg = np.zeros_like(padded)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            deg += np.roll(np.roll(padded, dy, axis=0), dx, axis=1)
    deg = deg[1:-1, 1:-1]
    deg[~skeleton] = 0
    return deg


def seed_region_in_subcrop(seed_box, sub_mask, x0, y0):
    """seed_box (full-image coords) intersected with sub_mask, in the
    sub-crop's own local coordinates."""
    bx1, by1, bx2, by2 = seed_box
    sub_bx1, sub_bx2 = max(bx1 - x0, 0), min(bx2 - x0, sub_mask.shape[1])
    sub_by1, sub_by2 = max(by1 - y0, 0), min(by2 - y0, sub_mask.shape[0])
    region = np.zeros(sub_mask.shape, dtype=bool)
    if sub_bx2 > sub_bx1 and sub_by2 > sub_by1:
        region[sub_by1:sub_by2, sub_bx1:sub_bx2] = sub_mask[sub_by1:sub_by2, sub_bx1:sub_bx2]
    return region


def split_touching_blob(comp_mask, seed_indices, centers, seed_boxes):
    """Split one connected leaf shape shared by two or more pots.

    Each pixel goes to the pot that is closest *through the leaf itself*
    (geodesic distance: multi-source Dijkstra on the 8-connected pixel
    graph, with each pot's centre region as a zero-cost source). A pot whose
    attachment point lies at a skeleton branch point (several blades
    meeting = a real crown) gets a 60-px head start, so a single unbranched
    blade whose tip reaches a neighbour's pot stays with its own plant.

    Simpler rules were tested against the photos and rejected: straight-line
    nearest pot (cuts through leaves), watershed (pulls tips to the wrong
    plant), and a narrowing ("neck") rule (unstable because blades taper).

    Returns {pot_id: piece_mask} for the pots that received pixels."""
    ys, xs = np.where(comp_mask)
    y0, y1p, x0, x1p = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub_mask = comp_mask[y0:y1p, x0:x1p]
    h, w = sub_mask.shape

    skeleton = thin_skeleton(sub_mask)
    degree = skeleton_degree(skeleton)
    branch_ys, branch_xs = np.where(degree >= 3)
    skel_ys, skel_xs = np.where(skeleton)

    py, px = np.where(sub_mask)
    n_real = len(py)
    node_id = np.full((h, w), -1, dtype=np.int64)
    node_id[py, px] = np.arange(n_real)

    rows, cols, weights = [], [], []
    for dy, dx, wgt in (
        (-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
        (0, -1, 1.0), (0, 1, 1.0),
        (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2),
    ):
        ny, nx = py + dy, px + dx
        in_bounds = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        src = np.arange(n_real)[in_bounds]
        nbr = node_id[ny[in_bounds], nx[in_bounds]]
        has_nbr = nbr >= 0
        rows.append(src[has_nbr])
        cols.append(nbr[has_nbr])
        weights.append(np.full(has_nbr.sum(), wgt))

    n_virtual = len(seed_indices)
    hub_bonus = np.zeros(n_virtual)
    HUB_BONUS_PX = 60.0
    HUB_SEARCH_RADIUS_PX = 25.0

    for vi, i in enumerate(seed_indices):
        region = seed_region_in_subcrop(seed_boxes[i], sub_mask, x0, y0)
        ry, rx = np.where(region)
        if len(ry) == 0:
            # this pot's own box doesn't reach any leaf pixel here (rare,
            # e.g. clipped at the image edge) -- snap to the nearest real
            # pixel in the blob so it still gets a source
            c = centers[i]
            sy = min(max(int(round(c["cy"])) - y0, 0), h - 1)
            sx = min(max(int(round(c["cx"])) - x0, 0), w - 1)
            d2 = (py - sy) ** 2 + (px - sx) ** 2
            k = np.argmin(d2)
            ry, rx = py[k : k + 1], px[k : k + 1]
        vids = node_id[ry, rx]
        vnode = n_real + vi
        rows.append(np.full(len(vids), vnode))
        cols.append(vids)
        weights.append(np.zeros(len(vids)))
        rows.append(vids)
        cols.append(np.full(len(vids), vnode))
        weights.append(np.zeros(len(vids)))

        # Hub check: does a genuine skeleton branch point (3+ blades
        # meeting -- real evidence of this pot's own plant crown) sit
        # near this pot's own attachment point on the skeleton?
        if len(branch_ys):
            c = centers[i]
            ay = min(max(int(round(c["cy"])) - y0, 0), h - 1)
            ax = min(max(int(round(c["cx"])) - x0, 0), w - 1)
            if len(skel_ys):
                d2_attach = (skel_ys - ay) ** 2 + (skel_xs - ax) ** 2
                k = np.argmin(d2_attach)
                attach_y, attach_x = skel_ys[k], skel_xs[k]
            else:
                attach_y, attach_x = ay, ax
            d2_branch = (branch_ys - attach_y) ** 2 + (branch_xs - attach_x) ** 2
            if d2_branch.min() <= HUB_SEARCH_RADIUS_PX**2:
                hub_bonus[vi] = HUB_BONUS_PX

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    weights = np.concatenate(weights)
    n_total = n_real + n_virtual
    graph = csr_matrix((weights, (rows, cols)), shape=(n_total, n_total))

    virtual_ids = [n_real + vi for vi in range(n_virtual)]
    dist = dijkstra(graph, directed=False, indices=virtual_ids)  # (n_virtual, n_total)
    adjusted = dist - hub_bonus[:, None]
    owner = np.argmin(adjusted[:, :n_real], axis=0)

    pieces = {}
    for vi, i in enumerate(seed_indices):
        sel = owner == vi
        if not sel.any():
            continue
        full_piece = np.zeros(comp_mask.shape, dtype=bool)
        full_piece[y0 + py[sel], x0 + px[sel]] = True
        pieces[centers[i]["pot_id"]] = full_piece
    return pieces


def voronoi_pot_labels(shape, centers):
    """For every pixel, the *positional index* (0..len(centers)-1) of the
    nearest pot center (Voronoi partition of the image by pot center).
    Positional index rather than pot_id itself, since pot_id may now be a
    non-numeric grid label (e.g. 'R2C5') and np.bincount needs small
    non-negative ints -- callers map the winning index back to
    centers[index]["pot_id"] for the actual output label."""
    h, w = shape
    seed = np.ones((h, w), dtype=bool)
    label_grid = np.full((h, w), -1, dtype=np.int32)
    for i, c in enumerate(centers):
        y = int(round(min(max(c["cy"], 0), h - 1)))
        x = int(round(min(max(c["cx"], 0), w - 1)))
        seed[y, x] = False
        label_grid[y, x] = i

    _, indices = distance_transform_edt(seed, return_indices=True)
    nearest_y, nearest_x = indices[0], indices[1]
    return label_grid[nearest_y, nearest_x]


def main():
    args = parse_args()
    per_image = load_centers(args.centers_csv)
    image_dir = Path(args.images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []

    for image_name, centers in per_image.items():
        image_path = image_dir / image_name
        image_bgr = read_image(str(image_path))
        if image_bgr is None:
            raise SystemExit(f"Cannot read RGB photo: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        if args.no_damaged:
            leaf_mask = leaf_mask_exg(image_rgb, args.exg_threshold)
        else:
            leaf_mask = leaf_mask_with_damage(image_rgb, centers, args)
        pot_index_labels = voronoi_pot_labels((h, w), centers)  # positional indices

        n_components, labels = cv2.connectedComponents(leaf_mask.astype(np.uint8), connectivity=8)

        masks_by_pot = defaultdict(lambda: np.zeros((h, w), dtype=bool))
        blob_counts = defaultdict(int)
        n_kept_blobs = 0
        n_split_blobs = 0
        n_dropped_far_blobs = 0

        for comp_id in range(1, n_components):
            comp_mask = labels == comp_id
            area = int(comp_mask.sum())
            if area < args.min_blob_area:
                continue

            # A pot only counts as a genuine stakeholder in this shape if
            # its own box holds a *meaningful share* of it -- not just any
            # overlap at all. An absolute floor (25px) was tried first and
            # was far too permissive: one plant's own compact, entirely
            # self-contained leaf could still graze a few dozen pixels into
            # a nearby, unrelated pot's shrunk box purely from being close
            # together, wrongly qualifying that neighbor as a co-owner and
            # triggering a split of a leaf that never touched anything.
            # Requiring a real fraction of the shape's own area (verified
            # against known-good splits: the weakest genuine case held 7.6%,
            # the false-positive case held only 1.3%) filters that out while
            # still catching real touching-leaf splits, which reliably show
            # each side holding a double-digit percentage.
            min_overlap_px = max(30, int(area * 0.05))

            seed_indices = []
            seed_boxes = {}
            for i, c in enumerate(centers):
                bx1, by1, bx2, by2 = pot_seed_box(c, args.seed_box_scale, w, h)
                seed_boxes[i] = (bx1, by1, bx2, by2)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                if int(comp_mask[by1:by2, bx1:bx2].sum()) >= min_overlap_px:
                    seed_indices.append(i)

            if len(seed_indices) <= 1:
                # Solitary leaf (possibly spilling into a neighbor's
                # territory) -- keep it whole, majority vote picks its
                # true origin pot. But first: is this shape actually
                # anywhere near that pot? A small disconnected fragment
                # (a leaf tip separated by a one-pixel mask gap, a stray
                # bit of background clutter that crossed the color
                # threshold) has no seed-box evidence of its own, so
                # majority vote falls back to whichever pot is nearest --
                # which can be a real but distant plant, indistinguishable
                # by color from an actual touching-leaf split. Require the
                # shape's closest point to be within real reach of the pot
                # it's about to be assigned to; otherwise drop it rather
                # than silently mislabel it as that pot's own material.
                owner_index = int(np.bincount(pot_index_labels[comp_mask]).argmax())
                oc = centers[owner_index]
                ys_c, xs_c = np.where(comp_mask)
                min_dist = float(np.sqrt((xs_c - oc["cx"]) ** 2 + (ys_c - oc["cy"]) ** 2).min())
                if min_dist > args.max_assign_dist:
                    n_dropped_far_blobs += 1
                    continue
                owner_pot = oc["pot_id"]
                masks_by_pot[owner_pot] |= comp_mask
                blob_counts[owner_pot] += 1
            else:
                # Two+ pots' own boxes have real material in this shape --
                # different plants' leaves touching. Split it instead of
                # handing the whole thing to one winner.
                pieces = split_touching_blob(comp_mask, seed_indices, centers, seed_boxes)
                for pot_id, piece in pieces.items():
                    masks_by_pot[pot_id] |= piece
                    blob_counts[pot_id] += 1
                n_split_blobs += 1
                involved = ",".join(centers[i]["pot_id"] for i in seed_indices)
                print(f"  split: {image_name} [{involved}]")
            n_kept_blobs += 1

        image_out_dir = out_dir / Path(image_name).stem
        image_out_dir.mkdir(parents=True, exist_ok=True)

        for c in centers:
            pot_id = c["pot_id"]
            mask = masks_by_pot.get(pot_id, np.zeros((h, w), dtype=bool))
            mask_path = image_out_dir / f"pot_{pot_id}_mask.png"
            if not write_image(str(mask_path), mask.astype(np.uint8) * 255):
                raise OSError(f"Could not write automatic mask: {mask_path}")
            meta_rows.append(
                {
                    "image": image_name,
                    "pot_id": pot_id,
                    "cx": c["cx"],
                    "cy": c["cy"],
                    "leaf_pixels": int(mask.sum()),
                    "n_blobs": blob_counts.get(pot_id, 0),
                }
            )

        print(
            f"{image_name}: {len(centers)} pots, {n_kept_blobs} leaf blobs kept "
            f"(of {n_components - 1} raw), {n_split_blobs} touching-leaf blob(s) split, "
            f"{n_dropped_far_blobs} dropped (too far from any pot)"
        )

    meta_path = out_dir / "segmentation_metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "pot_id", "cx", "cy", "leaf_pixels", "n_blobs"]
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"\nDone. {len(meta_rows)} pot masks written under {out_dir}/")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
