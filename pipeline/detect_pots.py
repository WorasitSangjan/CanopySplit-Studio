# SPDX-License-Identifier: AGPL-3.0-only

"""
Step 1: find pots and the colour card with the trained YOLO model.

Writes one row per detection: image, pot_id (detection order), class,
box (x1, y1, x2, y2), centre (cx, cy), confidence.

The model (yolo26s) is end-to-end / NMS-free, so Ultralytics skips duplicate
suppression; near-identical boxes for one pot do occur, so an IoU-based NMS
is applied here per class.

Usage:
    python detect_pots.py --images photos/ --out detections.csv
"""

import argparse
import csv
from pathlib import Path

from torchvision.ops import batched_nms
from ultralytics import YOLO

if __package__:
    from .image_io import read_image
    from .inputs import DEFAULT_WEIGHTS, photo_files
else:
    from image_io import read_image
    from inputs import DEFAULT_WEIGHTS, photo_files


def parse_args():
    p = argparse.ArgumentParser(description="Detect pots and colour card")
    p.add_argument("--images", required=True, help="Folder of photos")
    p.add_argument("--pattern", default=None, help="Override the default RGB photo naming pattern")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO weights")
    p.add_argument("--out", default="detections.csv", help="Output CSV")
    p.add_argument("--conf", type=float, default=0.05,
                   help="Confidence threshold. Real pots in these trays scored as low as 0.09; at 0.05 no false "
                        "pots were found in the chamber photos, and edge slivers are removed in assign_grid.py")
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    p.add_argument("--nms-iou", type=float, default=0.5, help="IoU for removing duplicate boxes")
    return p.parse_args()


def main():
    args = parse_args()
    folder = Path(args.images)
    if not folder.is_dir():
        raise SystemExit(f"Photo folder not found: {folder}")
    paths = (sorted(p for p in folder.glob(args.pattern) if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
             if args.pattern else photo_files(folder, "RGB"))
    if not paths:
        raise SystemExit(f"No matching RGB photos in {folder}")
    model = YOLO(args.weights)
    names = model.names

    rows = []
    for path in paths:
        image = read_image(str(path))
        if image is None:
            raise SystemExit(f"Cannot read RGB photo: {path}")
        r = model.predict(source=image, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes, confs, cls = r.boxes.xyxy.cpu(), r.boxes.conf.cpu(), r.boxes.cls.cpu().long()
        keep = batched_nms(boxes, confs, cls, args.nms_iou)
        for i, k in enumerate(keep.tolist()):
            x1, y1, x2, y2 = boxes[k].tolist()
            rows.append({
                "image": path.name, "pot_id": i, "class": names[int(cls[k])],
                "x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2),
                "cx": round((x1 + x2) / 2, 2), "cy": round((y1 + y2) / 2, 2),
                "confidence": round(float(confs[k]), 4),
            })
        print(f"{path.name}: {len(keep)} detections ({len(boxes) - len(keep)} duplicates removed)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "pot_id", "class", "x1", "y1", "x2", "y2", "cx", "cy", "confidence"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} detections from {len(paths)} photos -> {args.out}")


if __name__ == "__main__":
    main()
