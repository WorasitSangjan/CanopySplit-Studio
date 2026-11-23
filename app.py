# SPDX-License-Identifier: AGPL-3.0-only

"""
CanopySplit Studio runs the whole workflow in your browser, on your computer.

  1 Run      choose a photo folder and run the workflow
  2 Check    step through the check images (pots, names, leaves per plant)
  3 Correct  fix plants whose leaves overlap (reliable = no)
  4 Results  summary charts, where the result tables are, recalculate

Nothing is uploaded: the page talks only to this program on your computer.

Start:
    python app.py                      (or double-click start_windows.bat / start_mac.command)
    python app.py --results results/control     (open existing results directly)
Then use the page at http://localhost:8765 (opens automatically).
"""

import argparse
import csv
import json
import os
import re
import shutil
import string
import subprocess
import sys
import threading
import time
import webbrowser
from collections import OrderedDict, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from filelock import Timeout

from pipeline.folder_picker import choose_folder
from pipeline.image_io import read_image, write_image
from pipeline.inputs import PHOTO_NAME, file_digest, group_labels, image_files, photo_files, photo_key, saved_path
from pipeline.outputs import publish_outputs, results_lock
from pipeline.qc_images import colour
from pipeline.readiness import installed_checks, model_check
from pipeline.segment_leaves import leaf_mask_with_damage
from pipeline.traits import (AREA_UNITS, MEASUREMENT_HELP, SUMMARY_STATISTICS, SUMMARY_TRAITS,
                             percentile_columns, summary_choices, summary_overrides)

ROOT = Path(__file__).resolve().parent

DISPLAY = 0.5  # photo is sent to the browser at half resolution
SEG_ARGS = SimpleNamespace(exg_threshold=20.0, damaged_min_sat=45, damaged_min_val=90,
                           damaged_near_green=12, damaged_min_piece=400)
UNASSIGNED = np.iinfo(np.uint16).max
APP_DATA = ROOT / ".canopysplit"
RECENT = APP_DATA / "recent.json"
LEGACY_RECENT = Path.home() / ".pot_phenotyping_recent.json"
DEFAULT_RESULTS = ROOT.parent / "results"                 # new results go next to the code folder


# ---------------------------------------------------------------- results of one run
class Store:
    def __init__(self, results, photos=None):
        self.results = Path(results).resolve()
        if not (self.results / "grid.csv").exists():
            raise FileNotFoundError(f"No workflow results in {self.results} (grid.csv missing)")
        settings = self.results / "run_settings.json"
        saved = json.loads(settings.read_text(encoding="utf-8")) if settings.is_file() else {}
        self.source_digests = dict(saved.get("rgb_inputs") or [])
        if photos is None and saved:
            photos = saved_path(saved, "photos", self.results)
        self.photos = Path(photos).expanduser().resolve() if photos else None
        self.auto, self.corr = self.results / "masks", self.results / "corrections"
        self.pots = defaultdict(list)
        self.images = {}
        with (self.results / "grid.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                image = row["image"]
                if not image or Path(image).name != image:
                    raise ValueError("The saved grid contains an invalid photo filename.")
                self.images[Path(image).stem] = image
                if row["status"] in ("ok", "edge"):
                    self.pots[image].append(row)
        self.group_names = group_labels(self.source_digests or self.images.values())
        for pots in self.pots.values():
            pots.sort(key=lambda row: (int(row["grid_row"]), int(row["grid_col"])))
        self.flag = defaultdict(set)
        overlap = self.results / "overlap.csv"
        if overlap.is_file():
            with overlap.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if (row.get("auto_reliable") or row["reliable"]) == "no":
                        self.flag[row["image"]].add(row["pot_id"])
        self.cache = OrderedDict()

    def is_done(self, image):
        return (self.corr / Path(image).stem / "done.json").exists()

    def photos_available(self):
        if self.photos is None or not self.photos.is_dir():
            return False
        expected = self.pots or self.source_digests
        return not expected or any((self.photos / name).is_file() for name in expected)

    def photo_order(self, name):
        match = PHOTO_NAME.fullmatch(name)
        if not match:
            return name, 0
        group = match["tray"]
        return self.group_names.get(group.casefold(), group), int(match["day"])

    def photo_list(self):
        photos = []
        for image in sorted(self.pots, key=self.photo_order):
            tray, day = self.photo_order(image)
            photos.append({"image": image, "day": day, "tray": tray, "flagged": len(self.flag[image]),
                           "plants": len(self.pots[image]), "done": self.is_done(image)})
        return photos

    def overview(self):
        pl = self.photo_list()
        traits = self.results / "traits.csv"
        n_rows, measured_manual = 0, set()
        t_time = traits.stat().st_mtime if traits.is_file() else 0
        if t_time:
            with traits.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    n_rows += 1
                    if row.get("mask_source") == "manual":
                        measured_manual.add(row["image"])
        current_manual = {p["image"] for p in pl if p["done"]}
        changed = current_manual ^ measured_manual
        for image in current_manual:
            if (self.corr / Path(image).stem / "done.json").stat().st_mtime > t_time:
                changed.add(image)
        return {"results": str(self.results), "photos": str(self.photos) if self.photos else None,
                "photos_available": self.photos_available(), "n_photos": len(pl),
                "to_check": sum(1 for p in pl if p["flagged"] and not p["done"]),
                "corrected": len(current_manual), "plant_rows": n_rows,
                "skipped_photos": len(self.source_digests.keys() - self.pots.keys()),
                "needs_recalc": len(changed),
                "trays": sorted({p["tray"] for p in pl}),
                "files": {k: str(self.results / f) for k, f in
                          (("traits", "traits.csv"), ("summary", "summary/summary.csv"),
                           ("check_images", "qc_images"), ("corrections", "corrections"))}}

    def qc_list(self):
        alignment = {}
        qc_summary = self.results / "images_qc.csv"
        if qc_summary.is_file():
            with qc_summary.open(newline="", encoding="utf-8-sig") as file:
                for row in csv.DictReader(file):
                    status = row.get("nir_alignment")
                    if not status and row.get("nir_matches", "") not in ("", "0"):
                        status = "legacy"
                    filtered = (row.get("nir_filtered_px") or "0").strip()
                    alignment[row.get("image")] = (status, filtered.isdecimal() and int(filtered) > 0)
        directory = self.results / "qc_images"
        rows = []
        for image in sorted(self.images.values(), key=self.photo_order):
            stem = Path(image).stem
            name, nir_name = f"{stem}_check.jpg", f"{stem}_check_noir.jpg"
            if (directory / name).is_file():
                tray, day = self.photo_order(image)
                status, filtered = alignment.get(image, (None, False))
                rows.append({"name": name, "image": image, "label": f"day{day} · {tray}",
                             "nir_name": nir_name if (directory / nir_name).is_file() else None,
                             "nir_alignment": status, "nir_filtered": filtered,
                             "flagged": len(self.flag[image]), "done": self.is_done(image)})
        return rows

    def summary_table(self):
        path = self.results / "summary" / "summary.csv"
        if not path.is_file():
            return {"traits": [], "statistics": SUMMARY_STATISTICS, "rows": [], "latest": False, "has_csv": False}
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            columns = set(reader.fieldnames or ())
            suffixes = {column.rsplit("_", 1)[-1] for column in columns}
            percentiles = sorted(float(stat[1:]) for stat in suffixes if re.fullmatch(r"q\d+(?:\.\d+)?", stat))
            statistics = (*SUMMARY_STATISTICS, *percentile_columns(percentiles))
            traits = [(key, label) for key, label in SUMMARY_TRAITS
                      if any(f"{key}_{stat}" in columns for stat, _ in statistics)]
            latest = any(f"{key}_n" in columns for key, _ in traits)
            return {"traits": traits, "statistics": statistics, "rows": list(reader),
                    "latest": latest, "has_csv": True}

    def chart_list(self):
        return sorted(p.name for p in (self.results / "summary").glob("summary_*.png"))

    def summary_chart(self, measurement, statistic):
        summary = self.summary_table()
        label = dict(summary["traits"]).get(measurement)
        stat_label = dict(summary["statistics"]).get(statistic)
        if label is None or stat_label is None or statistic == "n":
            return None
        from pipeline.summarize import write_summary_chart

        percentiles = [(key, text) for key, text in summary["statistics"]
                       if key.startswith("q") and any(row.get(f"{measurement}_{key}") for row in summary["rows"])]
        with BytesIO() as output:
            if write_summary_chart(summary["rows"], measurement, label, statistic, stat_label, percentiles, output):
                return output.getvalue()
        return None

    # ---------- correction ----------
    def check_source(self, image):
        if not self.photos_available():
            raise FileNotFoundError("The original photo folder is needed to correct this run.")
        digest = self.source_digests.get(image)
        if digest is not None and file_digest(self.photos / image) != digest:
            raise ValueError(f"{image} has changed since it was measured. Run Measure again to rebuild its masks.")

    def state(self, image):
        if image in self.cache:
            self.cache.move_to_end(image)
            return self.cache[image]
        if image not in self.pots:
            raise FileNotFoundError(f"No measurable pots for photo: {image}")
        self.check_source(image)
        bgr = read_image(str(self.photos / image))
        if bgr is None:
            raise FileNotFoundError(image)
        h, w = bgr.shape[:2]
        pots = self.pots[image]
        src = self.corr / Path(image).stem if self.is_done(image) else self.auto / Path(image).stem
        label = np.zeros((h, w), np.uint16)
        missing = []
        for i, p in enumerate(pots):
            f = src / f"pot_{p['pot_id']}_mask.png"
            m = read_image(str(f), cv2.IMREAD_GRAYSCALE) if f.exists() else None
            if m is not None:
                if m.shape != (h, w):
                    raise ValueError(f"Mask size does not match {image}: {f.name}")
                label[m > 0] = i + 1
            else:
                missing.append(i)
        for i in missing:  # pots without a saved correction: automatic leaves not given to another plant
            m = read_image(str(self.auto / Path(image).stem / f"pot_{pots[i]['pot_id']}_mask.png"), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape != (h, w):
                raise ValueError(f"Missing or incompatible automatic mask for {pots[i]['pot_id']} in {image}.")
            label[(m > 0) & (label == 0)] = i + 1
        centers = [{"pot_id": p["pot_id"], **{k: float(p[k]) for k in ("x1", "y1", "x2", "y2", "cx", "cy")}} for p in pots]
        leaf = leaf_mask_with_damage(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), centers, SEG_ARGS) | (label > 0)
        label[leaf & (label == 0)] = UNASSIGNED
        small = cv2.resize(bgr, (int(w * DISPLAY), int(h * DISPLAY)), interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        st = {"label": label, "leaf": leaf, "pots": pots, "jpg": jpg, "undo": [], "dirty": False, "w": w, "h": h}
        self.cache[image] = st
        while len(self.cache) > 3:
            self.cache.popitem(last=False)
        return st

    def describe(self, image):
        st = self.state(image)
        areas = np.bincount(st["label"].ravel(), minlength=UNASSIGNED + 1)
        pots = [{"pot_id": p["pot_id"], "idx": i, "cx": float(p["cx"]), "cy": float(p["cy"]),
                 "color": "#%02x%02x%02x" % colour(i)[::-1], "area": int(areas[i + 1]),
                 "flagged": p["pot_id"] in self.flag[image], "edge": p["status"] == "edge"}
                for i, p in enumerate(st["pots"])]
        return {"image": image, "w": st["w"], "h": st["h"], "display": DISPLAY, "pots": pots,
                "unassigned": int(areas[UNASSIGNED]), "dirty": st["dirty"], "done": self.is_done(image),
                "can_undo": len(st["undo"]) > 0}

    def overlay(self, image):
        st = self.state(image)
        lut = np.zeros((UNASSIGNED + 1, 4), np.uint8)
        for i in range(len(st["pots"])):
            b, g, r = colour(i)
            lut[i + 1] = (b, g, r, 175)
        lut[UNASSIGNED] = (255, 255, 255, 235)
        small = cv2.resize(st["label"], (int(st["w"] * DISPLAY), int(st["h"] * DISPLAY)), interpolation=cv2.INTER_NEAREST)
        return cv2.imencode(".png", lut[small])[1].tobytes()

    @staticmethod
    def nearest_leaf(st, x, y, r=25):
        h, w = st["leaf"].shape
        x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        ys, xs = np.nonzero(st["label"][y0:y1, x0:x1])
        if len(ys) == 0:
            return None
        k = np.argmin((ys + y0 - y) ** 2 + (xs + x0 - x) ** 2)
        return int(xs[k] + x0), int(ys[k] + y0)

    def edit(self, image, op, idx, pts, radius):
        st = self.state(image)
        lab = st["label"]
        if op not in ("pick", "brush", "remove", "piece") or not pts:
            raise ValueError("Choose a valid correction tool and position.")
        if op in ("brush", "piece") and not 0 <= idx < len(st["pots"]):
            raise ValueError("Choose a plant before assigning leaf pixels.")
        pts = [(int(round(x)), int(round(y))) for x, y in pts]
        if op == "pick":
            p = self.nearest_leaf(st, *pts[0])
            v = int(lab[p[1], p[0]]) if p else 0
            return {"picked": v - 1 if 0 < v < UNASSIGNED else None}
        target = idx + 1 if op != "remove" else UNASSIGNED
        if op in ("brush", "remove"):
            m = np.zeros(lab.shape, np.uint8)
            r = max(1, int(radius))
            for p in pts:
                cv2.circle(m, p, r, 1, -1)
            if len(pts) > 1:
                cv2.polylines(m, [np.int32(pts).reshape(-1, 1, 2)], False, 1, thickness=2 * r)
            sel = m.astype(bool) & st["leaf"]
        else:
            p = self.nearest_leaf(st, *pts[0])
            if p is None:
                return {"changed": 0}
            v = lab[p[1], p[0]]
            _, cc = cv2.connectedComponents((lab == v).astype(np.uint8), connectivity=8)
            sel = cc == cc[p[1], p[0]]
        changed = np.flatnonzero(sel & (lab != target)).astype(np.int32)
        if not changed.size:
            return {"changed": 0}
        flat = lab.ravel()
        st["undo"].append((changed, flat[changed]))
        del st["undo"][:-15]
        flat[changed] = target
        st["dirty"] = True
        return {"changed": 1}

    def undo(self, image):
        st = self.state(image)
        if st["undo"]:
            changed, previous = st["undo"].pop()
            st["label"].ravel()[changed] = previous
            st["dirty"] = True

    def save(self, image, user):
        st = self.state(image)
        self.check_source(image)
        name = Path(image).stem
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with results_lock(self.results):
            self.corr.mkdir(exist_ok=True)
            with TemporaryDirectory(prefix=".edit-", dir=self.corr) as temporary:
                staged = Path(temporary)
                folder = staged / name
                folder.mkdir()
                for i, pot in enumerate(st["pots"]):
                    path = folder / f"pot_{pot['pot_id']}_mask.png"
                    mask = (st["label"] == i + 1).astype(np.uint8) * 255
                    if not write_image(str(path), mask):
                        raise OSError(f"Could not write correction mask: {path}")
                (folder / "done.json").write_text(json.dumps({"photo": image, "saved": stamp, "by": user}, indent=1),
                                                  encoding="utf-8")
                log = self.corr / "log.csv"
                staged_log = staged / log.name
                if log.is_file():
                    shutil.copyfile(log, staged_log)
                with staged_log.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(["time", "photo", "by"])
                    writer.writerow([stamp, image, user])
                publish_outputs(staged, self.corr, (name, log.name))
            st["dirty"] = False

    def discard(self, image):
        if image not in self.pots:
            raise FileNotFoundError(f"No measurable pots for photo: {image}")
        self.cache.pop(image, None)


# ---------------------------------------------------------------- app state
class App:
    def __init__(self):
        self.store = None
        self.lock = threading.Lock()
        self.setup_lock = threading.Lock()
        self.picker_lock = threading.Lock()
        self.runtime_status = None
        self.job = {"running": False, "log": "", "ok": None, "kind": None, "out_dir": None}

    def recent(self):
        try:
            try:
                saved = RECENT.read_text(encoding="utf-8")
            except FileNotFoundError:
                saved = LEGACY_RECENT.read_text(encoding="utf-8")
            items = json.loads(saved)
        except (OSError, ValueError):
            return []
        return [p for p in items if isinstance(p, str) and (Path(p) / "grid.csv").is_file()] if isinstance(items, list) else []

    def remember(self, path):
        new = str(Path(path).resolve())
        items = [new]
        for p in self.recent():   # one entry per folder (Windows paths ignore case)
            if all(os.path.normcase(p) != os.path.normcase(q) for q in items):
                items.append(p)
        try:
            RECENT.parent.mkdir(parents=True, exist_ok=True)
            RECENT.write_text(json.dumps(items[:12], indent=1), encoding="utf-8")
        except OSError:
            pass

    def open(self, results, photos=None):
        store = Store(results, photos)
        overview = store.overview()
        self.store = store
        self.remember(store.results)
        return overview

    def run_job(self, kind, cmd, out_dir):
        if self.job["running"]:
            raise RuntimeError("Another run is still going; wait until it has finished.")
        self.job.update(running=True, log="", ok=None, kind=kind, out_dir=str(out_dir))

        def go():
            ok = False
            try:
                env = dict(os.environ, PYTHONUNBUFFERED="1", YOLO_CONFIG_DIR=str(APP_DATA / "yolo"),
                           MPLCONFIGDIR=str(APP_DATA / "matplotlib"))
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, errors="replace", env=env, cwd=ROOT) as process:
                    for line in process.stdout:
                        self.job["log"] = (self.job["log"] + line)[-12000:]
                    finished = process.wait() == 0
                if finished:
                    with self.lock:
                        self.open(out_dir)
                    ok = True
            except Exception as error:
                self.job["log"] += f"\nCould not finish opening the results: {error}"
            finally:
                self.job.update(running=False, ok=ok)

        threading.Thread(target=go, daemon=True).start()

    def pick_folder(self, start, purpose="dataset"):
        if purpose not in ("dataset", "output", "saved"):
            raise ValueError("Choose a dataset or results folder.")
        if not self.picker_lock.acquire(blocking=False):
            return {"busy": True}
        try:
            path = Path(start).expanduser() if start else Path.home()
            initial = path if path.is_dir() else Path.home()
            if purpose != "dataset" and initial != path:
                initial = next((parent for parent in path.parents if parent.is_absolute() and parent.is_dir()), initial)
            selected = choose_folder(initial, purpose=purpose)
            return {"available": selected is not None, "path": selected}
        except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError):
            return {"available": False}
        finally:
            self.picker_lock.release()

    def setup_status(self, refresh=False):
        with self.setup_lock:
            checks = installed_checks() + [model_check()]
            if all(check["ok"] for check in checks):
                if refresh or self.runtime_status is None or not self.runtime_status["ok"]:
                    self.runtime_status = self.probe_runtime()
                checks.append(self.runtime_status)
            return {"ready": all(check["ok"] for check in checks), "checks": checks, "platform": sys.platform}

    @staticmethod
    def probe_runtime():
        try:
            with TemporaryDirectory(prefix="canopysplit-setup-check-") as temporary:
                env = dict(os.environ, YOLO_CONFIG_DIR=temporary, MPLCONFIGDIR=temporary)
                process = subprocess.run([sys.executable, "-B", "-m", "pipeline.readiness", "runtime"],
                                         cwd=ROOT, env=env, capture_output=True, text=True,
                                         errors="replace", check=True, timeout=45)
            return json.loads(process.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            print(f"Local software check failed. {error}", file=sys.stderr)
            return {"name": "Image processing and detection", "ok": False,
                    "detail": "The software check did not finish. Close and reopen CanopySplit Studio, then try again."}

    def run(self, b):
        photos = Path(b["photos"]).expanduser().resolve()
        if not photos.is_dir():
            raise FileNotFoundError(f"Photo folder not found: {photos}")
        if not photo_files(photos, "RGB"):
            raise FileNotFoundError("No RGB photos found. Use names like day1_RGB_group1.jpg.")
        area_unit = b.get("area_unit", "cm")
        if area_unit not in AREA_UNITS:
            raise ValueError("Choose centimetres or inches.")
        stats, percentiles = summary_choices(b.get("summary_stats"), b.get("percentiles"))
        if "percentiles" in b and len(percentiles) > 3:
            raise ValueError("Enter up to three percentiles.")
        overrides = summary_overrides(b.get("summary_overrides"))
        setup = self.setup_status()
        if not setup["ready"]:
            problems = " ".join(check["detail"] for check in setup["checks"] if not check["ok"])
            raise ValueError(f"Setup needs attention before measuring. {problems}")
        out = Path(b.get("out_dir") or (DEFAULT_RESULTS / photos.name)).expanduser().resolve()
        cmd = [sys.executable, "-B", "-u", str(ROOT / "run_pipeline.py"), "--photos", str(photos), "--out-dir", str(out)]
        cmd += ["--area-unit", area_unit, "--summary-stats", *stats, "--percentiles", *map(str, percentiles)]
        if overrides:
            cmd += ["--summary-overrides", json.dumps(overrides)]
        if b.get("pot_diameter_cm"):
            cmd += ["--pot-diameter-cm", str(float(b["pot_diameter_cm"]))]
        if not b.get("use_nir", True):
            cmd += ["--no-nir"]
        for k in ("stages", "genotypes"):
            if b.get(k):
                cmd += [f"--{k}", str(Path(b[k]).expanduser().resolve())]
        self.run_job("run", cmd, out)
        return self.job

    def recalculate(self):
        if self.store is None:
            raise RuntimeError("Open a results folder first.")
        if not self.store.photos_available():
            raise FileNotFoundError("The original photo folder is needed to recalculate this run.")
        cmd = [sys.executable, "-B", "-u", str(ROOT / "run_pipeline.py"), "--out-dir", str(self.store.results),
               "--photos", str(self.store.photos), "--recalculate"]
        self.run_job("recalculate", cmd, self.store.results)
        return self.job


def browse(path):
    """Folders and CSV files under path, for the folder picker."""
    if not path:
        roots = []
        if os.name == "nt":
            roots = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
        else:
            roots = ["/"] + (["/Volumes"] if Path("/Volumes").exists() else [])
        return {"path": "", "parent": None, "dirs": [str(Path.home())] + roots, "files": [], "n_rgb": 0, "n_nir": 0, "n_images": 0, "top": True}
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Not a folder: {p}")
    dirs, files = [], []
    n_rgb = n_nir = n_images = 0
    photo_problems = []
    try:
        n_images = len(image_files(p))
        for channel in ("RGB", "NIR"):
            photos = photo_files(p, channel)
            keys = [photo_key(PHOTO_NAME.fullmatch(photo.name)) for photo in photos]
            if len(keys) != len(set(keys)):
                photo_problems.append(f"Keep only one {channel} photo for each group and day.")
            if channel == "RGB":
                n_rgb = len(photos)
            else:
                n_nir = len(photos)
        for c in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if c.name.startswith("."):
                continue
            if c.is_dir():
                dirs.append(c.name)
            elif c.suffix.lower() == ".csv":
                files.append(c.name)
    except PermissionError:
        pass
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else "", "dirs": dirs, "files": files,
            "n_rgb": n_rgb, "n_nir": n_nir, "n_images": n_images, "photo_problems": photo_problems,
            "is_results": (p / "grid.csv").exists(), "top": False}


# ---------------------------------------------------------------- web server
def make_handler(app):
    page = (ROOT / "app.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype="application/json", code=200):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def need_store(self):
            if app.store is None:
                raise RuntimeError("Open a results folder first (tab 1).")
            return app.store

        def is_local_host(self):
            host = self.headers.get("Host")
            port = self.server.server_port
            names = ("localhost", "127.0.0.1")
            return host in tuple(f"{name}:{port}" for name in names) or (port == 80 and host in names)

        def do_GET(self):
            if not self.is_local_host():
                return self.send({"error": "Open CanopySplit Studio at its local browser address."}, code=403)
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self.send(page, "text/html; charset=utf-8")
                if u.path == "/assets/source-sans-3.woff2":
                    return self.send((ROOT / "assets" / "source-sans-3.woff2").read_bytes(), "font/woff2")
                if u.path == "/api/job":
                    return self.send(app.job)
                if u.path == "/api/setup":
                    return self.send(app.setup_status(refresh=q.get("refresh") == "1"))
                with app.lock:
                    if u.path == "/api/state":
                        return self.send({"open": app.store.overview() if app.store else None, "recent": app.recent(),
                                          "default_out": str(DEFAULT_RESULTS), "summary_traits": SUMMARY_TRAITS,
                                          "measurement_help": MEASUREMENT_HELP})
                    if u.path == "/api/browse":
                        return self.send(browse(q.get("path", "")))
                    s = self.need_store()
                    if u.path == "/api/photos":
                        return self.send(s.photo_list())
                    if u.path == "/api/load":
                        return self.send(s.describe(q["image"]))
                    if u.path == "/api/photo.jpg":
                        return self.send(s.state(q["image"])["jpg"], "image/jpeg")
                    if u.path == "/api/overlay.png":
                        return self.send(s.overlay(q["image"]), "image/png")
                    if u.path == "/api/qc_list":
                        return self.send(s.qc_list())
                    if u.path == "/api/qc.jpg":
                        return self.send((s.results / "qc_images" / Path(q["name"]).name).read_bytes(), "image/jpeg")
                    if u.path == "/api/summary":
                        return self.send(s.summary_table())
                    if u.path in ("/api/summary.csv", "/api/traits.csv"):
                        name = "summary/summary.csv" if u.path == "/api/summary.csv" else "traits.csv"
                        return self.send((s.results / name).read_bytes(), "text/csv; charset=utf-8")
                    if u.path == "/api/charts":
                        return self.send(s.chart_list())
                    if u.path == "/api/chart.png":
                        if "measurement" in q:
                            chart = s.summary_chart(q["measurement"], q.get("statistic"))
                            if chart is None:
                                return self.send({"error": "No chart data for this selection."}, code=404)
                            return self.send(chart, "image/png")
                        return self.send((s.results / "summary" / Path(q["name"]).name).read_bytes(), "image/png")
                self.send({"error": "not found"}, code=404)
            except Exception as e:
                self.send({"error": str(e)}, code=500)

        def do_POST(self):
            if not self.is_local_host():
                return self.send({"error": "Open CanopySplit Studio at its local browser address."}, code=403)
            u = urlparse(self.path)
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                return self.send({"error": "Open CanopySplit Studio directly in this browser."}, code=403)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if not isinstance(body, dict):
                    return self.send({"error": "Expected a set of CanopySplit settings."}, code=400)
                if u.path == "/api/pick-folder":
                    host = self.headers.get("Host")
                    if origin != f"http://{host}":
                        return self.send({"error": "Open CanopySplit Studio directly in this browser."}, code=403)
                    return self.send(app.pick_folder(body.get("start", ""), body.get("purpose", "dataset")))
                with app.lock:
                    if app.job["running"]:
                        return self.send({"error": "Wait for the current run to finish."}, code=409)
                    if u.path == "/api/open":
                        return self.send(app.open(body["results"]))
                    if u.path == "/api/run":
                        return self.send(app.run(body))
                    if u.path == "/api/recalculate":
                        return self.send(app.recalculate())
                    s = self.need_store()
                    im = body.get("image")
                    if u.path == "/api/edit":
                        res = s.edit(im, body["op"], body.get("idx", 0), body["points"], body.get("radius", 10))
                        return self.send({**res, **s.describe(im)})
                    if u.path == "/api/undo":
                        s.undo(im)
                        return self.send(s.describe(im))
                    if u.path == "/api/save":
                        s.save(im, body.get("user", ""))
                        return self.send(s.describe(im))
                    if u.path == "/api/discard":
                        s.discard(im)
                        return self.send({"discarded": True})
                self.send({"error": "not found"}, code=404)
            except Timeout:
                self.send({"error": "Another process is writing to this results folder."}, code=409)
            except (KeyError, TypeError, ValueError) as error:
                self.send({"error": str(error)}, code=400)
            except Exception as error:
                self.send({"error": str(error)}, code=500)

    return H


def main():
    ap = argparse.ArgumentParser(description="CanopySplit Studio (runs in your browser, on your computer)")
    ap.add_argument("--results", default=None, help="Open this results folder at start")
    ap.add_argument("--photos", default=None, help="Photo folder of those results (default: from run_settings.json)")
    ap.add_argument("--port", type=int, default=8765, help="Local port (0 picks an available port)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    app = App()
    if a.results:
        try:
            app.open(a.results, a.photos)
        except Exception as e:
            raise SystemExit(str(e))
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(app))
    except OSError:
        raise SystemExit(f"Port {a.port} is already in use: the app may already be running "
                         f"(open http://localhost:{a.port}), or start it with --port 8766.")
    port = srv.server_port
    url = f"http://localhost:{port}"
    print(f"CanopySplit Studio is running at {url} (or http://127.0.0.1:{port})")
    print("Keep this window open while you work; close it (or press Ctrl+C) to stop.")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
