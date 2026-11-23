# SPDX-License-Identifier: AGPL-3.0-only

"""
Run the whole workflow on one folder of photos:

  1. detect_pots.py        find pots and the colour card (YOLO)
  2. assign_grid.py        row and column for every pot (R1C1, R1C2, ...)
  3. segment_leaves.py     leaf pixels per plant (green + chill-damaged)
  3b apply_corrections.py  use manual corrections where they exist
  4. extract_traits.py     standardised per-plant traits
  5. overlap_score.py      flag plants whose leaves overlap neighbours
  6. qc_images.py          one check image per photo
  7. summarize.py          summary table and charts per group and day

Photos must be named  day<N>_RGB_<group>.jpg  (colour) and
day<N>_NIR_<group>.jpg  (NIR, optional). A folder can contain several groups,
each standardised as its own time series.

The same workflow can be run from the app (python app.py), which also has
the check images and the manual correction tool.

Manual correction (plants flagged reliable = no):
  python app.py --results <out-dir>   (tab 3 Correct)
  then "Recalculate results" (tab 4), or:
  python run_pipeline.py --out-dir <out-dir> --recalculate

Main output: <out-dir>/traits.csv (one row per plant per photo),
<out-dir>/qc_images/, <out-dir>/summary/.

Usage:
    python run_pipeline.py --photos path/to/photos --out-dir ../results/chill_low --pot-diameter-cm 22.23
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import Timeout

from pipeline.inputs import DEFAULT_WEIGHTS, PHOTO_NAME, file_digest, photo_files, photo_key, read_rows, saved_path
from pipeline.outputs import publish_outputs, results_lock
from pipeline.traits import AREA_UNITS, DEFAULT_STATISTICS, summary_choices, summary_overrides

HERE = Path(__file__).resolve().parent
PIPE = HERE / "pipeline"
PATH_SETTINGS = ("photos", "nir_dir", "stages", "genotypes", "weights")
AUTOMATIC_OUTPUTS = ("detections.csv", "grid.csv", "pots.csv", "masks")
MEASUREMENT_OUTPUTS = ("masks_final", "traits.csv", "images_qc.csv", "overlap.csv", "qc_images", "summary")


def parse_args():
    p = argparse.ArgumentParser(description="CanopySplit Studio workflow for measuring plants from photos")
    p.add_argument("--photos", default=None, help="Folder containing the input photos")
    p.add_argument("--nir-dir", default=None, help="Folder with NIR photos (default: same as --photos)")
    p.add_argument("--no-nir", action="store_true", help="Skip NIR traits")
    p.add_argument("--out-dir", required=True, help="Output folder")
    p.add_argument("--pot-diameter-cm", type=float, default=None, help="Pot top diameter in centimetres")
    p.add_argument("--area-unit", choices=AREA_UNITS, help="Length unit for physical area (squared, defaults to cm)")
    p.add_argument("--summary-stats", nargs="*", choices=DEFAULT_STATISTICS, help="Statistics across plants")
    p.add_argument("--percentiles", nargs="*", type=float, help="Percentiles across plants, from 0 to 100")
    p.add_argument("--summary-overrides", type=json.loads, help="JSON statistic and percentile choices by measurement")
    p.add_argument("--stages", default=None, help="Optional CSV: treatment, day, stage")
    p.add_argument("--genotypes", default=None, help="Optional CSV: treatment, pot_id, genotype")
    p.add_argument("--weights", default=None, help="YOLO weights (default: models/pot_detector_yolo26s.pt)")
    p.add_argument("--skip-existing", action="store_true", help="Reuse automatic masks from a completed run with the same RGB inputs and weights")
    p.add_argument("--recalculate", action="store_true",
                   help="Keep pots and automatic leaves; redo measurements, check images and summary "
                        "with manual corrections applied (settings from the first run)")
    return p.parse_args()


def step(name, script, args):
    print(f"\n== {name}", flush=True)
    result = subprocess.run([sys.executable, "-B", str(PIPE / script), *map(str, args)])
    if result.returncode:
        raise SystemExit(f"{name} failed (exit status {result.returncode}).")


def input_signature(paths):
    return [[path.name, file_digest(path)] for path in paths]


def automatic_available(out, pots):
    return (
        bool(pots)
        and all((out / name).is_file() for name in AUTOMATIC_OUTPUTS[:3])
        and (out / "masks" / "segmentation_metadata.csv").is_file()
        and all((out / "masks" / Path(row["image"]).stem / f"pot_{row['pot_id']}_mask.png").is_file()
                for row in pots)
    )


def previous_pots(out):
    path = out / "pots.csv"
    if not path.is_file() or not (out / "grid.csv").is_file():
        return []
    rows = read_rows(path)
    columns = {"image", "pot_id", "grid_row", "grid_col", "status", "cx", "cy", "x1", "y1", "x2", "y2"}
    return rows if rows and columns.issubset(rows[0]) else []


def protect_corrected_photos(out, saved, settings, old_pots, recalculate):
    stems = {marker.parent.name for marker in (out / "corrections").glob("*/done.json")}
    corrected = {row["image"] for row in old_pots if Path(row["image"]).stem in stems}
    if not stems:
        return corrected
    if stems != {Path(name).stem for name in corrected}:
        raise SystemExit("Some saved corrections have no matching pot data. Choose a different location to save results.")
    original = dict(saved.get("rgb_inputs") or [])
    current = dict(settings["rgb_inputs"])
    if original:
        compatible = all(name in original and current.get(name) == original[name] for name in corrected)
    else:
        compatible = corrected.issubset(current) and (
            recalculate or saved_path(saved, "photos", out) == Path(settings["photos"])
        )
    if not compatible:
        raise SystemExit("The RGB photos for saved corrections have changed. Choose a different location to save results.")
    return corrected


def check_corrected_positions(previous, current, corrected):
    new_pots = {(row["image"], row["pot_id"]): row for row in current if row["image"] in corrected}
    for old in previous:
        if old["image"] not in corrected:
            continue
        new = new_pots.get((old["image"], old["pot_id"]))
        if new is not None:
            offset = math.hypot(float(old["cx"]) - float(new["cx"]), float(old["cy"]) - float(new["cy"]))
            diameter = min(float(old["x2"]) - float(old["x1"]), float(old["y2"]) - float(old["y1"]))
            if offset <= 0.25 * diameter:
                continue
        raise SystemExit("The detected pot positions changed for a corrected photo. Choose a different location to save results.")


def settings_for(a, out, saved):
    parents = {"photos", "nir_dir"}
    a.no_nir = a.no_nir or saved.get("no_nir", False)
    settings = {}
    for key in PATH_SETTINGS:
        value = getattr(a, key)
        path = Path(value).expanduser().resolve() if value else saved_path(saved, key, out)
        setattr(a, key, path)
        settings[key] = str(path) if path is not None else None
        settings[f"{key}_rel"] = None
        if path is not None:
            required = not (key == "weights" and a.recalculate or key == "nir_dir" and a.no_nir)
            if required and not (path.is_dir() if key in parents else path.is_file()):
                raise SystemExit(f"Invalid --{key.replace('_', '-')}: {path}")
            try:
                settings[f"{key}_rel"] = Path(os.path.relpath(path, out)).as_posix()
            except ValueError:  # different drives on Windows
                pass
    if a.photos is None:
        raise SystemExit("--photos is required (or recalculate a completed run).")
    if a.pot_diameter_cm is None:
        a.pot_diameter_cm = saved.get("pot_diameter_cm")
    if a.pot_diameter_cm is not None and (not math.isfinite(a.pot_diameter_cm) or a.pot_diameter_cm <= 0):
        raise SystemExit("--pot-diameter-cm must be a positive, finite number.")
    a.area_unit = a.area_unit or saved.get("area_unit", "cm")
    if a.area_unit not in AREA_UNITS:
        raise SystemExit("Area unit must be cm or in.")
    try:
        a.summary_stats, a.percentiles = summary_choices(
            saved.get("summary_stats") if a.summary_stats is None else a.summary_stats,
            saved.get("percentiles") if a.percentiles is None else a.percentiles,
        )
        a.summary_overrides = summary_overrides(
            saved.get("summary_overrides") if a.summary_overrides is None else a.summary_overrides
        )
    except ValueError as error:
        raise SystemExit(str(error)) from None
    settings.update(no_nir=a.no_nir, pot_diameter_cm=a.pot_diameter_cm, area_unit=a.area_unit,
                    summary_stats=a.summary_stats, percentiles=a.percentiles, summary_overrides=a.summary_overrides)
    return settings


def main():
    a = parse_args()
    out = Path(a.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        with results_lock(out):
            run(a, out)
    except Timeout:
        raise SystemExit(f"Another run or correction is writing to {out}.") from None


def run(a, out):
    sfile = out / "run_settings.json"
    old_pots = previous_pots(out)
    names = MEASUREMENT_OUTPUTS + (sfile.name,)
    if not old_pots and any((out / name).exists() or (out / name).is_symlink()
                            for name in AUTOMATIC_OUTPUTS + names):
        raise SystemExit("This location already contains files named like CanopySplit outputs. Choose a different results folder.")
    saved = json.loads(sfile.read_text(encoding="utf-8")) if sfile.is_file() else {}
    settings = settings_for(a, out, saved if a.recalculate else {})

    with TemporaryDirectory(prefix=".canopysplit-", dir=out) as tmp:
        staged = Path(tmp)
        photos = photo_files(a.photos, "RGB")
        if not photos:
            raise SystemExit(f"No RGB photos named like day1_RGB_group1.jpg in {a.photos}.")
        keys = [photo_key(PHOTO_NAME.fullmatch(path.name)) for path in photos]
        if len(keys) != len(set(keys)):
            raise SystemExit("Provide only one RGB photo for each group and day.")
        if not a.no_nir:
            nir_keys = [photo_key(PHOTO_NAME.fullmatch(path.name)) for path in photo_files(a.nir_dir or a.photos, "NIR")]
            if len(nir_keys) != len(set(nir_keys)):
                raise SystemExit("Provide only one NIR photo for each group and day.")
        settings["rgb_inputs"] = input_signature(photos)
        corrected = protect_corrected_photos(out, saved, settings, old_pots, a.recalculate)
        settings["detector_weights"] = (saved.get("detector_weights") if a.recalculate
                                        else input_signature([a.weights or DEFAULT_WEIGHTS])[0])
        reuse = automatic_reuse(a, out, saved, settings, photos)
        generate(a, out, staged, reuse, old_pots, corrected)
        (staged / sfile.name).write_text(json.dumps(settings, indent=1), encoding="utf-8")
        publish_outputs(staged, out, names if reuse else AUTOMATIC_OUTPUTS + names)
    print(f"Results saved in {out}")


def automatic_reuse(a, out, saved, settings, photos):
    if not (a.recalculate or a.skip_existing):
        return False
    pots_file = out / "pots.csv"
    pots = read_rows(pots_file) if pots_file.is_file() else []
    available = automatic_available(out, pots)
    if a.recalculate:
        if not available:
            raise SystemExit("Automatic results are incomplete. Run Measure first, then recalculate.")
        if not {row["image"] for row in pots}.issubset({path.name for path in photos}):
            raise SystemExit("Some original RGB photos are missing. Open the original photo folder or run Measure again.")
        if saved.get("rgb_inputs") is not None and saved["rgb_inputs"] != settings["rgb_inputs"]:
            raise SystemExit("RGB photos have changed. Run Measure again to rebuild the masks.")
        return True
    return (
        available and Path(settings["photos"]) == saved_path(saved, "photos", out)
        and saved.get("rgb_inputs") == settings["rgb_inputs"]
        and saved.get("detector_weights") == settings["detector_weights"]
    )


def generate(a, out, work, reuse, old_pots, corrected):
    photos = a.photos
    nir = None if a.no_nir else a.nir_dir or photos
    auto = out if reuse else work
    det, grid, pots = auto / "detections.csv", auto / "grid.csv", auto / "pots.csv"
    masks, corr, final = auto / "masks", out / "corrections", work / "masks_final"
    traits, qc, ovl = work / "traits.csv", work / "images_qc.csv", work / "overlap.csv"
    check_images = work / "qc_images"

    if reuse:
        print("== Reusing pot detections, pot positions, and automatic masks")
    else:
        d_args = ["--images", photos, "--out", det] + (["--weights", a.weights] if a.weights else [])
        step("1/7 detect pots", "detect_pots.py", d_args)
        step("2/7 pot positions", "assign_grid.py", ["--centers-csv", det, "--out", grid, "--pots-out", pots])
        detected = read_rows(pots)
        if not detected:
            raise SystemExit("No measurable pots were detected in the RGB photos.")
        if corrected:
            check_corrected_positions(old_pots, detected, corrected)
        step("3/7 leaves per plant", "segment_leaves.py", ["--centers-csv", pots, "--images", photos, "--out-dir", masks])
    step("3b manual corrections", "apply_corrections.py", ["--auto", masks, "--corrections", corr, "--out", final])
    t_args = ["--centers-csv", pots, "--masks-dir", final, "--images", photos, "--out", traits,
              "--image-summary-out", qc, "--area-unit", a.area_unit]
    if nir is not None:
        t_args += ["--nir-dir", nir, "--nir-check-dir", check_images]
    if a.pot_diameter_cm is not None:
        t_args += ["--pot-diameter-cm", a.pot_diameter_cm]
    step("4/7 traits", "extract_traits.py", t_args)
    step("5/7 overlap score", "overlap_score.py", ["--centers-csv", pots, "--masks-dir", final, "--out", ovl])

    # merge: photos corrected by hand count as reviewed and reliable
    ov_rows = read_rows(ovl)
    corrected = {Path(r["image"]).stem for r in ov_rows if (corr / Path(r["image"]).stem / "done.json").is_file()}
    for r in ov_rows:
        r["auto_reliable"] = r.get("auto_reliable") or r["reliable"]
        manual = Path(r["image"]).stem in corrected
        r["reviewed"] = "yes" if manual else "no"
        r["reliable"] = "yes" if manual else r["auto_reliable"]
    with open(ovl, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "pot_id", "overlap_frac", "in_shared_mat", "image_overlap_frac",
                                          "auto_reliable", "reviewed", "reliable"], extrasaction="ignore")
        w.writeheader()
        w.writerows(ov_rows)
    flags = {(r["image"], r["pot_id"]): r for r in ov_rows}
    rows = read_rows(traits)
    if not rows:
        raise SystemExit("No plant measurements were generated.")
    extra = ("mask_source", "reliable")
    fields = [f for f in rows[0] if f not in extra] + list(extra)
    for r in rows:
        o = flags.get((r["image"], r["pot_id"]), {})
        r["reliable"] = o.get("reliable", "")
        r["mask_source"] = "manual" if Path(r["image"]).stem in corrected else "auto"
    with open(traits, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_bad = sum(r["reliable"] == "no" for r in rows)

    step("6/7 check images", "qc_images.py", ["--images", photos, "--grid", grid, "--masks-dir", final,
                                              "--overlap", ovl, "--out-dir", check_images])
    s_args = ["--traits", traits, "--out-dir", work / "summary"]
    if a.stages:
        s_args += ["--stages", a.stages]
    if a.genotypes:
        s_args += ["--genotypes", a.genotypes]
    s_args += ["--summary-stats", *a.summary_stats, "--percentiles", *a.percentiles]
    if a.summary_overrides:
        s_args += ["--summary-overrides", json.dumps(a.summary_overrides)]
    step("7/7 summary", "summarize.py", s_args)

    print(f"\nDone: {len(rows)} plant rows -> {out / traits.name}")
    print(f"  {len(corrected)} photos corrected by hand; {n_bad} plants still flagged for manual check (leaf overlap)")
    print(f"  check images: {out / 'qc_images'}   summary: {out / 'summary'}")
    if n_bad:
        print(f"  to correct: python app.py --results {out}   (tab 3 Correct)")


if __name__ == "__main__":
    main()
