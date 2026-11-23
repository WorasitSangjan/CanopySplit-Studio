# SPDX-License-Identifier: AGPL-3.0-only

"""
Step 7: summary table and charts per group and day.

Uses only plants with reliable = yes. Optional information from the
experiment team is joined in:
  --stages     CSV with columns  treatment, day, stage      (e.g. prechill / chilling / recovery)
  --genotypes  CSV with columns  treatment, pot_id, genotype
The joined per-plant table is written as traits_annotated.csv.

Outputs (in --out-dir):
  traits_annotated.csv   per-plant table with stage / genotype columns
  summary.csv            per group and day (and stage): valid count and selected statistics
  summary_<trait>.png    one line per group, with a band when two or more percentiles are selected

Usage:
    python summarize.py --traits traits.csv [traits2.csv ...] --out-dir summary/ [--stages stages.csv]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.ticker import MaxNLocator

if __package__:
    from .inputs import read_rows
    from .traits import (DEFAULT_STATISTICS, SUMMARY_STATISTICS, SUMMARY_TRAITS, percentile_columns,
                         summary_choices, summary_overrides, summary_trait_key)
else:
    from inputs import read_rows
    from traits import (DEFAULT_STATISTICS, SUMMARY_STATISTICS, SUMMARY_TRAITS, percentile_columns,
                        summary_choices, summary_overrides, summary_trait_key)

STAT_FUNCTIONS = {"mean": np.mean, "median": np.median, "min": min, "max": max}

CHART_TRAITS = {
    "leaf_area_cm2", "leaf_area_in2", "leaf_area_pot_frac", "VARI", "GLI", "NGRDI", "ExG",
    "NDVI", "NIR_rel", "damaged_frac", "dmg_brown_area_pot",
    "dmg_pinkred_area_pot", "dmg_yellow_area_pot",
}
LEGEND_FALLBACKS = (
    "Noto Sans Thai", "Thonburi", "Leelawadee UI", "Tahoma",
    "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "Arial Unicode MS",
)


def parse_args():
    p = argparse.ArgumentParser(description="Summary table and charts")
    p.add_argument("--traits", nargs="+", required=True, help="One or more traits.csv files")
    p.add_argument("--out-dir", default="summary")
    p.add_argument("--stages", default=None, help="CSV: treatment, day, stage")
    p.add_argument("--genotypes", default=None, help="CSV: treatment, pot_id, genotype")
    p.add_argument("--summary-stats", nargs="*", choices=DEFAULT_STATISTICS, help="Statistics across plants")
    p.add_argument("--percentiles", nargs="*", type=float, help="Percentiles across plants, from 0 to 100")
    p.add_argument("--summary-overrides", type=json.loads, help="JSON statistic and percentile choices by measurement")
    p.add_argument("--all-plants", action="store_true", help="Also use plants flagged reliable = no")
    p.add_argument("--min-plants", type=int, default=5, help="Leave out groups with fewer reliable plants on a given day")
    return p.parse_args()


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def write_summary_chart(rows, key, label, statistic, stat_label, percentiles, output):
    """Render a single group-by-day chart from computed or saved summary rows."""
    grouped = defaultdict(list)
    for row in rows:
        try:
            day = int(row["day"])
        except (KeyError, TypeError, ValueError):
            continue
        grouped[row.get("treatment") or ""].append((day, row))

    figure = Figure(figsize=(8, 4.2), dpi=150)
    ax = figure.subplots()
    series_key = f"{key}_{statistic}"
    band = len(percentiles) >= 2
    show_band = False
    plot_days = set()
    cmap = colormaps["tab10"]
    for i, (group, samples) in enumerate(sorted(grouped.items())):
        days, data = zip(*sorted(samples, key=lambda pair: pair[0]))
        values = np.array([num(row.get(series_key)) for row in data], dtype=float)
        present = np.isfinite(values)
        if not present.any():
            continue
        plot_days.update(day for day, valid in zip(days, present) if valid)
        color = "#175b43" if len(grouped) == 1 else cmap(i % cmap.N)
        if band:
            low = np.array([num(row.get(f"{key}_{percentiles[0][0]}")) for row in data], dtype=float)
            high = np.array([num(row.get(f"{key}_{percentiles[-1][0]}")) for row in data], dtype=float)
            bounds = present & np.isfinite(low) & np.isfinite(high)
            if bounds.any():
                show_band = True
                ax.fill_between(days, low, high, where=bounds, color=color, alpha=0.15, lw=0)
                isolated = bounds & ~np.r_[False, bounds[:-1]] & ~np.r_[bounds[1:], False]
                if isolated.any():
                    ax.vlines(np.array(days)[isolated], low[isolated], high[isolated],
                              colors=[color], alpha=0.35, lw=3)
        ax.plot(days, values, "-o", color=color, ms=3.5, lw=2, label=group)

    if not plot_days:
        return False
    if len(plot_days) == 1:
        ax.set_xticks(sorted(plot_days))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Experiment day", color="#364e43")
    ax.set_ylabel(label, color="#364e43")
    ax.tick_params(colors="#364e43")
    ax.grid(axis="y", color="#d6e0d9", linewidth=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#819387")
    installed_fonts = {font.name for font in fontManager.ttflist}
    families = ["DejaVu Sans", *(font for font in LEGEND_FALLBACKS if font in installed_fonts)]
    legend = ax.legend(frameon=False, prop={"family": families, "size": 9}, labelcolor="#172e25")
    for text in legend.get_texts():
        text.set_parse_math(False)
    band_label = f" (range: {percentiles[0][1]} to {percentiles[-1][1]})" if show_band else ""
    ax.set_title(f"{label} · {stat_label}{band_label}", color="#172e25", fontsize=11, fontweight="bold", loc="left")
    figure.tight_layout()
    figure.savefig(output, format="png")
    return True


def main():
    a = parse_args()
    try:
        selected_stats, percentiles = summary_choices(a.summary_stats, a.percentiles)
        overrides = summary_overrides(a.summary_overrides)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for t in a.traits for r in read_rows(t)]
    stages = {(r["treatment"].casefold(), int(r["day"])): r["stage"] for r in read_rows(a.stages)} if a.stages else {}
    genos = {(r["treatment"].casefold(), r["pot_id"]): r["genotype"] for r in read_rows(a.genotypes)} if a.genotypes else {}
    for r in rows:
        r["stage"] = stages.get((r["treatment"].casefold(), int(r["day"])), "")
        r["genotype"] = genos.get((r["treatment"].casefold(), r["pot_id"]), "")
    fields = []
    for r in rows:  # tables from different runs may have different columns
        fields += [k for k in r if k not in fields]
    with open(out / "traits_annotated.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    use = [r for r in rows if a.all_plants or r.get("reliable", "yes") != "no"]
    groups = defaultdict(list)
    for r in use:
        groups[(r["treatment"], int(r["day"]), r["stage"])].append(r)
    traits = [(k, lab) for k, lab in SUMMARY_TRAITS if any(num(r.get(k)) is not None for r in use)]
    defaults = {"summary_stats": selected_stats, "percentiles": percentiles}
    choices = {}
    for key, _ in traits:
        choice = overrides.get(summary_trait_key(key), defaults)
        items = percentile_columns(choice["percentiles"])
        choices[key] = (choice["summary_stats"], choice["percentiles"], items)

    sum_rows = []
    n_small = sum(len(rs) < a.min_plants for rs in groups.values())
    for (tr, day, stage), rs in sorted(groups.items()):
        if len(rs) < a.min_plants:
            continue
        row = {"treatment": tr, "day": day, "stage": stage, "n_plants": len(rs)}
        for k, _ in traits:
            # pots partly outside the photo: colour/damage share are valid, areas are not
            area_trait = "area" in k
            values = [value for r in rs if not (area_trait and r.get("edge_cut") == "yes")
                      and (value := num(r.get(k))) is not None]
            row[f"{k}_n"] = len(values)
            if values:
                measurement_stats, measurement_percentiles, items = choices[k]
                stats = {key: STAT_FUNCTIONS[key](values) for key in measurement_stats}
                if items:
                    stats.update(zip((key for key, _ in items), np.percentile(values, measurement_percentiles)))
                row.update({f"{k}_{stat}": round(float(value), 5) for stat, value in stats.items()})
        sum_rows.append(row)
    keys = ["treatment", "day", "stage", "n_plants"]
    for key, _ in traits:
        measurement_stats, _, items = choices[key]
        keys.extend(f"{key}_{stat}" for stat in ["n", *measurement_stats, *(name for name, _ in items)])
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sum_rows)

    chart_count = 0
    for key, label in traits:
        if key not in CHART_TRAITS:
            continue
        measurement_stats, _, items = choices[key]
        chart_stat = next((stat for stat in ("median", "mean", "min", "max") if stat in measurement_stats), None)
        if chart_stat is None and items:
            chart_stat = items[len(items) // 2][0]
        if chart_stat is None:
            continue
        chart_label = dict((*SUMMARY_STATISTICS, *items))[chart_stat]
        if write_summary_chart(sum_rows, key, label, chart_stat, chart_label, items, out / f"summary_{key}.png"):
            chart_count += 1
    print(f"{len(use)} plant rows used ({len(rows) - len(use)} unreliable left out); "
          f"{len(sum_rows)} group and day combinations ({n_small} left out: fewer than {a.min_plants} plants); "
          f"{chart_count} charts -> {out}")


if __name__ == "__main__":
    main()
