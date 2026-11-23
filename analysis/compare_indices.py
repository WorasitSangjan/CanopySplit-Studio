# SPDX-License-Identifier: AGPL-3.0-only

"""
Compare relative NDVI with colour-only vegetation indices (ExG, VARI, GLI,
NGRDI) on automatically measured plants only (Control from day 6 excluded:
leaf overlap needs the manual correction tool).

  1. Agreement: Spearman correlation between indices (all plant-days).
  2. Repeatability: same plant on consecutive days (Spearman rho).
  3. Damage: Spearman rho of each index with the plant's damaged-leaf fraction.
  4. Treatment difference on shared days: Cliff's delta (-1..1; |d| > 0.47 is
     usually called large) and Mann-Whitney p.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

IDX = ["NDVI", "VARI", "GLI", "NGRDI", "ExG", "NIR_rel"]
CONTROL_AUTO_LAST_DAY = 5


def value(row, key):
    try:
        return float(row.get(key) or "nan")
    except ValueError:
        return np.nan


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    rho = (spearmanr(x, y).correlation
           if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan)
    return rho, len(x)


def format_rho(rho):
    return f"{rho:.2f}" if np.isfinite(rho) else "n/a"


def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a) * len(b))

def main():
    parser = argparse.ArgumentParser(description="Compare indices from the packaged maize study runs")
    parser.add_argument("results", type=Path, help="Folder containing chill_low, chill_high, and control results")
    parser.add_argument("out", type=Path, help="Markdown report path")
    args = parser.parse_args()
    rows = []
    for sub in ("chill_low", "chill_high", "control"):
        path = args.results / sub / "traits.csv"
        if not path.is_file():
            parser.error(f"Missing study results: {path}")
        with path.open(newline="", encoding="utf-8-sig") as file:
            for r in csv.DictReader(file):
                if r["treatment"].startswith("control") and int(r["day"]) > CONTROL_AUTO_LAST_DAY:
                    continue
                if r.get("mask_source") == "manual" or r.get("reliable") == "no":
                    continue
                if np.isfinite(value(r, "NDVI")) and value(r, "leaf_area_px") >= 500:
                    rows.append(r)
    print(f"plant-days used: {len(rows)}")

    lines = ["# NDVI vs colour-only indices (automatic plants only)", "",
             f"Plant-days used: {len(rows)} (chill-low all days, chill-high all days, Control days 1-{CONTROL_AUTO_LAST_DAY}).", ""]

    # 1. agreement
    columns = {key: [value(row, key) for row in rows] for key in IDX}
    lines += ["## 1. Agreement (Spearman rho, all plant-days)", "", "| | " + " | ".join(IDX) + " |", "|---" * (len(IDX) + 1) + "|"]
    for k in IDX:
        lines.append(f"| {k} | " + " | ".join(format_rho(correlation(columns[k], columns[j])[0]) for j in IDX) + " |")
    lines.append("")

    # 2. repeatability, 3. damage
    per = defaultdict(dict)
    for r in rows:
        per[(r["treatment"], r["pot_id"])][int(r["day"])] = r
    rep, dmg = {}, {}
    for k in IDX:
        x, y = [], []
        for dd in per.values():
            ds = sorted(dd)
            for a, b in zip(ds, ds[1:]):
                if b - a == 1:
                    x.append(value(dd[a], k))
                    y.append(value(dd[b], k))
        rep[k] = correlation(x, y)
        chill = [r for r in rows if not r["treatment"].startswith("control")]
        dmg[k], _ = correlation([value(r, k) for r in chill], [value(r, "damaged_frac") for r in chill])
    lines += ["## 2-3. Repeatability and damage", "",
              "| index | repeatability (consecutive days, rho) | link to damaged-leaf share (chill trays, rho) |", "|---|---|---|"]
    for k in IDX:
        lines.append(f"| {k} | {format_rho(rep[k][0])} (n={rep[k][1]}) | {format_rho(dmg[k])} |")
    lines.append("")

    # 4. treatment differences on shared days
    comparisons = [("control2", "chill-low1", [3, 4, 5]),
                   ("chill-low1", "chill-hi1", [8, 10, 11]),
                   ("chill-low1", "chill-hi2", [8, 10, 11]),
                   ("chill-hi1", "chill-hi2", [8, 9, 10, 11, 12, 13, 15])]
    lines += ["## 4. Treatment difference on shared days (Cliff's delta; + = first group higher)", "",
              "| comparison | days | " + " | ".join(IDX) + " |", "|---|---" + "|---" * len(IDX) + "|"]
    for g1, g2, days in comparisons:
        cells = []
        for k in IDX:
            a = [value(r, k) for r in rows if r["treatment"] == g1 and int(r["day"]) in days]
            b = [value(r, k) for r in rows if r["treatment"] == g2 and int(r["day"]) in days]
            a, b = [v for v in a if np.isfinite(v)], [v for v in b if np.isfinite(v)]
            if len(a) < 5 or len(b) < 5:
                cells.append("n/a"); continue
            p = mannwhitneyu(a, b).pvalue
            cells.append(f"{cliffs(a, b):+.2f}{'*' if p < 0.05 else ''}")
        lines.append(f"| {g1} vs {g2} | {','.join(map(str, days))} | " + " | ".join(cells) + " |")
    lines += ["", "\\* p < 0.05 (Mann-Whitney). Note: each tray is a separate experiment run, so tray differences include run effects, not only treatment.", ""]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
