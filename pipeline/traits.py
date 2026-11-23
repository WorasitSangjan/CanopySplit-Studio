# SPDX-License-Identifier: AGPL-3.0-only

"""Labels and definitions for per-plant measurements and group statistics."""

from decimal import Decimal

AREA_UNITS = ("cm", "in")
CM_PER_INCH = 2.54

SUMMARY_TRAITS = (
    ("leaf_area_cm2", "Leaf area (cm²)"),
    ("leaf_area_in2", "Leaf area (in²)"),
    ("leaf_area_pot_frac", "Leaf area (fraction of pot)"),
    ("leaf_area_px", "Leaf area (pixels)"),
    ("VARI", "VARI"),
    ("GLI", "GLI"),
    ("NGRDI", "NGRDI"),
    ("ExG", "ExG"),
    ("R_rel", "Relative red"),
    ("G_rel", "Relative green"),
    ("B_rel", "Relative blue"),
    ("NDVI", "Relative NDVI"),
    ("NDVI_green_tissue", "Relative NDVI (RGB-classified green tissue)"),
    ("NDVI_damaged_tissue", "Relative NDVI (RGB-classified damaged tissue)"),
    ("NIR_rel", "Relative near-infrared"),
    ("damaged_frac", "Damaged share of leaf"),
    ("damaged_area_cm2", "Damaged leaf area (cm²)"),
    ("damaged_area_in2", "Damaged leaf area (in²)"),
    ("dmg_brown_frac", "Brown damage (share of leaf)"),
    ("dmg_pinkred_frac", "Pink/red damage (share of leaf)"),
    ("dmg_yellow_frac", "Yellow damage (share of leaf)"),
    ("dmg_brown_area_pot", "Brown damage (fraction of pot)"),
    ("dmg_pinkred_area_pot", "Pink/red damage (fraction of pot)"),
    ("dmg_yellow_area_pot", "Yellow damage (fraction of pot)"),
)

PROJECTED_LEAF_AREA_HELP = (
    "Visible leaf area viewed from above. CanopySplit uses the pot’s top diameter "
    "as a scale and reports the result in the selected square unit."
)
PROJECTED_DAMAGE_AREA_HELP = (
    "Area of leaf pixels with low greenness in RGB, expressed in the selected square unit. "
    "Treat this as an estimate of visible damage based on colour."
)

MEASUREMENT_HELP = {
    "leaf_area_cm2": PROJECTED_LEAF_AREA_HELP,
    "leaf_area_in2": PROJECTED_LEAF_AREA_HELP,
    "leaf_area_pot_frac": "Visible leaf area divided by the area of the detected pot’s bounding box.",
    "leaf_area_px": "The number of RGB pixels assigned to the plant’s visible leaves in the overhead photo.",
    "VARI": "Visible Atmospherically Resistant Index. Compares green with red and uses blue in the "
            "normalization. CanopySplit averages the index across the plant’s RGB leaf pixels.",
    "GLI": "Green Leaf Index. Compares green with red and blue relative to the combined channel signals.",
    "NGRDI": "Normalized Green–Red Difference Index. Measures the difference between green and red "
             "relative to their sum.",
    "ExG": "Excess Green. Measures how much twice the green signal exceeds red and blue combined.",
    "R_rel": "Mean red signal across the plant’s visible leaves, standardized against a white reference.",
    "G_rel": "Mean green signal across the plant’s visible leaves, standardized against a white reference.",
    "B_rel": "Mean blue signal across the plant’s visible leaves, standardized against a white reference.",
    "NDVI": "Compares estimated near-infrared with red using (NIR − red) ÷ (NIR + red). "
            "This is the average over all usable leaf pixels in an aligned RGB and NoIR pair. "
            "The value is relative to the camera setup and calibration.",
    "NDVI_green_tissue": "Relative NDVI averaged over usable pixels that the RGB photo classifies as green. "
                         "The RGB colour determines which pixels count as green, and the paired photos "
                         "provide the red and near-infrared signals.",
    "NDVI_damaged_tissue": "Relative NDVI averaged over usable pixels that the RGB photo classifies as "
                           "damaged based on low greenness. It uses the same formula as green-tissue NDVI. "
                           "Natural colour differences can also affect the RGB classification.",
    "NIR_rel": "Mean estimated near-infrared signal from usable leaf pixels in the aligned RGB and NoIR pair, "
               "after adjusting for visible light and using a white reference.",
    "damaged_frac": "Share of the plant’s visible leaf area that the RGB greenness rule classifies as damaged. "
                    "For example, 0.20 means 20% of the visible leaf area.",
    "damaged_area_cm2": PROJECTED_DAMAGE_AREA_HELP,
    "damaged_area_in2": PROJECTED_DAMAGE_AREA_HELP,
    "dmg_brown_frac": "Brown pixels identified by the RGB damage colour rules, divided by the full visible leaf area.",
    "dmg_pinkred_frac": "Pink and red pixels identified by the RGB damage colour rules, divided by the full visible leaf area.",
    "dmg_yellow_frac": "Yellow pixels identified by the RGB damage colour rules, divided by the full visible leaf area.",
    "dmg_brown_area_pot": "Area of brown pixels identified by the RGB damage colour rules, divided by the "
                         "area of the detected pot’s bounding box.",
    "dmg_pinkred_area_pot": "Area of pink and red pixels identified by the RGB damage colour rules, divided "
                           "by the area of the detected pot’s bounding box.",
    "dmg_yellow_area_pot": "Area of yellow pixels identified by the RGB damage colour rules, divided by the "
                          "area of the detected pot’s bounding box.",
}

SUMMARY_STATISTICS = (
    ("n", "Valid plants"),
    ("mean", "Mean"),
    ("median", "Median"),
    ("min", "Min"),
    ("max", "Max"),
)

DEFAULT_STATISTICS = tuple(name for name, _ in SUMMARY_STATISTICS if name != "n")
DEFAULT_PERCENTILES = (5, 25, 75, 95)


def summary_choices(statistics=None, percentiles=None):
    """Validate and order the selections, retaining the existing CLI defaults."""
    statistics = DEFAULT_STATISTICS if statistics is None else statistics
    if not isinstance(statistics, (list, tuple)) or any(name not in DEFAULT_STATISTICS for name in statistics):
        raise ValueError("Choose statistics from mean, median, minimum, and maximum.")
    selected = [name for name in DEFAULT_STATISTICS if name in statistics]

    percentiles = DEFAULT_PERCENTILES if percentiles is None else percentiles
    if not isinstance(percentiles, (list, tuple)):
        raise ValueError("Enter percentiles as numbers from 0 to 100.")
    try:
        values = sorted(float(value) for value in percentiles)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Enter percentiles as numbers from 0 to 100.") from None
    if any(not 0 <= value <= 100 for value in values):
        raise ValueError("Enter percentiles from 0 to 100.")
    if len(values) != len(set(values)):
        raise ValueError("Enter a different number for each percentile.")
    return selected, [value or 0.0 for value in values]


def percentile_columns(percentiles):
    numbers = (format(Decimal(str(float(value))), "f").removesuffix(".0") for value in percentiles)
    return [(f"q{number}", f"P{number}") for number in numbers]


def summary_trait_key(measurement):
    """Use the same summary settings for a physical area in either unit."""
    return f"{measurement[:-4]}_cm2" if measurement.endswith("_in2") else measurement


def summary_overrides(choices):
    """Validate per-measurement choices and normalize the measurement keys."""
    if choices is None:
        return {}
    if not isinstance(choices, dict):
        raise ValueError("Choose statistics by measurement.")
    labels = dict(SUMMARY_TRAITS)
    overrides = {}
    for measurement, selection in choices.items():
        if measurement not in labels or not isinstance(selection, dict):
            raise ValueError("Choose a supported measurement for the statistics.")
        key = summary_trait_key(measurement)
        if key in overrides:
            raise ValueError(f"Choose one set of statistics for {labels[key]}.")
        try:
            stats = selection.get("summary_stats", [])
            percentiles = selection.get("percentiles", [])
            if stats is None or percentiles is None:
                raise ValueError("Use lists of statistics and percentiles.")
            stats, percentiles = summary_choices(stats, percentiles)
            if len(percentiles) > 3:
                raise ValueError("Enter up to three percentiles.")
        except ValueError as error:
            raise ValueError(f"{labels[measurement]}: {error}") from None
        overrides[key] = {"summary_stats": stats, "percentiles": percentiles}
    return overrides
