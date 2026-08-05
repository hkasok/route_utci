"""Verification suite for absorbed radiant-flux recording and ranked plots.

Run: ``python3 verify_radiant_flux_contributions.py``
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_flux_contributions import (
    DEFAULT_CONFIG, LW_SOURCE_COLUMNS, PRIMARY_COLUMNS, SW_SOURCE_COLUMNS,
    aggregate_plot_categories, canonical_lw_source, canonical_sw_source,
    colors_for_order, mrt_from_absorbed_flux_c,
    independently_ranked_columns, order_categories_by_route_mean,
    plot_ranked_radiant_flux_contributions,
    plot_surface_longwave_classifications,
    route_contribution_summary, sample_route_contribution_matrices,
    sorted_visualization_copy, validate_contribution_arrays)


PASS = 0
FAIL = 0
SIGMA = 5.670374419e-8
EPS = 0.97


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        status = "PASS"
    else:
        FAIL += 1
        status = "FAIL"
    print(f"  [{status}] {name}" + (f" ({detail})" if detail else ""))


def synthetic_record() -> dict[str, np.ndarray]:
    direct = np.array([0.0, 120.0, 40.0, 0.0])
    diffuse = np.array([0.0, 30.0, 25.0, 0.0])
    reflected = np.array([0.0, 20.0, 10.0, 0.0])
    sky_lw = np.array([250.0, 210.0, 220.0, 245.0])
    surface_lw = np.array([135.0, 190.0, 180.0, 140.0])
    record = {
        PRIMARY_COLUMNS[0]: direct,
        PRIMARY_COLUMNS[1]: diffuse,
        PRIMARY_COLUMNS[2]: reflected,
        PRIMARY_COLUMNS[3]: sky_lw,
        PRIMARY_COLUMNS[4]: surface_lw,
        "sw_total_absorbed_Wm2": direct + diffuse + reflected,
        "lw_total_absorbed_Wm2": sky_lw + surface_lw,
        "total_absorbed_radiant_flux_Wm2": (
            direct + diffuse + reflected + sky_lw + surface_lw),
    }
    for key in SW_SOURCE_COLUMNS:
        record[key] = np.zeros(4)
    record[SW_SOURCE_COLUMNS[0]] = 0.60 * reflected
    record[SW_SOURCE_COLUMNS[2]] = 0.25 * reflected
    record[SW_SOURCE_COLUMNS[4]] = 0.15 * reflected
    for key in LW_SOURCE_COLUMNS:
        record[key] = np.zeros(4)
    record[LW_SOURCE_COLUMNS[0]] = 0.30 * surface_lw
    record[LW_SOURCE_COLUMNS[2]] = 0.20 * surface_lw
    record[LW_SOURCE_COLUMNS[4]] = 0.25 * surface_lw
    record[LW_SOURCE_COLUMNS[6]] = 0.15 * surface_lw
    record[LW_SOURCE_COLUMNS[8]] = 0.10 * surface_lw
    return record


print("=" * 72)
print("T1: conservation and MRT reconstruction")
record = synthetic_record()
total = record["total_absorbed_radiant_flux_Wm2"]
mrt = mrt_from_absorbed_flux_c(total, EPS, SIGMA)
report = validate_contribution_arrays(record, expected_mrt_c=mrt,
                                      person_emissivity=EPS, sigma=SIGMA)
check("primary shortwave summation", report["maximum_sw_closure_error_Wm2"] == 0.0)
check("classified reflected shortwave conservation",
      np.allclose(sum(record[key] for key in SW_SOURCE_COLUMNS), record[PRIMARY_COLUMNS[2]]))
check("classified surface longwave conservation",
      np.allclose(sum(record[key] for key in LW_SOURCE_COLUMNS), record[PRIMARY_COLUMNS[4]]))
check("total absorbed-flux conservation", report["maximum_total_closure_error_Wm2"] == 0.0)
check("MRT consistency with total flux", report["maximum_mrt_reconstruction_error_C"] == 0.0)
check("complete shade has zero direct shortwave", record[PRIMARY_COLUMNS[0]][0] == 0.0)
check("night has zero total shortwave",
      record["sw_total_absorbed_Wm2"][[0, 3]].sum() == 0.0)

print("\nT2: independent sorting, line order, and aggregation")
frame = pd.DataFrame(record)
frame.insert(0, "original_route_index", np.arange(len(frame)))
original = frame["original_route_index"].to_numpy(copy=True)
ranked = sorted_visualization_copy(frame)
check("descending receptor sorting",
      np.all(np.diff(ranked["total_absorbed_radiant_flux_Wm2"]) <= 0))
check("original route order preserved",
      np.array_equal(frame["original_route_index"], original))
primary_plot, primary_columns, _ = aggregate_plot_categories(frame, "primary", DEFAULT_CONFIG)
check("default main plot retains combined surface longwave",
      DEFAULT_CONFIG["plot_mode"] == "primary"
      and PRIMARY_COLUMNS[4] in primary_columns)
order = order_categories_by_route_mean(primary_plot, primary_columns)
means = [primary_plot[key].mean() for key in order]
check("legend ordered by descending route mean", np.all(np.diff(means) <= 0))
check("largest mean contribution receives first line", order[0] == PRIMARY_COLUMNS[3])
independent = independently_ranked_columns(frame, primary_columns)
check("every contribution is independently descending",
      all(np.all(np.diff(independent[key]) <= 0) for key in primary_columns))
check("independent ranks do not imply simultaneous receptor values",
      not np.allclose(independent[PRIMARY_COLUMNS].sum(axis=1),
                      np.sort(total)[::-1]))
lw_plot, lw_columns, lw_grouped = aggregate_plot_categories(
    frame, "surface_longwave_resolved", DEFAULT_CONFIG)
positive_lw = [key for key in LW_SOURCE_COLUMNS if frame[key].mean() > 1.0e-12]
descending_lw = independently_ranked_columns(
    frame, positive_lw, ascending=False)
check("surface-longwave classifications are independently descending",
      all(np.all(np.diff(descending_lw[key]) <= 0) for key in positive_lw))
check("surface longwave is replaced by all available classified sources",
      PRIMARY_COLUMNS[4] not in lw_columns
      and set(positive_lw).issubset(lw_columns)
      and not lw_grouped)
check("classified longwave line mode conserves original receptor totals",
      np.allclose(lw_plot[lw_columns].sum(axis=1), total))

cfg = {**DEFAULT_CONFIG,
       "material_resolved": {**DEFAULT_CONFIG["material_resolved"],
                             "maximum_categories": 5,
                             "minimum_mean_fraction_percent": 5.0}}
material_plot, material_columns, grouped = aggregate_plot_categories(
    frame, "material_resolved", cfg)
check("minor material categories grouped into Other", bool(grouped) and "Other" in material_columns)
check("conservation after Other aggregation",
      np.allclose(material_plot[material_columns].sum(axis=1), total))
check("material-specific aggregation retains major source",
      SW_SOURCE_COLUMNS[0] in material_columns or SW_SOURCE_COLUMNS[0] in grouped)

print("\nT3: optional sources, interpolation closure, and colors")
check("current grass material maps to grass SW source",
      canonical_sw_source("grass_lawn") == "sw_reflected_grass_absorbed_Wm2")
check("current grass material maps to grass LW source",
      canonical_lw_source("grass_lawn") == "lw_grass_absorbed_Wm2")
primary_only = {key: record[key] for key in PRIMARY_COLUMNS + [
    "sw_total_absorbed_Wm2", "lw_total_absorbed_Wm2",
    "total_absorbed_radiant_flux_Wm2"]}
try:
    validate_contribution_arrays(primary_only, expected_mrt_c=mrt,
                                 person_emissivity=EPS, sigma=SIGMA)
    optional_ok = True
except Exception:
    optional_ok = False
check("missing optional source classifications handled", optional_ok)

matrices = {key: np.vstack([value, 1.1 * value]) for key, value in record.items()}
time_hours = np.array([0.0, 12.0])
point_indices = np.array([0, 1, 2, 3])
arrival = np.array([0.0, 3.0, 6.0, 9.0])
raw_total = np.array([
    np.interp(hour, time_hours, matrices["total_absorbed_radiant_flux_Wm2"][:, point], period=24.0)
    for hour, point in zip(arrival, point_indices)])
target_mrt = mrt_from_absorbed_flux_c(raw_total * np.array([1.0, 1.01, 0.99, 1.02]), EPS, SIGMA)
sampled, scale = sample_route_contribution_matrices(
    matrices, time_hours, arrival, point_indices, target_mrt,
    person_emissivity=EPS, sigma=SIGMA, validation=DEFAULT_CONFIG["validation"])
check("route-time closure normalization reconstructs authoritative MRT",
      np.allclose(mrt_from_absorbed_flux_c(
          sampled["total_absorbed_radiant_flux_Wm2"], EPS, SIGMA), target_mrt))
check("route interpolation remains in input order",
      len(scale) == 4 and np.isclose(scale[0], 1.0)
      and np.allclose(scale[1:], [1.01, 0.99, 1.02]))
colors_a = colors_for_order(order, "ranked_emphasis", DEFAULT_CONFIG)
colors_b = colors_for_order(order, "ranked_emphasis", DEFAULT_CONFIG)
check("largest ranked contribution is red",
      colors_a[0] == DEFAULT_CONFIG["largest_contribution_color"])
check("colors and legend order deterministic", colors_a == colors_b)

print("\nT4: synthetic export and plotting integration")
frame["route_id"] = 1
frame["point_id"] = np.arange(len(frame))
frame["distance_along_route_m"] = np.arange(len(frame), dtype=float)
frame["mrt_C"] = mrt
with tempfile.TemporaryDirectory(prefix="trec_flux_verify_") as temporary:
    output = Path(temporary)
    exported = output / "route_1_radiant_flux_contributions.csv"
    frame.to_csv(exported, index=False)
    figures = plot_ranked_radiant_flux_contributions(
        {1: frame}, output, DEFAULT_CONFIG,
        route_annotations={1: "Length 3 m; walking time 0.1 min"})
    surface_lw_figures, surface_lw_classes = plot_surface_longwave_classifications(
        {1: frame}, output, DEFAULT_CONFIG,
        route_annotations={1: "Length 3 m; walking time 0.1 min"})
    summary = route_contribution_summary({1: frame}, DEFAULT_CONFIG)
    summary.to_csv(output / "route_radiant_flux_contribution_summary.csv", index=False)
    loaded = pd.read_csv(exported)
    check("receptor export contains expected absorbed-flux fields",
          set(PRIMARY_COLUMNS + ["total_absorbed_radiant_flux_Wm2"]).issubset(loaded.columns))
    check("no additive MRT contribution columns exported",
          not any("mrt_contribution" in key.lower() for key in loaded.columns))
    check("line-only PDF and 300-dpi PNG figure outputs created",
          any(path.suffix == ".pdf" for path in figures)
          and any(path.suffix == ".png" for path in figures)
          and all(path.is_file() for path in figures))
    check("separate surface-longwave classification figures created",
          any("surface_longwave_by_class_descending" in path.name
              for path in surface_lw_figures)
          and all(path.is_file() for path in surface_lw_figures))
    check("surface-longwave figure contains only available nonzero classes",
          set(surface_lw_classes["1"]) == set(positive_lw))
    check("summary exports W m-2 statistics and fractions",
          {"mean_absorbed_flux_Wm2", "mean_fraction_percent", "stack_rank"}.issubset(summary.columns))

print("\n" + "=" * 72)
print(f"RESULT: {PASS} passed, {FAIL} failed")
if FAIL:
    raise SystemExit(1)
