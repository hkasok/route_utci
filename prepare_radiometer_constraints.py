#!/usr/bin/env python3
"""prepare_radiometer_constraints.py -- build the measurement-constrained layer.

Reads the Lisbon four-component radiometer series, quality-controls them,
derives the effective local radiative quantities, associates them with the
model's receptor points, and writes the constraint tables, the apples-to-apples
sensor validation and the diagnostic figures.

This is an ADDITIONAL layer. It does not modify the material-classification
system, the MRT model, the UI, geometry processing, JOS-3 or UTCI.

CONSTRAINED IS NOT VALIDATED
----------------------------
``--constraint-routes`` and ``--validation-routes`` split the six cases. Any
case named as a constraint is labelled ``constraint`` in every output and its
comparison statistics are reported as MEASUREMENT-CONSTRAINED, not as
independent validation. With the default (no split) every case is treated as
validation-only, i.e. the measurements are compared against a model that did
not see them -- which is the honest default, because nothing downstream
currently consumes the constraints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import radiometer_constraints as rc

DEFAULT_CASES = [f"lisbon{index}" for index in range(1, 7)]
CONSTRAINTS_CSV = "radiometer_constraints.csv"
METADATA_JSON = "radiometer_constraint_metadata.json"
PROFILES_CSV = "route_radiative_profiles.csv"
VALIDATION_CSV = "validation_four_component.csv"

# Model-side sensor-equivalent columns produced by stage 05. These are PLANAR
# hemispherical quantities on the instrument's own weighting -- no body
# projected-area factor, no human absorptivity, no human emissivity -- which is
# what makes the comparison apples-to-apples. See README_sensor_equivalents.md.
MODEL_CHANNEL_COLUMNS = {
    "SW_down": "sensor_shortwave_down_Wm2",
    "SW_up": "sensor_shortwave_up_Wm2",
    "LW_down": "sensor_longwave_down_Wm2",
    "LW_up": "sensor_longwave_up_Wm2",
}
# Body-absorbed columns that must NEVER appear on either side of this
# comparison. Guarded explicitly rather than by convention.
FORBIDDEN_BODY_COLUMNS = (
    "sw_direct_absorbed_Wm2", "sw_diffuse_sky_absorbed_Wm2",
    "sw_reflected_total_absorbed_Wm2", "lw_sky_absorbed_Wm2",
    "lw_surface_total_absorbed_Wm2", "sw_total_absorbed_Wm2",
    "lw_total_absorbed_Wm2", "total_absorbed_radiant_flux_Wm2",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Route-resolved four-component radiometer constraints")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Case ids (default: every lisbon case found).")
    parser.add_argument("--input-root", default="input")
    parser.add_argument("--output-root", default="run_output")
    parser.add_argument("--aggregate-output-dir",
                        default="run_output/radiometer_constraints",
                        help="Where the combined products go.")
    parser.add_argument("--sensor-height-m", type=float, default=1.0,
                        help="Radiometer height above local ground. The source "
                             "data does NOT record it, so this is the "
                             "configured validation sensor height and is "
                             "recorded as an assumption in the metadata.")
    parser.add_argument("--min-sw-down-for-albedo", type=float, default=50.0,
                        help="Minimum SW_down for an effective reflectance to "
                             "be computed at all (W/m2).")
    parser.add_argument("--interpolation", choices=rc.VALID_METHODS,
                        default="nearest",
                        help="Default 'nearest' preserves measured sun/shade "
                             "transitions instead of inventing intermediate "
                             "irradiances across them.")
    parser.add_argument("--max-measurement-gap-s", type=float, default=60.0)
    parser.add_argument("--max-measurement-gap-m", type=float, default=25.0)
    parser.add_argument("--constraint-routes", nargs="*", default=None,
                        help="Case ids whose measurements are treated as model "
                             "CONSTRAINTS. Their comparison is reported as "
                             "measurement-constrained, never as independent "
                             "validation.")
    parser.add_argument("--validation-routes", nargs="*", default=None,
                        help="Case ids held out for independent validation.")
    parser.add_argument("--thermal-dir-name", default="thermal_out",
                        help="Case subdirectory holding facets.npz. When it "
                             "exists, each measurement point records which "
                             "surface GROUPS its downward sensor actually sees "
                             "and in what share (section 10).")
    parser.add_argument("--no-surface-groups", action="store_true",
                        help="Skip the visible-surface-group association.")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args(argv)


def discover_cases(input_root: Path, selected):
    if selected:
        return list(selected)
    found = sorted(path.name for path in input_root.glob("lisbon*")
                   if (path / "measurements").is_dir())
    return found or DEFAULT_CASES


def load_model_channels(case_dir: Path):
    """Model sensor-equivalent channels paired to measurements by the existing
    validation pairing, so no new alignment logic is introduced."""
    path = case_dir / "validation" / "mrt_lisbon" / "radiant_flux_comparison_points.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    missing = [column for column in MODEL_CHANNEL_COLUMNS.values()
               if column not in frame.columns]
    if missing:
        return None
    return frame


def build_case(case_id: str, args, usage: str):
    """Constraints and validation rows for one case."""
    input_dir = Path(args.input_root) / case_id
    output_dir = Path(args.output_root) / case_id
    measurement_dir = input_dir / "measurements"
    if not measurement_dir.is_dir():
        return None, None, None

    qc_config = rc.QualityControlConfig(
        minimum_sw_down_for_reflectance_Wm2=args.min_sw_down_for_albedo)
    interpolation = rc.InterpolationConfig(
        method=args.interpolation,
        max_gap_s=args.max_measurement_gap_s,
        max_gap_m=args.max_measurement_gap_m)

    blocks = []
    digests = {}
    for path in sorted(measurement_dir.glob("route_*_experimental_measurements.csv")):
        frame = rc.load_route_measurements(path, case_id=case_id)
        constraints = rc.build_constraints(
            frame, sensor_height_m=args.sensor_height_m,
            qc_config=qc_config, usage=usage)
        constraints["route_file"] = path.name
        constraints["period"] = "day" if "day" in path.name else "night"
        blocks.append(constraints)
        digests[path.name] = rc.file_digest(path)
    if not blocks:
        return None, None, None
    constraints = pd.concat(blocks, ignore_index=True)

    # Association demonstration: carry each route's own measurements onto its
    # own receptor spacing, exercising the gap thresholds on real geometry.
    associated = []
    for (route_id, period), block in constraints.groupby(["route_id", "period"]):
        block = block.sort_values("distance_along_route_m")
        mapped = rc.interpolate_to_route(
            block, block["distance_along_route_m"].to_numpy(float),
            block["elapsed_s"].to_numpy(float), interpolation)
        mapped["case_id"] = case_id
        mapped["route_id"] = route_id
        mapped["period"] = period
        associated.append(mapped)
    associated = pd.concat(associated, ignore_index=True)

    validation = None
    model_frame = load_model_channels(output_dir)
    if model_frame is not None:
        rows = []
        for channel, column in MODEL_CHANNEL_COLUMNS.items():
            measured_column = {
                "SW_down": "measured_swin_wm2", "SW_up": "measured_swout_wm2",
                "LW_down": "measured_lwin_wm2", "LW_up": "measured_lwout_wm2",
            }[channel]
            if measured_column not in model_frame.columns:
                continue
            for scope, block in (("all", model_frame),
                                 *[(f"period_{value}",
                                    model_frame[model_frame["period"].eq(value)])
                                   for value in sorted(model_frame["period"].unique())]):
                stats = rc.channel_statistics(block[measured_column],
                                              block[column])
                if stats.get("n", 0) < 3:
                    continue
                stats.update({"case_id": case_id, "channel": channel,
                              "scope": scope, "usage": usage,
                              "measured_column": measured_column,
                              "model_column": column})
                rows.append(stats)
        validation = pd.DataFrame(rows)
    return constraints, associated, (validation, digests)


def associate_visible_surface_groups(constraints: pd.DataFrame,
                                     thermal_dir: Path, sensor_height_m: float):
    """Which surface groups the downward sensor sees at each measurement point.

    Reuses stage 05's OWN footprint kernel rather than re-deriving it, by
    handing that bound method the arrays it needs. Duplicating the cos*cos/d^2
    weighting here would create a second implementation that could drift from
    the one the emulated radiometer actually uses -- and the whole point of the
    footprint work was that a downward sensor is NOT well described by its
    nearest triangle.
    """
    facets_path = thermal_dir / "facets.npz"
    if not facets_path.is_file():
        return None
    facets = np.load(facets_path, allow_pickle=False)
    needed = {"centroid", "normal", "area", "cls"}
    if not needed <= set(facets.files):
        return None
    ground = facets["cls"] == 0
    if not ground.any():
        return None
    group_ids = (facets["surface_group_id"].astype(str)[ground]
                 if "surface_group_id" in facets.files
                 else np.array([f"ground_facet_{i}" for i in np.flatnonzero(ground)]))

    stage05 = _load_stage05()

    class _Stub:
        build_sensor_ground_footprint = (
            stage05.FacetLongwave.build_sensor_ground_footprint)

    stub = _Stub()
    stub._ground_mask = ground
    stub._facet_centroid = facets["centroid"].astype(float)
    stub._facet_area = facets["area"].astype(float)
    stub._facet_normal_ground = facets["normal"].astype(float)[ground]
    stub.facet_albedo = np.full(len(ground), 0.2)
    stub.facet_eps = np.full(len(ground), 0.95)
    stub.local_ground_albedo = None
    stub.args = argparse.Namespace(ground_albedo=0.18)
    stub._footprint = None
    stub.sensor_ground_albedo = None
    stub.sensor_ground_emissivity = None
    stub.sensor_footprint_coverage = 0.0
    stub.sensor_footprint_radius_m = None

    positions = np.column_stack([
        constraints["x"].to_numpy(float), constraints["y"].to_numpy(float),
        np.full(len(constraints), float(sensor_height_m))])
    if not stub.build_sensor_ground_footprint(positions, sensor_height_m,
                                              sensor_height_m):
        return None
    return rc.visible_group_weights(stub._footprint, group_ids)


def _load_stage05():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stage05_for_constraints",
        Path(__file__).resolve().parent / "05_mrt_network_raytrace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_route_profiles(constraints: pd.DataFrame, output: Path, dpi: int):
    """Section 19: the five along-route diagnostic panels per case."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    for (case_id, route_id, period), block in constraints.groupby(
            ["case_id", "route_id", "period"]):
        block = block.sort_values("distance_along_route_m")
        distance = block["distance_along_route_m"].to_numpy(float)
        panels = [
            ("SW_down_measured_Wm2", "Downwelling shortwave (W m$^{-2}$)", "#d62728"),
            ("SW_up_measured_Wm2", "Upwelling shortwave (W m$^{-2}$)", "#ff7f0e"),
            ("LW_down_measured_Wm2", "Downwelling longwave (W m$^{-2}$)", "#1f77b4"),
            ("LW_up_measured_Wm2", "Upwelling longwave (W m$^{-2}$)", "#2ca02c"),
            ("effective_lower_SW_reflectance",
             "Effective lower-hemisphere\nSW reflectance (-)", "#9467bd"),
        ]
        fig, axes = plt.subplots(len(panels), 1, figsize=(11, 12.5), sharex=True)
        for axis, (column, label, colour) in zip(axes, panels):
            values = block[column].to_numpy(float)
            axis.plot(distance, values, color=colour, lw=1.2)
            invalid = block["qc_flag"].to_numpy() == rc.QC_INVALID
            if invalid.any():
                axis.scatter(distance[invalid], values[invalid], s=12,
                             facecolors="none", edgecolors="black", lw=0.6,
                             label="QC invalid")
                axis.legend(fontsize=7)
            axis.set_ylabel(label, fontsize=8.5)
            axis.grid(alpha=0.25)
            finite = np.isfinite(values)
            if finite.any():
                axis.set_title(
                    f"mean {values[finite].mean():.3g}, "
                    f"range {values[finite].min():.3g} to {values[finite].max():.3g}",
                    fontsize=8, loc="right")
        axes[-1].set_xlabel("Distance along measured route (m)")
        fig.suptitle(f"{case_id} route {int(route_id)} ({period}): "
                     "four-component radiometer along the route", y=0.997)
        fig.tight_layout()
        stem = output / f"{case_id}_route_{int(route_id)}_{period}_four_component"
        for suffix in ("png", "pdf"):
            fig.savefig(f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_combined_validation(paired: pd.DataFrame, output: Path, dpi: int):
    """Section 20: measured vs modelled, one panel per channel, all cases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    for channel, model_column in MODEL_CHANNEL_COLUMNS.items():
        measured_column = {
            "SW_down": "measured_swin_wm2", "SW_up": "measured_swout_wm2",
            "LW_down": "measured_lwin_wm2", "LW_up": "measured_lwout_wm2",
        }[channel]
        if measured_column not in paired.columns:
            continue
        fig, axis = plt.subplots(figsize=(6.4, 6.4))
        for case_id, block in paired.groupby("case_id"):
            axis.scatter(block[measured_column], block[model_column], s=6,
                         alpha=0.45, label=case_id, linewidths=0)
        values = np.concatenate([paired[measured_column].to_numpy(float),
                                 paired[model_column].to_numpy(float)])
        values = values[np.isfinite(values)]
        if values.size:
            span = [values.min(), values.max()]
            axis.plot(span, span, color="black", ls="--", lw=1.0, label="1:1")
            axis.set_xlim(span)
            axis.set_ylim(span)
        stats = rc.channel_statistics(paired[measured_column],
                                      paired[model_column])
        axis.set_xlabel(f"Measured {channel} (W m$^{{-2}}$)")
        axis.set_ylabel(f"Modelled {channel}, sensor-equivalent (W m$^{{-2}}$)")
        axis.set_title(f"{channel}: bias {stats.get('bias_Wm2', float('nan')):+.1f}, "
                       f"RMSE {stats.get('rmse_Wm2', float('nan')):.1f} W m$^{{-2}}$, "
                       f"r = {stats.get('pearson_r', float('nan')):.2f}",
                       fontsize=10)
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        stem = output / f"combined_{channel}_validation"
        for suffix in ("png", "pdf"):
            fig.savefig(f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input_root)
    cases = discover_cases(input_root, args.cases)
    constraint_routes = set(args.constraint_routes or [])
    validation_routes = set(args.validation_routes or [])
    overlap = constraint_routes & validation_routes
    if overlap:
        raise SystemExit(
            f"cases {sorted(overlap)} are named as BOTH constraint and "
            "validation; a sample cannot constrain a model and independently "
            "validate it at the same time")

    print("=" * 72)
    print("MEASUREMENT-CONSTRAINED RADIATIVE LAYER (four-component radiometer)")
    print("=" * 72)
    print(f"Cases: {cases}")
    print(f"Sensor height: {args.sensor_height_m:.2f} m "
          "(NOT recorded in the source data; configured validation height)")
    print(f"Interpolation: {args.interpolation}, gaps capped at "
          f"{args.max_measurement_gap_m:.0f} m / {args.max_measurement_gap_s:.0f} s")

    all_constraints, all_associated, all_validation = [], [], []
    digests: dict[str, str] = {}
    for case_id in cases:
        if case_id in constraint_routes:
            usage = "constraint"
        elif case_id in validation_routes:
            usage = "validation"
        elif constraint_routes or validation_routes:
            usage = "unconstrained"
        else:
            # Nothing downstream consumes the constraints yet, so with no split
            # the honest label is validation: the model never saw these data.
            usage = "validation"
        constraints, associated, extra = build_case(case_id, args, usage)
        if constraints is None:
            print(f"  {case_id}: no measurements; skipped")
            continue
        validation, case_digests = extra
        digests.update({f"{case_id}/{name}": digest
                        for name, digest in case_digests.items()})
        all_constraints.append(constraints)
        all_associated.append(associated)
        if validation is not None and len(validation):
            all_validation.append(validation)
        valid = (constraints["qc_flag"] == rc.QC_VALID).mean()
        reflectance = constraints["effective_lower_SW_reflectance"]
        usable_reflectance = reflectance.notna().mean()
        print(f"  {case_id}: {len(constraints):5d} samples, usage={usage:14s} "
              f"QC valid {valid:5.1%}, effective reflectance computable "
              f"{usable_reflectance:5.1%}")

        case_output = (Path(args.output_root) / case_id / "validation"
                       / "radiometer_constraints")
        case_output.mkdir(parents=True, exist_ok=True)
        if not args.no_surface_groups:
            thermal_dir = Path(args.output_root) / case_id / args.thermal_dir_name
            try:
                weights = associate_visible_surface_groups(
                    constraints, thermal_dir, args.sensor_height_m)
            except Exception as error:  # pragma: no cover - optional product
                print(f"    surface-group association skipped ({error})")
                weights = None
            if weights is not None:
                constraints["visible_surface_groups"] = [
                    json.dumps({name: round(share, 4)
                                for name, share in item.items()})
                    for item in weights]
                covered = sum(1 for item in weights if item)
                print(f"    visible surface groups recorded for "
                      f"{covered}/{len(weights)} measurement points")
        constraints.to_csv(case_output / CONSTRAINTS_CSV, index=False)
        associated.to_csv(case_output / PROFILES_CSV, index=False)
        if not args.no_figures:
            plot_route_profiles(constraints, case_output / "figures", args.dpi)

    if not all_constraints:
        print("No measurements found.")
        return 2

    aggregate = Path(args.aggregate_output_dir)
    aggregate.mkdir(parents=True, exist_ok=True)
    constraints = pd.concat(all_constraints, ignore_index=True)
    associated = pd.concat(all_associated, ignore_index=True)
    constraints.to_csv(aggregate / CONSTRAINTS_CSV, index=False)
    associated.to_csv(aggregate / PROFILES_CSV, index=False)

    validation = (pd.concat(all_validation, ignore_index=True)
                  if all_validation else pd.DataFrame())
    if len(validation):
        validation.to_csv(aggregate / VALIDATION_CSV, index=False)

    # Combined scatter panels need the paired model/measurement rows.
    paired_blocks = []
    for case_id in cases:
        frame = load_model_channels(Path(args.output_root) / case_id)
        if frame is not None:
            frame = frame.copy()
            frame["case_id"] = case_id
            paired_blocks.append(frame)
    if paired_blocks and not args.no_figures:
        paired = pd.concat(paired_blocks, ignore_index=True)
        forbidden = [column for column in FORBIDDEN_BODY_COLUMNS
                     if column in MODEL_CHANNEL_COLUMNS.values()]
        if forbidden:
            raise RuntimeError(
                "a body-absorbed column reached the sensor comparison: "
                f"{forbidden}")
        plot_combined_validation(paired, aggregate / "figures", args.dpi)

    metadata = {
        "layer": "measurement_constrained_radiative_layer",
        "source": rc.CONSTRAINT_SOURCE,
        "priority": ["four_component_radiometer", "manual_override",
                     "osm_direct", "imagery", "osm_inferred", "colour_hint",
                     "default"],
        "sensor": {
            "model": ("four-component net radiometer (up/down pyranometer and "
                      "pyrgeometer pair) on the Lisbon mobile cart, logged by a "
                      "Campbell Scientific CR350 alongside a Gill MaxiMet "
                      "GMX500 and a Campbell BLACKGLOBE-L"),
            "height_m": args.sensor_height_m,
            "height_provenance": ("ASSUMED: the source files carry no sensor "
                                  "height column; this is the configured "
                                  "validation sensor height"),
            "orientation": ("upward-facing pair -> SWin/LWin (upper "
                            "hemisphere); downward-facing pair -> SWout/LWout "
                            "(lower hemisphere)"),
            "orientation_provenance": ("inferred from the channel naming; not "
                                       "recorded explicitly in the data"),
            "sampling_interval_s": "5 or 10 depending on route",
            "units": "W m-2 for all four channels",
        },
        "source_columns": rc.MEASURED_COLUMN_MAP,
        "coordinate_columns": {"x": rc.LOCAL_X_COLUMN, "y": rc.LOCAL_Y_COLUMN,
                               "note": ("x_local_m/y_local_m are already in the "
                                        "STL/stage-05 metric frame, so no "
                                        "transformation is applied")},
        "timestamp_column": rc.TIMESTAMP_COLUMN,
        "quality_control": rc.QualityControlConfig(
            minimum_sw_down_for_reflectance_Wm2=args.min_sw_down_for_albedo
        ).as_metadata(),
        "interpolation": rc.InterpolationConfig(
            method=args.interpolation,
            max_gap_s=args.max_measurement_gap_s,
            max_gap_m=args.max_measurement_gap_m).as_metadata(),
        "derived_quantities": {
            "effective_lower_SW_reflectance": (
                "SW_up/SW_down where SW_down >= the configured minimum. An "
                "EFFECTIVE LOCAL reflectance of the sensor's mixed footprint, "
                "NOT an intrinsic material albedo."),
            "effective_lower_LW_radiosity_Wm2": (
                "LW_up as measured. NOT an emissivity: recovering one needs an "
                "independent surface temperature."),
            "effective_upper_LW_irradiance_Wm2": "LW_down as measured.",
        },
        "usage": {
            "constraint_routes": sorted(constraint_routes),
            "validation_routes": sorted(validation_routes),
            "default_usage_when_unsplit": "validation",
            "note": ("Nothing downstream currently consumes these constraints, "
                     "so with no explicit split the comparison IS independent "
                     "validation. Naming a case under --constraint-routes "
                     "relabels its statistics as measurement-constrained."),
        },
        "model_side_columns": MODEL_CHANNEL_COLUMNS,
        "model_side_note": (
            "Stage-05 sensor-equivalent channels: planar hemispherical, on the "
            "instrument's own weighting. No body projected-area factor, no "
            "human shortwave absorptivity, no human emissivity. Body-absorbed "
            "flux is never compared against these."),
        "cases": cases,
        "n_samples": int(len(constraints)),
        "source_file_sha256": digests,
    }
    (aggregate / METADATA_JSON).write_text(json.dumps(metadata, indent=2) + "\n",
                                           encoding="utf-8")

    print("\n" + "=" * 72)
    print("CONSTRAINT LAYER SUMMARY")
    print("=" * 72)
    print(f"Samples              : {len(constraints):,}")
    counts = constraints["qc_flag"].value_counts()
    for flag in (rc.QC_VALID, rc.QC_QUESTIONABLE, rc.QC_INVALID):
        share = counts.get(flag, 0) / len(constraints)
        print(f"  QC {flag:13s}: {counts.get(flag, 0):5d}  ({share:5.1%})")
    day = constraints[constraints["period"].eq("day")]
    reflectance = day["effective_lower_SW_reflectance"].dropna()
    if len(reflectance):
        print(f"\nEffective lower-hemisphere SW reflectance (day samples):")
        print(f"  computable for {len(reflectance):,}/{len(day):,} samples "
              f"({len(reflectance) / max(len(day), 1):.1%})")
        print(f"  median {reflectance.median():.3f}, "
              f"5-95% {reflectance.quantile(0.05):.3f}-{reflectance.quantile(0.95):.3f}")
        print("  (effective LOCAL reflectance of a mixed footprint, "
              "NOT an intrinsic material albedo)")
    print("\nAlong-route heterogeneity retained (day routes, per case):")
    for case_id, block in day.groupby("case_id"):
        for channel in ("SW_down", "SW_up", "LW_up"):
            values = block[f"{channel}_measured_Wm2"]
            if channel == "SW_down":
                print(f"  {case_id}: SW_down {values.min():6.0f}-{values.max():6.0f} "
                      f"(sd {values.std():5.0f})", end="")
            else:
                print(f" | {channel} {values.min():5.0f}-{values.max():5.0f}", end="")
        print()
    if len(validation):
        print("\nApples-to-apples sensor validation (all scope, per channel):")
        overall = validation[validation["scope"].eq("all")]
        for channel in MODEL_CHANNEL_COLUMNS:
            block = overall[overall["channel"].eq(channel)]
            if block.empty:
                continue
            print(f"  {channel:8s} bias {block['bias_Wm2'].mean():+7.1f}  "
                  f"MAE {block['mae_Wm2'].mean():6.1f}  "
                  f"RMSE {block['rmse_Wm2'].mean():6.1f} W/m2  "
                  f"r {block['pearson_r'].mean():+.2f}")
        label = ("MEASUREMENT-CONSTRAINED (not independent)"
                 if constraint_routes else "INDEPENDENT VALIDATION")
        print(f"\n  Status: {label}")
    print(f"\nWritten to {aggregate}")
    print(f"[radiometer_constraints] samples={len(constraints)} "
          f"cases={len(cases)} output_dir={aggregate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
