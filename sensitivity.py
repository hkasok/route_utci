"""Air-temperature sensitivity of route-based JOS-3 core-temperature rise.

This is intentionally separate from ``start.sh`` and the production pipeline.
It reads the existing stage-09 result as its baseline, repeats the same JOS-3
route calculation with air temperature raised by exactly 1 degC, and writes a
grouped-bar comparison as a PDF.

Default usage (matches the defaults currently used by start.sh):

    python3 sensitivity.py

If stage 09 was run with a subject preset, pass the same preset here:

    python3 sensitivity.py --subject-profile elderly_female
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pythermalcomfort.models import JOS3
from scipy.spatial import cKDTree

from route_selection import load_selected_routes
from subject_profiles import PROFILES, get_profile
from weather_provider import WeatherProvider


AIR_TEMPERATURE_PERTURBATION_C = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare standard JOS-3 core rise with air temperature +1 C."
    )
    parser.add_argument(
        "--baseline-csv",
        default="run_output/viz/route_jos3/route_ranking_summary.csv",
        help="Existing standard stage-09 route_ranking_summary.csv.",
    )
    parser.add_argument(
        "--routes-pkl",
        default="run_output/osm_paths/selected_routes.pkl",
        help="The selected routes used by stage 09.",
    )
    parser.add_argument(
        "--mrt-results-dir",
        default="run_output/mrt_facet_out",
        help="MRT directory used by stage 09.",
    )
    parser.add_argument(
        "--weather-csv",
        default="weather.csv",
        help="Weather forcing used by stage 09 (all values except Ta stay unchanged).",
    )
    parser.add_argument("--output-dir", default="sensitivity output")
    parser.add_argument("--departure-hour", type=float, default=13.0)
    parser.add_argument("--walking-speed-ms", type=float, default=1.3)
    parser.add_argument("--equilibration-min", type=float, default=10.0)
    parser.add_argument("--activity-par", type=float, default=2.5)
    parser.add_argument(
        "--subject-profile", choices=sorted(PROFILES), default=None
    )
    parser.add_argument("--person-height-m", type=float, default=None)
    parser.add_argument("--person-weight-kg", type=float, default=None)
    parser.add_argument("--person-age", type=int, default=None)
    parser.add_argument("--person-sex", choices=("male", "female"), default=None)
    return parser.parse_args()


def resolve_subject(args: argparse.Namespace) -> dict:
    profile = get_profile(args.subject_profile) if args.subject_profile else None
    return {
        "height": (
            args.person_height_m
            if args.person_height_m is not None
            else (profile.height if profile else 1.72)
        ),
        "weight": (
            args.person_weight_kg
            if args.person_weight_kg is not None
            else (profile.weight if profile else 74.0)
        ),
        "age": (
            args.person_age
            if args.person_age is not None
            else (profile.age if profile else 30)
        ),
        "sex": (
            args.person_sex
            if args.person_sex is not None
            else (profile.sex if profile else "male")
        ),
        "fat": profile.fat if profile else 15.0,
        "ci": profile.ci if profile else 2.59,
        "setpoint_shift_c": profile.setpoint_shift_c if profile else 0.0,
    }


def load_mrt(mrt_dir: Path):
    xyz = np.load(mrt_dir / "path_xyz.npy")
    tmrt = np.load(mrt_dir / "tmrt_matrix_C.npy")
    times = pd.read_csv(mrt_dir / "times.csv", parse_dates=["time"])["time"]
    hours = np.array(
        [
            value.hour + value.minute / 60.0 + value.second / 3600.0
            for value in times
        ]
    )
    if tmrt.shape != (len(hours), len(xyz)):
        raise ValueError(
            "MRT dimensions are inconsistent: expected "
            f"({len(hours)}, {len(xyz)}), got {tmrt.shape}."
        )
    return cKDTree(xyz[:, :2]), tmrt, hours


def simulate_plus_one(
    routes: list[dict],
    mrt_tree: cKDTree,
    tmrt_matrix: np.ndarray,
    time_hours: np.ndarray,
    weather: WeatherProvider,
    subject: dict,
    args: argparse.Namespace,
) -> dict[int, float]:
    """Repeat stage 09, changing only JOS-3 dry-bulb air temperature."""
    rises = {}
    for route_number, route in enumerate(routes, start=1):
        xy = np.asarray(route["xy"], dtype=float)
        segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative_distance = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        arrival_hour = (
            args.departure_hour
            + cumulative_distance / args.walking_speed_ms / 3600.0
        )
        _, nearest_idx = mrt_tree.query(xy)

        model = JOS3(
            height=subject["height"],
            weight=subject["weight"],
            age=subject["age"],
            sex=subject["sex"],
            fat=subject["fat"],
            ci=subject["ci"],
        )
        if subject["setpoint_shift_c"]:
            model.cr_set_point = (
                model.cr_set_point + subject["setpoint_shift_c"]
            )
        model.par = args.activity_par
        bsa_weights = model.bsa / model.bsa.sum()

        def weighted_core_c() -> float:
            return float(np.sum(model.t_core * bsa_weights))

        # The +1 C perturbation also applies during the standard 10-minute
        # equilibration; MRT, RH, wind, activity, route, and timing are unchanged.
        start_hour = args.departure_hour % 24.0
        start_tmrt = np.interp(
            start_hour,
            time_hours,
            tmrt_matrix[:, nearest_idx[0]],
            period=24.0,
        )
        model.tdb = float(weather.air_temp_c(start_hour)) + AIR_TEMPERATURE_PERTURBATION_C
        model.tr = float(start_tmrt)
        model.rh = float(weather.rh_pct(start_hour))
        model.v = float(weather.wind_ms(start_hour))
        if args.equilibration_min > 0:
            model.simulate(
                times=int(args.equilibration_min), dtime=60, output=False
            )
        start_core_c = weighted_core_c()

        final_core_c = start_core_c
        for point_number, hour_unwrapped in enumerate(arrival_hour):
            hour = hour_unwrapped % 24.0
            model.tdb = (
                float(weather.air_temp_c(hour))
                + AIR_TEMPERATURE_PERTURBATION_C
            )
            model.tr = float(
                np.interp(
                    hour,
                    time_hours,
                    tmrt_matrix[:, nearest_idx[point_number]],
                    period=24.0,
                )
            )
            model.rh = float(weather.rh_pct(hour))
            model.v = float(weather.wind_ms(hour))
            if point_number > 0:
                step_seconds = (
                    arrival_hour[point_number] - arrival_hour[point_number - 1]
                ) * 3600.0
                model.simulate(times=1, dtime=step_seconds, output=False)
            final_core_c = weighted_core_c()

        rises[route_number] = final_core_c - start_core_c
        print(
            f"Route {route_number}: +1 C air-temperature core rise "
            f"= {rises[route_number]:+.6f} C"
        )
    return rises


def save_bar_chart(
    baseline: pd.DataFrame, perturbed: dict[int, float], output_pdf: Path
) -> None:
    baseline = baseline.sort_values("route_id")
    route_ids = baseline["route_id"].astype(int).to_numpy()
    missing = sorted(set(route_ids) - set(perturbed))
    extra = sorted(set(perturbed) - set(route_ids))
    if missing or extra:
        raise ValueError(
            f"Baseline and selected-route IDs differ (missing={missing}, extra={extra})."
        )

    standard = baseline["final_tcore_rise_c"].to_numpy(dtype=float)
    plus_one = np.array([perturbed[route_id] for route_id in route_ids])
    change = plus_one - standard
    x = np.arange(len(route_ids))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6.5))
    standard_bars = ax.bar(
        x - width / 2,
        standard,
        width,
        label="Standard air temperature",
        color="#4C78A8",
    )
    perturbed_bars = ax.bar(
        x + width / 2,
        plus_one,
        width,
        label="Air temperature +1 °C",
        color="#E45756",
    )
    ax.bar_label(standard_bars, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(perturbed_bars, fmt="%.3f", padding=3, fontsize=9)
    for position, delta in zip(x, change):
        top = max(standard[position], plus_one[position])
        ax.annotate(
            f"Δ {delta:+.3f} °C",
            (position, top),
            xytext=(0, 21),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#7A1F1F",
        )

    ax.set_xticks(x, [f"Route {route_id}" for route_id in route_ids])
    ax.set_ylabel("Final JOS-3 core-temperature rise [°C]")
    ax.set_title(
        "Sensitivity of Route Core-Temperature Rise to +1 °C Air Temperature"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(standard.max(), plus_one.max()) * 1.25)
    fig.text(
        0.5,
        0.015,
        "Only JOS-3 dry-bulb air temperature is perturbed; "
        "MRT, RH, wind, activity, timing, routes, and subject are unchanged.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("\nComparison:")
    for route_id, base, warmer, delta in zip(
        route_ids, standard, plus_one, change
    ):
        print(
            f"  Route {route_id}: standard={base:+.6f} C, "
            f"+1 C air={warmer:+.6f} C, difference={delta:+.6f} C"
        )
    print(f"\nSaved PDF: {output_pdf}")


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline_csv)
    required = {
        "route_id",
        "final_tcore_rise_c",
    }
    baseline = pd.read_csv(baseline_path)
    absent = required - set(baseline.columns)
    if absent:
        raise ValueError(
            f"{baseline_path} is missing columns: {', '.join(sorted(absent))}"
        )

    selection = load_selected_routes(args.routes_pkl)
    routes = selection["routes"]
    if len(routes) != len(baseline):
        raise ValueError(
            f"Selected routes ({len(routes)}) and baseline rows "
            f"({len(baseline)}) do not match."
        )

    weather = WeatherProvider(csv_path=args.weather_csv, strict=True)
    subject = resolve_subject(args)
    mrt_tree, tmrt_matrix, time_hours = load_mrt(Path(args.mrt_results_dir))
    perturbed = simulate_plus_one(
        routes,
        mrt_tree,
        tmrt_matrix,
        time_hours,
        weather,
        subject,
        args,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_bar_chart(
        baseline,
        perturbed,
        output_dir / "core_temperature_air_temp_plus_1C_comparison.pdf",
    )


if __name__ == "__main__":
    main()
