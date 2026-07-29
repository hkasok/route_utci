"""Factorial uniform-reference Ta/moisture analysis for route-based JOS-3.

The existing stage-09 CSV is a read-only, fully resolved benchmark.  Each
factorial case replaces only dry-bulb air temperature and atmospheric moisture
with plausible spatially uniform references. Route/arrival-time-resolved MRT
and wind, route timing, activity, subject, and JOS-3 configuration are retained.
Tested references span known local ranges and are not assumed spatial means.
"""
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from pythermalcomfort.models import JOS3
from scipy.spatial import cKDTree
from scipy.stats import kendalltau, spearmanr

from route_selection import load_selected_routes
from subject_profiles import PROFILES, get_profile
from weather_provider import WeatherProvider

TIE_TOL_C = 1e-8
RELATIVE_ERROR_TOL_C = 1e-8
PNG_DPI = 300


@dataclass(frozen=True)
class PreparedRoute:
    """Geometry/timing quantities reused by every factorial case."""
    route_id: int
    xy: np.ndarray
    length_m: float
    duration_min: float
    arrival_hour: np.ndarray
    step_duration_s: np.ndarray
    nearest_mrt_index: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse project paths, factorial ranges, and production JOS-3 settings."""
    p = argparse.ArgumentParser(description="Uniform Ta-e simplification analysis")
    p.add_argument("--baseline-csv", default="run_output/viz/route_jos3/route_ranking_summary.csv")
    p.add_argument("--routes-pkl", default="run_output/osm_paths/selected_routes.pkl")
    p.add_argument("--mrt-results-dir", default="run_output/mrt_facet_out")
    p.add_argument("--weather-csv", default="weather.csv",
                   help="Production weather source; only resolved wind is retained")
    p.add_argument("--output-dir", default="sensitivity output")
    p.add_argument("--ta-min", type=float, default=31.0)
    p.add_argument("--ta-max", type=float, default=35.0)
    p.add_argument("--ta-levels", type=int, default=5)
    p.add_argument("--e-min", type=float, default=28.0)
    p.add_argument("--e-max", type=float, default=35.0)
    p.add_argument("--e-levels", type=int, default=5)
    p.add_argument("--departure-hour", type=float, default=13.0)
    p.add_argument("--walking-speed-ms", type=float, default=1.3)
    p.add_argument("--equilibration-min", type=float, default=10.0)
    p.add_argument("--activity-par", type=float, default=2.5)
    p.add_argument("--subject-profile", choices=sorted(PROFILES), default=None)
    p.add_argument("--person-height-m", type=float, default=None)
    p.add_argument("--person-weight-kg", type=float, default=None)
    p.add_argument("--person-age", type=int, default=None)
    p.add_argument("--person-sex", choices=("male", "female"), default=None)
    return p.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def validate_settings(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Validate settings and return ordered factorial levels."""
    if not np.isfinite(args.walking_speed_ms) or args.walking_speed_ms <= 0:
        raise ValueError("--walking-speed-ms must be finite and positive")
    if not np.isfinite(args.departure_hour):
        raise ValueError("--departure-hour must be finite")
    if not np.isfinite(args.equilibration_min) or args.equilibration_min < 0:
        raise ValueError("--equilibration-min must be finite and non-negative")
    if not np.isfinite(args.activity_par) or args.activity_par <= 0:
        raise ValueError("--activity-par must be finite and positive")
    for name, lo, hi, levels in (("Ta", args.ta_min, args.ta_max, args.ta_levels),
                                  ("vapor pressure", args.e_min, args.e_max, args.e_levels)):
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f"Invalid {name} range: minimum must be finite and below maximum")
        if levels < 2:
            raise ValueError(f"{name} requires at least two levels for sensitivity slopes")
    if args.e_min < 0:
        raise ValueError("Vapor pressure cannot be negative")
    return np.linspace(args.ta_min, args.ta_max, args.ta_levels), np.linspace(args.e_min, args.e_max, args.e_levels)


def resolve_subject(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve subject with the same flag > preset > default precedence as stage 09."""
    prof = get_profile(args.subject_profile) if args.subject_profile else None
    subject = {
        "height": args.person_height_m if args.person_height_m is not None else (prof.height if prof else 1.72),
        "weight": args.person_weight_kg if args.person_weight_kg is not None else (prof.weight if prof else 74.0),
        "age": args.person_age if args.person_age is not None else (prof.age if prof else 30),
        "sex": args.person_sex if args.person_sex is not None else (prof.sex if prof else "male"),
        "fat": prof.fat if prof else 15.0, "ci": prof.ci if prof else 2.59,
        "setpoint_shift_c": prof.setpoint_shift_c if prof else 0.0,
    }
    if not all(np.isfinite(float(subject[k])) for k in ("height", "weight", "age", "fat", "ci", "setpoint_shift_c")):
        raise ValueError("Resolved subject contains non-finite values")
    if subject["height"] <= 0 or subject["weight"] <= 0 or subject["age"] <= 0:
        raise ValueError("Subject height, weight, and age must be positive")
    return subject


def load_mrt(directory: Path) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    """Load and validate MRT coordinates, matrix, and decimal-hour axis."""
    xyz_path, matrix_path, times_path = directory/"path_xyz.npy", directory/"tmrt_matrix_C.npy", directory/"times.csv"
    for path, label in ((xyz_path, "MRT coordinates"), (matrix_path, "MRT matrix"), (times_path, "MRT times")):
        require_file(path, label)
    xyz, matrix = np.load(xyz_path), np.load(matrix_path)
    if xyz.ndim != 2 or xyz.shape[1] < 2 or len(xyz) == 0:
        raise ValueError(f"Malformed MRT coordinates: shape {xyz.shape}")
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or not np.isfinite(xyz[:, :2]).all():
        raise ValueError("MRT coordinates/matrix must be finite and matrix must be 2-D")
    times = pd.read_csv(times_path)
    if "time" not in times:
        raise ValueError(f"{times_path} lacks required 'time' column")
    t = pd.to_datetime(times["time"], errors="raise")
    hours = (t.dt.hour + t.dt.minute/60 + t.dt.second/3600 + t.dt.microsecond/3.6e9).to_numpy(float)
    if matrix.shape != (len(hours), len(xyz)):
        raise ValueError(f"Inconsistent MRT dimensions: matrix {matrix.shape}, expected {(len(hours), len(xyz))}")
    if len(hours) < 2 or not np.isfinite(hours).all():
        raise ValueError("MRT times require at least two finite entries")
    return cKDTree(xyz[:, :2]), matrix, hours


def prepare_routes(raw_routes: list[dict[str, Any]], tree: cKDTree, args: argparse.Namespace) -> list[PreparedRoute]:
    """Validate and precompute route geometry, timing, and nearest MRT points."""
    result, ids = [], []
    for position, raw in enumerate(raw_routes, 1):
        route_id = int(raw.get("route_id", position)); ids.append(route_id)
        xy = np.asarray(raw.get("xy"), float)
        if xy.ndim != 2 or xy.shape[1] < 2 or len(xy) < 2 or not np.isfinite(xy).all():
            raise ValueError(f"Route {route_id} is malformed or non-finite")
        xy = xy[:, :2]; segments = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if not np.isfinite(segments).all() or np.any(segments <= 0):
            raise ValueError(f"Route {route_id} has duplicate/non-positive segments")
        cumulative = np.r_[0.0, np.cumsum(segments)]; geometric_length = float(cumulative[-1])
        length = float(raw.get("length_m", geometric_length))
        if not np.isfinite(length) or length <= 0 or geometric_length <= 0:
            raise ValueError(f"Route {route_id} has invalid length")
        arrival = args.departure_hour + cumulative/args.walking_speed_ms/3600
        durations = np.diff(arrival)*3600
        if not np.isfinite(durations).all() or np.any(durations <= 0):
            raise ValueError(f"Route {route_id} has non-positive point-to-point durations")
        nearest = np.asarray(tree.query(xy)[1], int)
        result.append(PreparedRoute(route_id, xy, length, geometric_length/args.walking_speed_ms/60,
                                    arrival, durations, nearest))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Selected routes contain duplicate IDs: {ids}")
    return sorted(result, key=lambda r: r.route_id)


def validate_baseline(path: Path, route_ids: list[int]) -> pd.DataFrame:
    """Validate unique, finite, exact route coverage in the read-only benchmark."""
    require_file(path, "fully resolved benchmark CSV")
    df = pd.read_csv(path)
    missing = {"route_id", "final_tcore_rise_c"} - set(df)
    if missing: raise ValueError(f"Benchmark lacks columns: {sorted(missing)}")
    ids = pd.to_numeric(df["route_id"], errors="raise")
    if ids.isna().any() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError("Benchmark route IDs must be non-missing integers")
    df["route_id"] = ids.astype(int)
    if df["route_id"].duplicated().any():
        raise ValueError(f"Benchmark has duplicate route IDs: {df.loc[df.route_id.duplicated(False), 'route_id'].tolist()}")
    df["final_tcore_rise_c"] = pd.to_numeric(df["final_tcore_rise_c"], errors="raise")
    if not np.isfinite(df["final_tcore_rise_c"]).all():
        raise ValueError("Benchmark core-rise values must be finite")
    selected, benchmark = set(route_ids), set(df.route_id)
    if selected != benchmark:
        raise ValueError(f"Route IDs do not match exactly; missing={sorted(selected-benchmark)}, unexpected={sorted(benchmark-selected)}")
    return df.sort_values("route_id").reset_index(drop=True)


def vapor_pressure_to_rh_pct(ta_c: float, vapor_pressure_hpa: float) -> float:
    """Convert vapor pressure to RH, rejecting rather than clipping invalid RH."""
    ta_c, vapor_pressure_hpa = float(ta_c), float(vapor_pressure_hpa)
    if not np.isfinite(ta_c) or not np.isfinite(vapor_pressure_hpa):
        raise ValueError("Ta and vapor pressure must be finite")
    es = 6.112 * math.exp(17.67*ta_c/(ta_c+243.5))
    rh = 100.0*vapor_pressure_hpa/es
    if not np.isfinite(es) or not np.isfinite(rh):
        raise ValueError(f"Non-finite RH for Ta={ta_c}, e={vapor_pressure_hpa}")
    if rh < 0 or rh > 100:
        raise ValueError(f"Invalid RH={rh:.3f}% for Ta={ta_c:.3f} C, e={vapor_pressure_hpa:.3f} hPa")
    return rh


def initialize_jos3_model(subject: dict[str, Any], activity: float) -> tuple[JOS3, np.ndarray]:
    """Create fresh production-equivalent JOS-3 state and normalized BSA weights."""
    model = JOS3(height=subject["height"], weight=subject["weight"], age=subject["age"],
                 sex=subject["sex"], fat=subject["fat"], ci=subject["ci"])
    if subject["setpoint_shift_c"]:
        model.cr_set_point = model.cr_set_point + subject["setpoint_shift_c"]
    model.par = activity
    weights = np.asarray(model.bsa, float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("JOS-3 returned invalid BSA weights")
    return model, weights/weights.sum()


def simulate_uniform_reference_case(routes: list[PreparedRoute], mrt_matrix: np.ndarray,
                                    time_hours: np.ndarray, weather: WeatherProvider,
                                    subject: dict[str, Any], args: argparse.Namespace,
                                    ta_ref_c: float, e_ref_hpa: float) -> list[dict[str, Any]]:
    """Run one fresh JOS-3 model per route with constant Ta/RH and resolved MRT/wind."""
    rh = vapor_pressure_to_rh_pct(ta_ref_c, e_ref_hpa); rows = []
    for route in routes:
        model, weights = initialize_jos3_model(subject, args.activity_par)
        def core() -> float:
            value = float(np.sum(np.asarray(model.t_core)*weights))
            if not np.isfinite(value):
                raise ValueError(f"Non-finite core temperature: route {route.route_id}, Ta={ta_ref_c}, e={e_ref_hpa}")
            return value
        h0 = args.departure_hour % 24
        tr0 = float(np.interp(h0, time_hours, mrt_matrix[:, route.nearest_mrt_index[0]], period=24.0))
        v0 = float(weather.wind_ms(h0))
        if not np.isfinite(tr0) or not np.isfinite(v0): raise ValueError(f"Non-finite start MRT/wind, route {route.route_id}")
        model.tdb, model.tr, model.rh, model.v = ta_ref_c, tr0, rh, v0
        if args.equilibration_min > 0:
            model.simulate(times=int(args.equilibration_min), dtime=60, output=False)
        start_core = core(); final_core = start_core
        for j, unwrapped_hour in enumerate(route.arrival_hour):
            hour = unwrapped_hour % 24
            tr = float(np.interp(hour, time_hours, mrt_matrix[:, route.nearest_mrt_index[j]], period=24.0))
            wind = float(weather.wind_ms(hour))
            if not np.isfinite(tr) or not np.isfinite(wind):
                raise ValueError(f"Non-finite MRT/wind: route {route.route_id}, point {j}")
            model.tdb, model.tr, model.rh, model.v = ta_ref_c, tr, rh, wind
            if j:
                dt = float(route.step_duration_s[j-1])
                if not np.isfinite(dt) or dt <= 0: raise ValueError(f"Invalid duration {dt} on route {route.route_id}")
                model.simulate(times=1, dtime=dt, output=False)
            final_core = core()
        rise = final_core-start_core
        if not np.isfinite(rise): raise ValueError(f"Non-finite core rise on route {route.route_id}")
        rows.append(dict(route_id=route.route_id, route_length_m=route.length_m,
                         route_duration_min=route.duration_min,
                         reference_final_tcore_rise_c=rise))
    return rows


def relation(difference: float) -> str:
    """Tolerance-aware relation for core(A)-core(B)."""
    return "tie" if abs(difference) <= TIE_TOL_C else ("route_a_higher" if difference > 0 else "route_a_lower")


def ranks(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=True)


def calculate_ranking_metrics(case_id: str, ta: float, e: float, rh: float,
                              benchmark: pd.DataFrame, reference: pd.DataFrame,
                              departure: float) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate route-selection fidelity and annotate route-case rows."""
    merged = benchmark[["route_id", "final_tcore_rise_c"]].merge(reference, on="route_id", validate="one_to_one").sort_values("route_id")
    ids = merged.route_id.to_numpy(int); b = merged.final_tcore_rise_c.to_numpy(float); r = merged.reference_final_tcore_rise_c.to_numpy(float)
    bmap, rmap = dict(zip(ids, b)), dict(zip(ids, r))
    pair_relations = [(relation(bmap[a]-bmap[c]), relation(rmap[a]-rmap[c])) for a,c in itertools.combinations(ids, 2)]
    complete = all(x == y for x,y in pair_relations)
    bbest = int(merged.loc[merged.final_tcore_rise_c.idxmin(), "route_id"])
    rbest = int(merged.loc[merged.reference_final_tcore_rise_c.idxmin(), "route_id"])
    if len(ids) >= 2 and np.ptp(b) > TIE_TOL_C and np.ptp(r) > TIE_TOL_C:
        sr = spearmanr(b, r); rho, sp = float(sr.statistic), float(sr.pvalue)
    else: rho = sp = np.nan
    if len(ids) >= 2:
        kr = kendalltau(b, r); tau, kp = float(kr.statistic), float(kr.pvalue)
    else: tau = kp = np.nan
    error = r-b
    relative = np.where(np.abs(b) >= RELATIVE_ERROR_TOL_C, 100*error/b, np.nan)
    rows = pd.DataFrame(dict(case_id=case_id, route_id=ids, ta_ref_c=ta,
        vapor_pressure_ref_hpa=e, rh_ref_pct=rh, departure_hour=departure,
        route_length_m=merged.route_length_m, route_duration_min=merged.route_duration_min,
        benchmark_final_tcore_rise_c=b, reference_final_tcore_rise_c=r,
        signed_error_c=error, absolute_error_c=np.abs(error), relative_error_pct=relative,
        benchmark_rank=ranks(merged.final_tcore_rise_c).to_numpy(),
        reference_rank=ranks(merged.reference_final_tcore_rise_c).to_numpy(),
        benchmark_best_route=bbest, reference_best_route=rbest, best_route_preserved=bbest==rbest))
    summary = dict(case_id=case_id, ta_ref_c=ta, vapor_pressure_ref_hpa=e, rh_ref_pct=rh,
        benchmark_best_route=bbest, reference_best_route=rbest, best_route_preserved=bbest==rbest,
        complete_ranking_preserved=complete, spearman_rho=rho, spearman_p_value=sp,
        kendall_tau=tau, kendall_p_value=kp, maximum_route_absolute_error_c=float(np.abs(error).max()),
        mean_route_absolute_error_c=float(np.abs(error).mean()))
    return summary, rows


def calculate_pairwise_metrics(summary: dict[str, Any], benchmark: pd.DataFrame,
                               reference: pd.DataFrame) -> list[dict[str, Any]]:
    """Calculate contrast error, retention, and pair ordering for one case."""
    b = benchmark.set_index("route_id").final_tcore_rise_c.to_dict()
    r = reference.set_index("route_id").reference_final_tcore_rise_c.to_dict(); rows=[]
    for a,c in itertools.combinations(sorted(b), 2):
        db, dr = float(b[a]-b[c]), float(r[a]-r[c]); rb, rr = relation(db), relation(dr)
        rows.append(dict(case_id=summary["case_id"], ta_ref_c=summary["ta_ref_c"],
            vapor_pressure_ref_hpa=summary["vapor_pressure_ref_hpa"], rh_ref_pct=summary["rh_ref_pct"],
            route_a=a, route_b=c, benchmark_difference_c=db, reference_difference_c=dr,
            route_contrast_error_c=dr-db,
            contrast_retention_ratio=dr/db if abs(db)>TIE_TOL_C else np.nan,
            benchmark_relation=rb, reference_relation=rr, ranking_preserved=rb==rr))
    return rows


def run_factorial_analysis(routes, matrix, hours, weather, subject, args, ta_values, e_values, benchmark):
    """Run every case/route with fresh state and return detailed metrics."""
    all_rows=[]; cases=[]; pairs=[]; combinations=list(itertools.product(ta_values, e_values))
    for number,(ta,e) in enumerate(combinations, 1):
        case_id=f"CASE_{number:03d}"; rh=vapor_pressure_to_rh_pct(ta,e)
        print(f"Case {number}/{len(combinations)}: Ta={ta:.2f} C, e={e:.2f} hPa, RH={rh:.2f}%")
        reference=pd.DataFrame(simulate_uniform_reference_case(routes,matrix,hours,weather,subject,args,float(ta),float(e)))
        for row in reference.itertuples(): print(f"  Route {row.route_id}: final core rise {row.reference_final_tcore_rise_c:+.6f} C")
        summary, annotated=calculate_ranking_metrics(case_id,float(ta),float(e),rh,benchmark,reference,args.departure_hour)
        all_rows.append(annotated); cases.append(summary); pairs.extend(calculate_pairwise_metrics(summary,benchmark,reference))
    return pd.concat(all_rows,ignore_index=True),pd.DataFrame(cases),pd.DataFrame(pairs)


def calculate_route_errors(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize absolute prediction performance route by route."""
    rows=[]
    for route_id,g in results.groupby("route_id",sort=True):
        best=g.loc[g.absolute_error_c.idxmin()]; worst=g.loc[g.absolute_error_c.idxmax()]
        pred=g.reference_final_tcore_rise_c.to_numpy(); err=g.signed_error_c.to_numpy()
        rows.append(dict(route_id=int(route_id),benchmark_final_tcore_rise_c=float(g.benchmark_final_tcore_rise_c.iloc[0]),
            minimum_reference_prediction_c=float(pred.min()),maximum_reference_prediction_c=float(pred.max()),
            prediction_range_c=float(np.ptp(pred)),mean_signed_error_c=float(err.mean()),
            mean_absolute_error_c=float(np.abs(err).mean()),rmse_c=float(np.sqrt(np.mean(err**2))),
            maximum_absolute_error_c=float(np.abs(err).max()),best_ta_ref_c=float(best.ta_ref_c),
            best_e_ref_hpa=float(best.vapor_pressure_ref_hpa),best_rh_ref_pct=float(best.rh_ref_pct),
            worst_ta_ref_c=float(worst.ta_ref_c),worst_e_ref_hpa=float(worst.vapor_pressure_ref_hpa),
            worst_rh_ref_pct=float(worst.rh_ref_pct)))
    return pd.DataFrame(rows)


def calculate_sensitivity_slopes(results: pd.DataFrame) -> pd.DataFrame:
    """Adjacent finite differences over the tested ranges (not universal sensitivities)."""
    rows=[]
    for route_id,g in results.groupby("route_id",sort=True):
        st=[]; se=[]
        for _,q in g.groupby("vapor_pressure_ref_hpa"):
            q=q.sort_values("ta_ref_c"); st.extend(np.diff(q.reference_final_tcore_rise_c)/np.diff(q.ta_ref_c))
        for _,q in g.groupby("ta_ref_c"):
            q=q.sort_values("vapor_pressure_ref_hpa"); se.extend(np.diff(q.reference_final_tcore_rise_c)/np.diff(q.vapor_pressure_ref_hpa))
        pivot=g.pivot(index="vapor_pressure_ref_hpa",columns="ta_ref_c",values="reference_final_tcore_rise_c").sort_index().sort_index(axis=1)
        z=pivot.to_numpy(); ts=pivot.columns.to_numpy(float); es=pivot.index.to_numpy(float); interaction=[]
        for i in range(len(es)-1):
            for j in range(len(ts)-1):
                interaction.append((z[i+1,j+1]-z[i+1,j]-z[i,j+1]+z[i,j])/((es[i+1]-es[i])*(ts[j+1]-ts[j])))
        rows.append(dict(route_id=int(route_id),mean_dcore_dta_c_per_c=float(np.mean(st)),
            minimum_dcore_dta_c_per_c=float(np.min(st)),maximum_dcore_dta_c_per_c=float(np.max(st)),
            mean_dcore_de_c_per_hpa=float(np.mean(se)),minimum_dcore_de_c_per_hpa=float(np.min(se)),
            maximum_dcore_de_c_per_hpa=float(np.max(se)),mean_interaction_c_per_c_hpa=float(np.mean(interaction)),
            sensitivity_scope="finite differences over tested study-area ranges only"))
    return pd.DataFrame(rows)


def build_overall_summary(results: pd.DataFrame,cases: pd.DataFrame,pairs: pd.DataFrame) -> pd.DataFrame:
    """Combine route-selection fidelity and absolute-accuracy metrics."""
    err=results.signed_error_c.to_numpy(); worst=pairs.loc[pairs.route_contrast_error_c.abs().idxmax()]
    changed=cases.loc[~cases.best_route_preserved]
    changed_text="none" if changed.empty else "; ".join(f"{x.case_id}(Ta={x.ta_ref_c:g},e={x.vapor_pressure_ref_hpa:g})" for x in changed.itertuples())
    pair_rates=pairs.groupby(["route_a","route_b"]).ranking_preserved.mean()*100
    pair_text="; ".join(f"{a}-{b}:{v:.1f}%" for (a,b),v in pair_rates.items())
    valid=pairs.loc[pairs.contrast_retention_ratio.notna() & pairs.ranking_preserved,"contrast_retention_ratio"]
    if valid.empty: behavior="not_evaluable"
    elif valid.median()<.95: behavior="generally_compressed"
    elif valid.median()>1.05: behavior="generally_exaggerated"
    else: behavior="generally_retained"
    return pd.DataFrame([dict(total_reference_cases=len(cases),total_route_simulations=len(results),
        best_route_preservation_pct=100*cases.best_route_preserved.mean(),
        complete_ranking_preservation_pct=100*cases.complete_ranking_preserved.mean(),
        pairwise_ranking_preservation_pct=100*pairs.ranking_preserved.mean(),
        pooled_mean_signed_error_c=float(err.mean()),pooled_mae_c=float(np.abs(err).mean()),
        pooled_rmse_c=float(np.sqrt(np.mean(err**2))),pooled_maximum_absolute_error_c=float(np.abs(err).max()),
        minimum_spearman_rho=float(cases.spearman_rho.min()),mean_spearman_rho=float(cases.spearman_rho.mean()),
        maximum_spearman_rho=float(cases.spearman_rho.max()),best_route_changed_cases=changed_text,
        pairwise_preservation_by_pair=pair_text,worst_pairwise_route_contrast_error_c=float(abs(worst.route_contrast_error_c)),
        worst_pairwise_case_id=worst.case_id,worst_pairwise_routes=f"{int(worst.route_a)}-{int(worst.route_b)}",
        route_contrast_interpretation=behavior,relative_error_denominator_tolerance_c=RELATIVE_ERROR_TOL_C,
        ranking_tie_tolerance_c=TIE_TOL_C,
        interpretation_note="Route-selection fidelity and absolute-prediction accuracy are separate; references are plausible uniform conditions, not assumed means.")])


def save_figure(fig: plt.Figure, base: Path) -> list[Path]:
    """Save PDF and >=300-dpi PNG."""
    fig.tight_layout(); paths=[base.with_suffix(".pdf"),base.with_suffix(".png")]
    fig.savefig(paths[0],bbox_inches="tight"); fig.savefig(paths[1],dpi=PNG_DPI,bbox_inches="tight"); plt.close(fig)
    return paths


def grid(group: pd.DataFrame, column: str):
    p=group.pivot(index="vapor_pressure_ref_hpa",columns="ta_ref_c",values=column).sort_index().sort_index(axis=1)
    return p.columns.to_numpy(float),p.index.to_numpy(float),p.to_numpy(float)


def heatmap(ta,e,z,title,label,cmap,vmin,vmax,norm=None):
    fig,ax=plt.subplots(figsize=(8.2,6.3)); im=ax.imshow(z,origin="lower",aspect="auto",cmap=cmap,
        vmin=None if norm else vmin,vmax=None if norm else vmax,norm=norm)
    ax.set_xticks(range(len(ta)),[f"{x:g}" for x in ta]); ax.set_yticks(range(len(e)),[f"{x:.2f}" for x in e])
    ax.set_xlabel("Uniform air-temperature reference, Ta [°C]"); ax.set_ylabel("Uniform vapor-pressure reference, e [hPa]"); ax.set_title(title)
    span=vmax-vmin
    for i in range(len(e)):
        for j in range(len(ta)):
            frac=(z[i,j]-vmin)/span if span else .5
            ax.text(j,i,f"{z[i,j]:.3f}",ha="center",va="center",fontsize=8.5,color="white" if frac<.2 or frac>.8 else "black")
    fig.colorbar(im,ax=ax).set_label(label); return fig


def save_surface_figures(results: pd.DataFrame,out: Path) -> list[Path]:
    """Save consistently scaled core-rise and signed-error surfaces per route."""
    paths=[]; lo=float(results.reference_final_tcore_rise_c.min()); hi=float(results.reference_final_tcore_rise_c.max())
    limit=max(float(results.signed_error_c.abs().max()),np.finfo(float).eps); norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    for route_id,g in results.groupby("route_id",sort=True):
        ta,e,z=grid(g,"reference_final_tcore_rise_c")
        paths+=save_figure(heatmap(ta,e,z,f"Route {route_id}: uniform-reference core-temperature rise","Final core-temperature rise [°C]","viridis",lo,hi),out/f"route_{route_id}_core_rise_surface")
        ta,e,z=grid(g,"signed_error_c")
        paths+=save_figure(heatmap(ta,e,z,f"Route {route_id}: error relative to resolved benchmark","Signed core-rise error [°C]","RdBu_r",-limit,limit,norm),out/f"route_{route_id}_absolute_error_surface")
    return paths


def save_ranking_figures(cases: pd.DataFrame,out: Path) -> list[Path]:
    """Save best-route and three-category ranking-preservation grids."""
    ta=np.sort(cases.ta_ref_c.unique()); e=np.sort(cases.vapor_pressure_ref_hpa.unique()); benchmark=int(cases.benchmark_best_route.iloc[0])
    best=cases.pivot(index="vapor_pressure_ref_hpa",columns="ta_ref_c",values="reference_best_route").reindex(index=e,columns=ta).to_numpy(int)
    route_ids=sorted(set(cases.reference_best_route)|set(cases.benchmark_best_route)); codes={r:i for i,r in enumerate(route_ids)}; coded=np.vectorize(codes.get)(best)
    cmap=ListedColormap(plt.cm.tab10(np.linspace(0,1,max(2,len(route_ids))))); norm=BoundaryNorm(np.arange(-.5,len(route_ids)+.5),cmap.N)
    fig,ax=plt.subplots(figsize=(8.2,6.3)); im=ax.imshow(coded,origin="lower",aspect="auto",cmap=cmap,norm=norm)
    ax.set_xticks(range(len(ta)),[f"{x:g}" for x in ta]); ax.set_yticks(range(len(e)),[f"{x:.2f}" for x in e]); ax.set_xlabel("Uniform Ta [°C]"); ax.set_ylabel("Uniform e [hPa]"); ax.set_title(f"Best route (resolved benchmark: Route {benchmark})")
    for i in range(len(e)):
        for j in range(len(ta)):
            changed=best[i,j]!=benchmark; ax.text(j,i,f"R{best[i,j]}{'*' if changed else ''}",ha="center",va="center",fontweight="bold" if changed else "normal")
            if changed: ax.add_patch(plt.Rectangle((j-.47,i-.47),.94,.94,fill=False,edgecolor="red",lw=2))
    cb=fig.colorbar(im,ax=ax,ticks=range(len(route_ids))); cb.ax.set_yticklabels([f"Route {r}" for r in route_ids]); cb.set_label("Predicted best route"); fig.text(.5,.01,"* red outline: differs from benchmark best route",ha="center")
    paths=save_figure(fig,out/"best_route_map")
    category=np.select([cases.complete_ranking_preserved,cases.best_route_preserved],[2,1],default=0)
    temp=cases.assign(category=category); z=temp.pivot(index="vapor_pressure_ref_hpa",columns="ta_ref_c",values="category").reindex(index=e,columns=ta).to_numpy(int)
    labels=["Best route changed","Best route only preserved","Complete ranking preserved"]; cm=ListedColormap(["#D95F5F","#F2C14E","#63A375"]); no=BoundaryNorm([-.5,.5,1.5,2.5],cm.N)
    fig,ax=plt.subplots(figsize=(8.2,6.3)); im=ax.imshow(z,origin="lower",aspect="auto",cmap=cm,norm=no)
    ax.set_xticks(range(len(ta)),[f"{x:g}" for x in ta]); ax.set_yticks(range(len(e)),[f"{x:.2f}" for x in e]); ax.set_xlabel("Uniform Ta [°C]"); ax.set_ylabel("Uniform e [hPa]"); ax.set_title("Route-ranking preservation")
    for i in range(len(e)):
        for j in range(len(ta)): ax.text(j,i,str(z[i,j]),ha="center",va="center")
    cb=fig.colorbar(im,ax=ax,ticks=[0,1,2]); cb.ax.set_yticklabels(labels); cb.set_label("Ranking-fidelity category")
    return paths+save_figure(fig,out/"ranking_preservation_map")


def save_prediction_envelope(results: pd.DataFrame,out: Path) -> list[Path]:
    """Distinguish selection robustness from absolute prediction spread."""
    ids=sorted(results.route_id.unique()); fig,ax=plt.subplots(figsize=(9,6.2))
    for x,route_id in enumerate(ids):
        g=results[results.route_id==route_id]; v=g.reference_final_tcore_rise_c.to_numpy(); q1,med,q3=np.percentile(v,[25,50,75]); b=float(g.benchmark_final_tcore_rise_c.iloc[0])
        ax.vlines(x,v.min(),v.max(),color="#4C78A8",lw=2); ax.vlines(x,q1,q3,color="#4C78A8",lw=9,alpha=.65); ax.scatter(x,med,marker="_",s=180,color="black",zorder=3); ax.scatter(x,b,marker="D",s=60,color="#E45756",edgecolor="black",zorder=4,label="Resolved benchmark" if x==0 else None)
    ax.set_xticks(range(len(ids)),[f"Route {r}" for r in ids]); ax.set_ylabel("Final JOS-3 core-temperature rise [°C]"); ax.set_title("Uniform-reference prediction envelope and resolved benchmark"); ax.grid(axis="y",alpha=.25); ax.legend(); fig.text(.5,.01,"Thin: min–max; thick: IQR; black: median",ha="center")
    return save_figure(fig,out/"route_prediction_envelope")


def save_benchmark_scatter(results: pd.DataFrame,out: Path) -> list[Path]:
    fig,ax=plt.subplots(figsize=(7.2,6.5)); ids=sorted(results.route_id.unique()); colors=plt.cm.tab10(np.linspace(0,1,max(2,len(ids))))
    for color,route_id in zip(colors,ids):
        g=results[results.route_id==route_id]; ax.scatter(g.benchmark_final_tcore_rise_c,g.reference_final_tcore_rise_c,s=34,alpha=.72,color=color,label=f"Route {route_id}")
    values=np.r_[results.benchmark_final_tcore_rise_c,results.reference_final_tcore_rise_c]; margin=max(np.ptp(values)*.06,.005); lo,hi=values.min()-margin,values.max()+margin
    ax.plot([lo,hi],[lo,hi],"--",color="black",label="1:1"); ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="Resolved benchmark core rise [°C]",ylabel="Uniform-reference core rise [°C]",title="Absolute prediction: benchmark versus uniform references"); ax.set_aspect("equal",adjustable="box"); ax.grid(alpha=.25); ax.legend()
    return save_figure(fig,out/"benchmark_vs_reference_scatter")


def save_sensitivity_plot(slopes: pd.DataFrame,out: Path) -> list[Path]:
    x=np.arange(len(slopes)); width=.36; fig,ax=plt.subplots(figsize=(9,6.2))
    ax.bar(x-width/2,slopes.mean_dcore_dta_c_per_c,width,label="Mean d(core rise)/dTa [°C/°C]",color="#4C78A8"); ax.bar(x+width/2,slopes.mean_dcore_de_c_per_hpa,width,label="Mean d(core rise)/de [°C/hPa]",color="#F28E2B")
    ax.set_xticks(x,[f"Route {r}" for r in slopes.route_id]); ax.set_ylabel("Adjacent finite-difference slope"); ax.set_title("Response sensitivity over tested study-area ranges"); ax.axhline(0,color="black",lw=.8); ax.grid(axis="y",alpha=.25); ax.legend(); fig.text(.5,.01,"Tested-range response, not a universal physiological sensitivity",ha="center")
    return save_figure(fig,out/"tested_range_sensitivity_slopes")


def save_csv_outputs(out,results,cases,route_summary,pairs,overall,slopes) -> list[Path]:
    """Write required tables plus tested-range slopes in reproducible order."""
    tables={"uniform_reference_all_results.csv":results.sort_values(["ta_ref_c","vapor_pressure_ref_hpa","route_id"]),
      "reference_case_ranking_summary.csv":cases.sort_values(["ta_ref_c","vapor_pressure_ref_hpa"]),
      "route_absolute_error_summary.csv":route_summary.sort_values("route_id"),
      "pairwise_route_comparison.csv":pairs.sort_values(["ta_ref_c","vapor_pressure_ref_hpa","route_a","route_b"]),
      "overall_simplification_summary.csv":overall,"tested_range_sensitivity_slopes.csv":slopes.sort_values("route_id")}
    paths=[]
    for name,df in tables.items():
        path=out/name; df.to_csv(path,index=False); paths.append(path)
    return paths


def print_summary(overall: pd.DataFrame,cases: pd.DataFrame,pairs: pd.DataFrame) -> None:
    x=overall.iloc[0]; print("\nKey summary metrics")
    print(f"  Best-route preservation: {x.best_route_preservation_pct:.1f}%")
    print(f"  Complete-ranking preservation: {x.complete_ranking_preservation_pct:.1f}%")
    print(f"  Pairwise-ordering preservation: {x.pairwise_ranking_preservation_pct:.1f}%")
    print(f"  Spearman rho min/mean/max: {x.minimum_spearman_rho:.3f}/{x.mean_spearman_rho:.3f}/{x.maximum_spearman_rho:.3f}")
    print(f"  Cases changing best route: {x.best_route_changed_cases}")
    print(f"  Worst pairwise contrast error: {x.worst_pairwise_route_contrast_error_c:.6f} C ({x.worst_pairwise_case_id}, routes {x.worst_pairwise_routes})")
    print(f"  Route contrasts: {x.route_contrast_interpretation.replace('_',' ')}")
    print(f"  Pooled signed error / MAE / RMSE / max |error|: {x.pooled_mean_signed_error_c:+.6f} / {x.pooled_mae_c:.6f} / {x.pooled_rmse_c:.6f} / {x.pooled_maximum_absolute_error_c:.6f} C")
    print("  Pairwise ordering preservation:")
    for (a,b),rate in (pairs.groupby(["route_a","route_b"]).ranking_preserved.mean()*100).items(): print(f"    Routes {a}-{b}: {rate:.1f}%")


def main() -> None:
    args=parse_args(); ta_values,e_values=validate_settings(args)
    routes_path=Path(args.routes_pkl); require_file(routes_path,"selected routes"); require_file(Path(args.weather_csv),"weather CSV")
    tree,matrix,hours=load_mrt(Path(args.mrt_results_dir)); selection=load_selected_routes(routes_path); raw=selection.get("routes")
    if not isinstance(raw,list) or not raw: raise ValueError(f"{routes_path} contains no routes")
    routes=prepare_routes(raw,tree,args); benchmark=validate_baseline(Path(args.baseline_csv),[r.route_id for r in routes])
    weather=WeatherProvider(csv_path=args.weather_csv,strict=True)
    for route in routes:
        wind=np.asarray(weather.wind_ms(route.arrival_hour%24),float)
        if not np.isfinite(wind).all(): raise ValueError(f"Non-finite resolved wind for route {route.route_id}")
    subject=resolve_subject(args); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); ncases=len(ta_values)*len(e_values)
    print("Uniform-reference JOS-3 factorial analysis"); print(f"  Routes: {len(routes)}"); print(f"  Ta levels: {len(ta_values)}"); print(f"  Vapor-pressure levels: {len(e_values)}"); print(f"  Reference cases: {ncases}"); print(f"  Total JOS-3 simulations: {ncases*len(routes)}"); print("  Uniform Ta/e; resolved MRT and wind retained")
    results,cases,pairs=run_factorial_analysis(routes,matrix,hours,weather,subject,args,ta_values,e_values,benchmark)
    route_summary=calculate_route_errors(results); slopes=calculate_sensitivity_slopes(results); overall=build_overall_summary(results,cases,pairs)
    paths=save_csv_outputs(out,results,cases,route_summary,pairs,overall,slopes)
    paths+=save_surface_figures(results,out)+save_ranking_figures(cases,out)+save_prediction_envelope(results,out)+save_benchmark_scatter(results,out)+save_sensitivity_plot(slopes,out)
    print_summary(overall,cases,pairs); print("\nSaved outputs")
    for path in paths: print(f"  {path}")


if __name__ == "__main__":
    main()
