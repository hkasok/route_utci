#!/usr/bin/env python3
"""Recompute paper numbers that no pipeline output stores, into derived_values.json.

Read-only with respect to the pipeline: it imports/reads pipeline code and
run_output files but writes only next to this script. Slower than the checker
(a few minutes); re-run it after the pipeline outputs change, then re-run
check_numbers.py.

usage: python3 derive_values.py [PIPELINE_ROOT] [PAPER_DIR]
"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else "/media/harshin/data_drive/route based UTCI"
PAPER = sys.argv[2] if len(sys.argv) > 2 else "/home/harshin/files/fastUTEC paper"
OUT = os.path.join(HERE, "derived_values.json")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

out = json.load(open(OUT)) if os.path.exists(OUT) else {}
SIG = 5.670374419e-8


def wallview():
    """Supp S6: per-scene wall share of the globe's view, direct -3 K load response,
    facade-albedo-0.60 wall temperature change (sun above 5 deg)."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    res = {}
    for n in range(1, 7):
        c = f"lisbon{n}"
        th = f"{R}/run_output/{c}/thermal_out"
        Wg = sp.load_npz(f"{th}/globe_view_matrix.npz")
        fz = np.load(f"{th}/facets.npz")
        pm = np.load(f"{th}/point_map.npy")
        walls = fz["cls"] == 1
        wv = np.asarray(Wg[:, walls].sum(axis=1)).ravel()[pm]
        P = pd.read_csv(f"{R}/run_output/{c}/validation/mrt_lisbon/radiant_flux_comparison_points.csv")
        P = P[P.period == "day"]
        case = json.load(open(f"{R}/input/{c}/case.json"))
        ox, oy = case["coordinates"]["local_origin_x"], case["coordinates"]["local_origin_y"]
        path = np.load(f"{R}/run_output/{c}/mrt_facet_out/path_xyz.npy")[:, :2]
        _, i = cKDTree(path).query(np.column_stack([P.x_projected_m - ox, P.y_projected_m - oy]))
        T = np.load(f"{th}/facet_T_matrix_K.npy")
        eps = np.load(f"{th}/facet_eps.npy")
        t = pd.read_csv(f"{R}/run_output/{c}/mrt_facet_out/times.csv")
        day = np.flatnonzero(t.elevation_deg.values > 5)
        dJ = (eps[walls][None, :] * SIG * ((T[day][:, walls] - 3) ** 4 - T[day][:, walls] ** 4)).mean(0)
        dL = np.asarray(Wg[:, walls] @ dJ).ravel()[pm][i].mean() * 0.957
        T2 = np.load(f"{R}/facade_tests/albedo_060/run_output/{c}/thermal_out/facet_T_matrix_K.npy")
        dT = float((T2[day][:, walls] - T[day][:, walls]).mean())
        res[c] = dict(wall_view_pct=float(100 * wv[i].mean()), dload_m3K=float(dL), facade060_dT=dT)
    return res


def phase():
    """Supp S3: lag maximising the cross-correlation of emulated vs measured globe per walk."""
    src = open(f"{R}/paper_adhoc/phase_check.py").read().split("for tag,root in")[0]
    ns = {}
    cwd = os.getcwd()
    os.chdir(R)
    try:
        exec(src, ns)
        o = ns["analyse"](lambda c: f"{R}/run_output/lisbon{c}")
    finally:
        os.chdir(cwd)
    day = [x for x in o if x[1] == "day"]
    return dict(n_walks=len(o), n_model_ahead=sum(x[2] > 0 for x in o), n_day_ahead=sum(x[2] > 0 for x in day),
                max_day_lead_s=float(max(x[2] for x in day)), walks=[list(map(str, x)) for x in o])


def flip_calibration():
    """Supp S14: mean simulated flipped-stretch length for 2 m correlated flips."""
    import glob
    sys.path.insert(0, R)
    from paper_ranking_uncertainty import correlated_flips
    s = json.load(open(os.path.join(PAPER, "validation_summary.json")))["shadow"]
    rate = 1.0 - s["classification_agreement"]
    steps = []
    for f in sorted(glob.glob(f"{R}/run_output/lisbon*/validation/mrt_lisbon/radiant_flux_comparison_points.csv")):
        d = pd.read_csv(f)
        d = d[d.period == "day"]
        for _, g in d.groupby("route_id"):
            steps.append(np.median(np.diff(g.sort_values("seq").distance_along_route_m.values)))
    sample_m = float(np.median(steps))
    stride = max(1, int(round(sample_m / 2.0)))
    f = correlated_flips(400_000, 2.0, rate, 2.0, np.random.default_rng(11))[::stride]
    edges = np.diff(np.r_[0, f.astype(int), 0])
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return dict(mean_run_2m=float(runs.mean() * stride * 2.0), sample_spacing_m=sample_m, flip_rate=rate)


def reloc():
    """Main Sec. 3.1: Lisbon samples whose position was moved out of a building footprint."""
    from scipy.spatial import cKDTree
    out_ = []
    for i in range(1, 7):
        d = f"{R}/run_output/lisbon{i}/mrt_facet_out/"
        Pp = np.load(d + "path_xyz.npy")
        Rr = np.load(d + "receptor_relocation_m.npy")
        p = pd.read_csv(f"{R}/run_output/lisbon{i}/validation/mrt_lisbon/radiant_flux_comparison_points.csv")
        fl = pd.concat([pd.read_csv(f"{R}/run_output/lisbon{i}/viz/route_utci/radiant_flux_contributions/route_{r}_radiant_flux_contributions.csv",
                                    usecols=["route_id", "point_id", "x", "y"]) for r in (1, 2)])
        m = p.merge(fl.rename(columns={"point_id": "seq"}), on=["route_id", "seq"])
        _, idx = cKDTree(Pp[:, :2]).query(m[["x", "y"]].values)
        m["reloc_m"] = Rr[idx]
        out_.append(m)
    m = pd.concat(out_, ignore_index=True)
    m = m[~((m.period == "day") & (m.measured_swin_wm2 < 10) & (m.measured_swout_wm2 < 0))]
    day = m[m.period == "day"]
    night = m[m.period == "night"]
    dm = day[day.reloc_m > 0]
    g = dm.trec_atmospheric_ghi_wm2
    return dict(day_moved_frac=float((day.reloc_m > 0).mean()), night_moved_frac=float((night.reloc_m > 0).mean()),
                median_move_m=float(m.loc[m.reloc_m > 0, "reloc_m"].median()),
                moved_day_model_sunlit=float((dm.sensor_shortwave_down_Wm2 > 0.5 * g).mean()),
                moved_day_measured_sunlit=float((dm.measured_swin_wm2 > 0.5 * g).mean()), n_moved_day=int(len(dm)))


def campus_geometry():
    """Main Sec. 3.2: tile extent, Route 3 share within 25 m of the tile edge, covered-passage shares."""
    import importlib.util
    sys.path.insert(0, R)
    spec = importlib.util.spec_from_file_location("s05", f"{R}/05_mrt_network_raytrace.py")
    s05 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s05)
    d = f"{R}/run_output/MMC/mrt_facet_out/"
    xyz = np.load(d + "path_xyz.npy")
    seg = np.load(d + "path_segment_id.npy")
    Rr = np.load(d + "receptor_relocation_m.npy")
    geo = f"{R}/input/MMC/geometry"
    raw = open(f"{geo}/ground_and_water_final.stl", "rb").read()[84:]
    v = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))["v"].reshape(-1, 3).astype(float)
    lo, hi = v[:, :2].min(0), v[:, :2].max(0)
    edge = np.minimum(xyz[:, :2] - lo, hi - xyz[:, :2]).min(1)
    gi = s05.get_intersector(s05.load_mesh(f"{geo}/ground_and_water_final.stl"))
    bi = s05.get_intersector(s05.load_mesh(f"{geo}/building_final.stl"))
    zg = s05.ground_height_lookup(xyz[:, :2], gi)
    inside = s05.inside_building_footprint(xyz[:, :2], zg, bi)
    cov = inside & (Rr == 0)
    res = dict(tile_x_km=float((hi[0] - lo[0]) / 1000), tile_y_km=float((hi[1] - lo[1]) / 1000))
    for s in np.unique(seg):
        k = int(s) + 1
        res[f"route{k}_covered_pct"] = float(100 * cov[seg == s].mean())
        res[f"route{k}_within25m_frac"] = float((edge[seg == s] < 25).mean())
    return res


def jos3_bsa():
    from pythermalcomfort.models import JOS3
    return dict(bsa_m2=float(np.sum(JOS3(height=1.72, weight=74, age=30, sex="male", fat=15, ci=2.59).bsa)))


def utci_wind10():
    """Supp S12: change of walk-mean UTCI when the cart wind is referred to 10 m."""
    from pythermalcomfort.models import utci
    res = {}
    for c in range(1, 7):
        P = pd.read_csv(f"{R}/run_output/lisbon{c}/viz/route_utci/routes_points.csv")
        for rid, g in P.groupby("route_id"):
            per = str(g.route_name.iloc[0]).split("_")[0] if "route_name" in g else str(rid)
            kw = dict(tdb=g.ta_c.values, tr=g.tmrt_c.values, rh=g.rh_pct.values, limit_inputs=False, round_output=False)
            u_loc = np.asarray(utci(v=g.local_wind_ms.values, **kw).utci)
            u_10 = np.asarray(utci(v=g.wind_ms.values, **kw).utci)
            t = g.arrival_hour.values
            res[f"lisbon{c}_{per}"] = float((np.trapz(u_loc, t) - np.trapz(u_10, t)) / (t[-1] - t[0]))
    return res


def utci_prefill():
    """Supp S15: extra t38 if the 3-min window were pre-filled with the sunlit start UTCI."""
    sys.path.insert(0, R)
    import route_measures as rm
    P = pd.read_csv(f"{R}/run_output/MMC/viz/route_utci/routes_points.csv")
    start = float(pd.read_csv(f"{R}/run_output/MMC/viz/route_utci/route_ranking_summary.csv").start_utci_c.iloc[0])
    res = {}
    for rid, g in P.groupby("route_id"):
        u = g.utci_c.values
        t = (g.arrival_hour.values - g.arrival_hour.values[0]) * 60
        res[f"route{rid}"] = float(rm.minutes_at_or_above(rm.backward_mean(u, t, before=u[0]), t, 38)
                                   - rm.minutes_at_or_above(rm.backward_mean(u, t, before=start), t, 38))
    return res


def linke_clim():
    """Main Sec. 2.6: climatological Linke turbidity (pvlib monthly table) at the case sites/dates."""
    import pvlib
    vals = []
    for i in range(1, 7):
        pv = json.load(open(f"{R}/input/lisbon{i}/weather/weather_from_sensors_provenance.json"))["location"]
        ts = pd.DatetimeIndex([pd.Timestamp(pv["date"] + " 13:00", tz="Europe/Lisbon")])
        vals.append(float(pvlib.clearsky.lookup_linke_turbidity(ts, pv["latitude"], pv["longitude"]).iloc[0]))
    return dict(mean=float(np.mean(vals)), values=vals)


def linke_campus():
    """Main Sec. 3.2: Linke turbidity of the campus clear sky, as Location.get_clearsky
    interpolates it for the case day (pvlib day-of-year interpolation of the monthly table)."""
    import pvlib
    case = json.load(open(f"{R}/input/MMC/case.json"))
    loc = case["location"]
    ts = pd.date_range(case["simulation_defaults"]["date"] + " 07:00", periods=13, freq="1h",
                       tz=loc["timezone"])
    v = pvlib.clearsky.lookup_linke_turbidity(ts, loc["latitude"], loc["longitude"]).to_numpy(float)
    return dict(min=float(v.min()), max=float(v.max()), median=float(np.median(v)))


def tie_rule_checks():
    """Supp S15: 13:00 pairs resolved by the band test that the other-departure
    threshold would tie, and the departure-sweep margins at a 40 m correlation length."""
    import itertools
    from pathlib import Path
    sys.path.insert(0, R)
    cwd = os.getcwd()
    os.chdir(R)
    try:
        import paper_route_measures_table as T
        post = Path("run_output/MMC/postprocessing")
        main = post / "ranking_uncertainty_ws_main"
        res = json.loads((main / "summary.json").read_text())["measures"]
        rules = T.bootstrap_rules(main, list(res))
        src = T.load(Path("run_output/MMC/viz"))
        resolved, lost = 0, {}
        for row in T.ROWS:
            if row[1] is None or row[5] is None:
                continue
            _l, _u, s_, col, _d, key = row
            v = {r: float(src[s_].loc[r, col]) for r in T.ROUTES}
            for a, b in itertools.combinations(T.ROUTES, 2):
                if rules[key]["resolved"][(13.0, (a, b))]:
                    resolved += 1
                    thr = rules[key]["shift"][(a, b)]
                    if abs(v[b] - v[a]) <= thr:
                        lost[f"{col}:{a}-{b}"] = {"gap": v[b] - v[a], "threshold": thr}
        out40 = {}
        r40 = T.bootstrap_rules(post / "ranking_uncertainty_ws_corr40m", ["tcore_rise_c", "walk_utci_c"])
        sweep = post / "route_measures_sweep"
        for tag in ("1430", "1800"):
            sw = T.load(sweep / f"dep_{tag}")
            core = sw["jos3"]["final_tcore_rise_c"]
            utci = sw["utci"]["walk_utci_c"]
            out40[tag] = {"core_1_3_gap": float(core.loc[3] - core.loc[1]),
                          "core_1_3_threshold": r40["tcore_rise_c"]["shift"][(1, 3)],
                          "utci_1_2_gap": float(utci.loc[2] - utci.loc[1]),
                          "utci_1_2_threshold": r40["walk_utci_c"]["shift"][(1, 2)],
                          "utci_1_4_gap": float(utci.loc[4] - utci.loc[1]),
                          "utci_1_4_threshold": r40["walk_utci_c"]["shift"][(1, 4)]}
        return {"resolved_13h": resolved, "would_tie_13h": len(lost), "pairs": lost, "corr40m": out40}
    finally:
        os.chdir(cwd)


def campus_paving_check():
    """Supp S16: campus route results with the library's revised paving-stone and
    plaza properties (variants/mmc_paving_library) against the published run."""
    out = {}
    for f, cols in (("route_jos3", ["final_tcore_rise_c", "mean_tmrt_c"]),
                    ("route_utci", ["walk_utci_c"])):
        a = pd.read_csv(f"{R}/run_output/MMC/viz/{f}/route_ranking_summary.csv").set_index("route_id")
        b = pd.read_csv(f"{R}/variants/mmc_paving_library/run_output/MMC/viz/{f}/"
                        "route_ranking_summary.csv").set_index("route_id")
        for c in cols:
            out[c + "_max_abs_change"] = float((b[c] - a[c]).abs().max())
    f = np.load(f"{R}/run_output/MMC/thermal_out/facets.npz", allow_pickle=True)
    names = np.load(f"{R}/run_output/MMC/thermal_out/facet_material_name.npy", allow_pickle=True)
    m = np.isin(names, ["paving_stone_pedestrian", "pedestrian_plaza"])
    out["selected_area_pct"] = float(100 * f["area"][m].sum() / f["area"].sum())
    used = json.load(open(f"{R}/run_output/MMC/thermal_out/materials_used.json"))
    out["campus_k"] = used["paving_stone_pedestrian"]["k"]
    out["campus_depths"] = [used["paving_stone_pedestrian"]["depth"], used["pedestrian_plaza"]["depth"]]
    return out


def forcing_variants():
    """Main Secs 2.4, 4.1 and Supp S2, S16: Lisbon validation under the two forcing
    sensitivity runs of 26 Oct 2026 (variants/forcing_measured_level,
    variants/wind_floor_0p5) and the unchanged control rerun, with the paper's
    statistics (paper_adhoc/heldout_summary.summarise)."""
    import glob
    import warnings
    from pathlib import Path
    sys.path.insert(0, os.path.join(R, "paper_adhoc"))
    sys.path.insert(0, R)
    import heldout_summary as H
    cases = [f"lisbon{i}" for i in range(1, 7)]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, root in (("published", Path(R)), ("control", Path(R, "variants/control_rerun")),
                           ("measured_level", Path(R, "variants/forcing_measured_level")),
                           ("wind_floor_0p5", Path(R, "variants/wind_floor_0p5"))):
            p = H.summarise(cases, root, ["day", "night"], [])["pooled"]
            per = H.summarise(cases, root, ["day"], [])["per_walk"]
            out[name] = {"globe_day_mbe": p["globe"]["day"]["mbe"],
                         "globe_day_rmse": p["globe"]["day"]["rmse"],
                         "globe_night_mbe": p["globe"]["night"]["mbe"],
                         "kdown_day_mbe": p["radiometer"]["K_down_day"]["mbe"],
                         "lup_day_mbe": p["radiometer"]["L_up_day"]["mbe"],
                         "per_case_day_mbe": {c: per[c]["globe"]["day"]["mbe"] for c in cases}}
    factors = [json.load(open(f))["factor"] for f in sorted(glob.glob(
        f"{R}/variants/forcing_measured_level/input_alt/lisbon*/config/solar_forcing_scale.json"))]
    out["measured_level_factor_range"] = [min(factors), max(factors)]
    return out


def review_checks_1026():
    """Main Secs 4.1, 4.5 and Supp S2, S15, S17 (review of 26 Oct 2026): globe
    statistics of an inputs-only predictor (globe = forcing air temperature),
    the Lisbon 1 dropout gap, the 14:00 Route 1/3 core-rise margin of the 2 m
    sweep, and held-out pooling with Lisbon 5 walk 3 forced with the 10 K
    diurnal fallback (variants/l5w3_diurnal_fallback)."""
    import warnings
    from pathlib import Path
    sys.path.insert(0, os.path.join(R, "paper_adhoc"))
    sys.path.insert(0, R)
    out = {}
    fr = []
    for i in range(1, 7):
        d = pd.read_csv(f"{R}/run_output/lisbon{i}/validation/mrt_lisbon/radiant_flux_comparison_points.csv")
        d["case"] = i
        fr.append(d)
    d = pd.concat(fr)
    d = d[~((d.period == "day") & (d.measured_swin_wm2 < 10) & (d.measured_swout_wm2 < 0))]
    d = d[~d.globe_spinup_affected.astype(bool)].dropna(
        subset=["measured_black_globe_temperature_c", "globe_transient_temperature_C"])
    for per in ("day", "night"):
        x = d[d.period == per]
        m = x.measured_black_globe_temperature_c
        for name, p in (("model", x.globe_transient_temperature_C), ("air_only", x.trec_route_air_temperature_c)):
            e = p - m
            ec = e - e.groupby(x.case).transform("mean")
            out[f"{per}_{name}"] = {"n": int(len(x)), "mbe": float(e.mean()),
                                    "rmse": float(np.sqrt((e ** 2).mean())),
                                    "within_crmse": float(np.sqrt((ec ** 2).mean()))}
    x = pd.read_csv(f"{R}/run_output/lisbon1/validation/mrt_lisbon/radiant_flux_comparison_points.csv")
    x = x[x.period == "day"].sort_values("timestamp_utc")
    s = x.distance_along_route_m.to_numpy()
    j = int(np.where(s <= 1103.3)[0][-1])
    out["lisbon1_dropout_gap_m"] = float(s[j + 1] - s[j])
    sw = pd.read_csv(f"{R}/run_output/MMC/postprocessing/departure_sweep/departure_sweep_fixedclothing.csv")
    sw = sw.pivot(index="departure_hour", columns="route_id", values="tcore_rise_c")
    out["sweep2m_core_gap_3_1"] = {str(h): float(sw.loc[h, 3] - sw.loc[h, 1]) for h in (13.5, 14.0, 14.5)}
    import heldout_summary as H
    cases = [f"lisbon{a}_w{w}" for a in range(1, 7) for w in (2, 3)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, root in (("published", Path(R)),
                           ("l5w3_fallback", Path(R, "variants/l5w3_diurnal_fallback/pooled_root"))):
            p = H.summarise(cases, root, ["day"], [])["pooled"]
            out["heldout_" + name] = {"globe_day_mbe": p["globe"]["day"]["mbe"],
                                      "globe_day_rmse": p["globe"]["day"]["rmse"],
                                      "agreement": p["shadow"]["classification_agreement"],
                                      "kappa": p["shadow"]["cohen_kappa"]}
    prov = json.load(open(f"{R}/variants/l5w3_diurnal_fallback/input_alt/lisbon5_w3/weather/"
                          "weather_from_sensors_provenance.json"))["meteorology"]["diurnal_fit"]
    out["l5w3_anchor_temperature_c"] = prov["anchor_temperature_c"]
    pub = json.load(open(f"{R}/input/lisbon5_w3/weather/weather_from_sensors_provenance.json"))["meteorology"]
    out["l5w3_published_reconstructed_rows"] = pub["reconstructed_rows"]
    lis = [f"lisbon{i}" for i in range(1, 7)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, root in (("published", Path(R)), ("paving_old", Path(R, "variants/lisbon_paving_old"))):
            p = H.summarise(lis, root, ["day", "night"], [])["pooled"]
            out["paving_" + name] = {"globe_day_mbe": p["globe"]["day"]["mbe"],
                                     "globe_night_mbe": p["globe"]["night"]["mbe"],
                                     "lup_day_mbe": p["radiometer"]["L_up_day"]["mbe"]}
    old = json.load(open(f"{R}/run_output/MMC/thermal_out/materials_used.json"))["paving_stone_pedestrian"]
    out["paving_old_admittance"] = float(np.sqrt(old["k"] * old["C"] * 2 * np.pi / 86400.0))
    return out


TASKS = {"wallview": wallview, "phase": phase, "flip_calibration": flip_calibration, "reloc": reloc,
         "campus_geometry": campus_geometry, "jos3_bsa": jos3_bsa, "utci_wind10": utci_wind10,
         "utci_prefill": utci_prefill, "linke_clim": linke_clim,
         "linke_campus": linke_campus, "tie_rule_checks": tie_rule_checks,
         "campus_paving_check": campus_paving_check,
         "forcing_variants": forcing_variants, "review_checks_1026": review_checks_1026}

for name in (sys.argv[3].split(",") if len(sys.argv) > 3 else TASKS):
    try:
        out[name] = TASKS[name]()
        print(name, "ok")
    except Exception as ex:  # noqa: BLE001
        print(name, "FAILED:", type(ex).__name__, ex)
json.dump(out, open(OUT, "w"), indent=1)
