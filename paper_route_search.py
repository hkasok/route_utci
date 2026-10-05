#!/usr/bin/env python3
"""paper_route_search.py -- search the campus network for a shaded route that
is longer than an existing route yet predicted to strain less.

STANDALONE. Used once, to construct campus Route 1 (Sec. 3.2 of the paper):

1. DIRECT SUN ON EVERY EDGE. Each edge of the domain-clipped pedestrian
   network is sampled every 2 m at the receptor height, and direct-beam
   transmission is traced every 10 min from 07:00 to 19:00 with stage 05's own
   functions (buildings opaque, crowns Beer-Lambert). The share of each edge
   lying beneath a building solid is recorded; edges more than 25 % covered
   (breezeways, building passages) are excluded, because the level-of-detail-1
   reconstruction puts a walker there inside a solid.
2. PROXY. On the full-model departure sweep of the existing routes (stage 09
   physics, fixed clothing) the walking part of the core rise is regressed on
   walk duration and the radiant part on the beam dose on a standing body
   (tau * f_p(beta) * DNI integrated along the walk at arrival times).
3. CANDIDATES. 400 k-shortest paths by length plus shade- and sun-weighted
   shortest paths (several weights, departures 09-17 h) between the common
   endpoints; each is scored at every sweep departure with the proxy.
4. SCREEN. Each candidate is compared with every existing route it is at least
   50 m longer than; the output lists the departures at which it is predicted
   to strain more than 0.025 C less. Shortlisted routes must then be run
   through the full pipeline -- the proxy only ranks.

Usage: python3 paper_route_search.py [--case MMC] [--out DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("s05", ROOT / "05_mrt_network_raytrace.py")
s05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s05)

DS, H, K_DIR, U = 2.0, 1.1, 0.45, 1.3
COVER_MAX = 0.25
H0 = np.arange(7, 19.01, 0.5)


def f_p(el):
    return 0.308 * np.cos(np.deg2rad(el * (0.998 - el ** 2 / 50000)))


def edge_sun(graphml, geo, times_csv):
    G = ox.load_graphml(graphml)
    edges, pts, owner = [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        geom = d.get("geometry") or LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                                                (G.nodes[v]["x"], G.nodes[v]["y"])])
        xy = s05.sample_polyline(np.asarray(geom.coords)[:, :2], DS)
        if len(xy) == 0:
            xy = np.asarray(geom.coords)[:1, :2]
        edges.append((u, v, k, float(geom.length)))
        owner.append(np.full(len(xy), len(edges) - 1))
        pts.append(xy)
    xy, owner = np.vstack(pts), np.concatenate(owner)
    gi = s05.get_intersector(s05.load_mesh(geo / "ground_and_water_final.stl"))
    bi = s05.get_intersector(s05.load_mesh(geo / "building_final.stl"))
    vi = s05.get_intersector(s05.load_mesh(geo / "vegetation_final.stl"))
    xyz = np.column_stack([xy, s05.ground_height_lookup(xy, gi) + H])
    covered = bi.intersects_any(xyz, np.tile([0, 0, 1.0], (len(xyz), 1)))
    cover = (np.bincount(owner, weights=covered, minlength=len(edges))
             / np.maximum(1, np.bincount(owner, minlength=len(edges))))
    t = pd.read_csv(times_csv)
    hh = (pd.to_datetime(t.time).dt.hour + pd.to_datetime(t.time).dt.minute / 60).values
    sel = (hh >= 7) & (hh <= 19.5) & (t.elevation_deg.values > 0)
    T = t[sel].reset_index(drop=True)
    tau = np.zeros((len(T), len(xyz)), np.float32)
    for i, r in T.iterrows():
        sv = s05.sun_vector_enu(r.azimuth_deg, r.elevation_deg)
        tau[i] = s05.direct_solar_transmission_batched(xyz, sv, bi, vi, K_DIR, 20000)
    return dict(edges=edges, owner=owner, tau=tau, cover=cover, hours=hh[sel],
                elev=T.elevation_deg.values, dni=T.DNI_Wm2.values)


def beam_dose(tau_at, hours, elev, dni, cum, h0):
    """Beam on a standing body integrated along the walk (kJ m-2)."""
    ta = h0 + cum / U / 3600
    j = np.clip(np.searchsorted(hours, ta) - 1, 0, len(hours) - 2)
    w = (ta - hours[j]) / (hours[j + 1] - hours[j])
    idx = np.arange(len(cum))
    tv = (1 - w) * tau_at[j, idx] + w * tau_at[j + 1, idx]
    load = tv * np.maximum(0, (1 - w) * f_p(elev[j]) * dni[j] + w * f_p(elev[j + 1]) * dni[j + 1])
    return float((load * np.gradient(cum) / U).sum() / 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default="MMC")
    ap.add_argument("--graphml", default="run_output/MMC/osm_paths/pedestrian_network_domain.graphml")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "run_output" / args.case / "postprocessing" / "route_search"
    out.mkdir(parents=True, exist_ok=True)
    mrt = ROOT / "run_output" / args.case / "mrt_facet_out"

    print("1. direct sun and building cover on every edge ...")
    E = edge_sun(ROOT / args.graphml, ROOT / "input" / args.case / "geometry", mrt / "times.csv")
    print(f"   {len(E['edges'])} directed edges; {(E['cover'] > COVER_MAX).sum()} more than "
          f"{COVER_MAX:.0%} under building solids (excluded)")

    print("2. proxy calibrated on the full-model sweep ...")
    t = pd.read_csv(mrt / "times.csv")
    hh = (pd.to_datetime(t.time).dt.hour + pd.to_datetime(t.time).dt.minute / 60).values
    tau_r = np.load(mrt / "direct_transmission_matrix.npy")
    xyz = np.load(mrt / "path_xyz.npy")
    seg = np.load(mrt / "path_segment_id.npy")
    sw = pd.read_csv(ROOT / "run_output" / args.case / "postprocessing" / "departure_sweep"
                     / "departure_sweep_fixedclothing.csv")
    rows = []
    for r in sorted(sw.route_id.unique()):
        m = seg == (r - 1)
        cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xyz[m, :2], axis=0).T))])
        for h0 in H0:
            q = sw[(sw.route_id == r) & np.isclose(sw.departure_hour, h0)].iloc[0]
            rows.append(dict(route=r, h0=h0, dur=cum[-1] / U / 60,
                             dose=beam_dose(tau_r[:, m], hh, t.elevation_deg.values,
                                            t.DNI_Wm2.values, cum, h0),
                             walk=q.neutral_tcore_rise_c, rad=q.radiation_attributable_rise_c))
    D = pd.DataFrame(rows)
    a_walk = np.polyfit(D.dur, D.walk, 1)[0]
    b_rad = np.polyfit(D.dose, D.rad, 1)[0]
    print(f"   walking {a_walk:.4f} C/min; radiation {b_rad * 1000:.3f} C per MJ m-2 of beam")

    print("3. candidate paths ...")
    edges, owner, tau = E["edges"], E["owner"], E["tau"]
    order = np.argsort(owner, kind="stable")
    starts = np.searchsorted(owner[order], np.arange(len(edges) + 1))
    def pts_of(i):
        return order[starts[i]:starts[i + 1]]
    Gs = nx.Graph()
    for i, (u, v, k, L) in enumerate(edges):
        if E["cover"][i] > COVER_MAX:
            continue
        if not Gs.has_edge(u, v) or Gs[u][v]["length"] > L:
            Gs.add_edge(u, v, length=L, eid=i)
    routes_dir = ROOT / "input" / args.case / "routes"
    from generate_route import load_routes_directory
    routes = load_routes_directory(routes_dir)
    pos = {n: np.array([float(d["x"]), float(d["y"])]) for n, d in ox.load_graphml(ROOT / args.graphml).nodes(data=True)}
    nearest = lambda p: min(Gs.nodes, key=lambda n: np.hypot(*(pos[n] - p)))
    S = nearest(routes[0]["xy"][0]); T_ = nearest(routes[0]["xy"][-1])
    cands = {}
    for n, pth in enumerate(nx.shortest_simple_paths(Gs, S, T_, weight="length")):
        cands.setdefault(tuple(pth), f"k{n}")
        if n >= 399:
            break
    for h in (9, 11, 13, 15, 17):
        j = int(np.argmin(abs(E["hours"] - h)))
        for lam in (0.25, 0.5, 1, 2, 4, 8):
            for sign in (+1, -1):
                for u, v, dd in Gs.edges(data=True):
                    s = float(tau[j, pts_of(dd["eid"])].mean())
                    dd["w"] = dd["length"] * max(0.05, 1 + sign * lam * (s - 0.5))
                cands.setdefault(tuple(nx.shortest_path(Gs, S, T_, weight="w")),
                                 f"{'shade' if sign > 0 else 'sun'}{lam}@{h}")
    print(f"   {len(cands)} candidate paths")

    rows = []
    for pth, tag in cands.items():
        p = np.concatenate([pts_of(Gs[u][v]["eid"]) if edges[Gs[u][v]["eid"]][0] == u
                            else pts_of(Gs[u][v]["eid"])[::-1] for u, v in zip(pth[:-1], pth[1:])])
        L = sum(Gs[u][v]["length"] for u, v in zip(pth[:-1], pth[1:]))
        cum = np.linspace(0, L, len(p))
        row = dict(tag=tag, length_m=L, dur=L / U / 60,
                   cover=float(E["cover"][owner[p]].mean()), path=pth)
        for h0 in H0:
            row[f"d{h0:g}"] = beam_dose(tau[:, p], E["hours"], E["elev"], E["dni"], cum, h0)
        rows.append(row)
    C = pd.DataFrame(rows)

    print("4. screen against the existing routes ...")
    out_rows = []
    for r in sorted(D.route.unique()):
        R = D[D.route == r].set_index("h0")
        for i, c in C.iterrows():
            if (c.dur - R.dur.iloc[0]) * U * 60 < 50:
                continue
            adv = np.array([a_walk * (c.dur - R.loc[h, "dur"]) + b_rad * (c[f"d{h:g}"] - R.loc[h, "dose"])
                            for h in H0])
            better = H0[adv < -0.025]
            if len(better):
                out_rows.append(dict(candidate=i, tag=c.tag, length_m=round(c.length_m, 1),
                                     vs_route=r, longer_by_m=round((c.dur - R.dur.iloc[0]) * U * 60, 1),
                                     n_departures_better=len(better), best_pred_c=round(adv.min(), 4),
                                     at=float(H0[adv.argmin()]), cover=round(c.cover, 3)))
    res = pd.DataFrame(out_rows)
    if len(res):
        res = res.sort_values(["n_departures_better", "best_pred_c"], ascending=[False, True])
        print(res.head(15).to_string(index=False))
    res.to_csv(out / "screen.csv", index=False)
    pickle.dump(dict(candidates=C, proxy=dict(a_walk=a_walk, b_rad=b_rad)), open(out / "candidates.pkl", "wb"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
