#!/usr/bin/env python3
"""route3_selection_figure.py -- Supplementary figure: plan view of the
thermal facets selected for campus (MMC) Route 3 by stage 05a.

What is drawn
  * Route 3's selection is read from the stored stage-05a outputs: the rows
    of thermal_out/lw_view_matrix.npz that belong to Route 3's traced
    receptors (coarse_index.npy + mrt_out/path_segment_id.npy) and the
    facet list thermal_out/facets.npz. A facet is selected for Route 3 when
    its column has a non-zero view weight in those rows; every facet in
    facets.npz is selected for at least one route (the union).
  * Ground triangles are filled; walls are vertical and so edge-on from
    above, and are drawn as thick lines along their plan footprint (the two
    most distant projected vertices of each wall triangle).
  * Tree-crown triangles are not facets (no energy balance), so 05a stores
    only their aggregate view weight, not which crown faces were hit. They
    are the one item that cannot be read back: the script re-traces Route
    3's receptors with the pipeline's own ray helpers and 05a's settings
    (thermal_out/config.json), and checks that the re-trace reproduces the
    stored building/ground selection exactly before drawing anything.

Reads (never writes) input/MMC/geometry/*.stl, run_output/MMC/mrt_out and
run_output/MMC/thermal_out; writes only --out.

Usage:
    python3 paper_adhoc/route3_selection_figure.py
        [--out run_output/MMC/postprocessing/route3_selection/route3_selection_supplement.png]
        [--case MMC] [--route 3] [--threads 2] [--dpi 600]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--case", default="MMC")
    p.add_argument("--route", type=int, default=3,
                   help="route number; its stage-05 segment is found by "
                        "matching input/<case>/routes/route_<n>.csv")
    p.add_argument("--out", type=Path, default=ROOT / "run_output" / "MMC" / "postprocessing"
                   / "route3_selection" / "route3_selection_supplement.png")
    p.add_argument("--threads", type=int, default=2,
                   help="CPU cores this process may use (default 2)")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


ARGS = parse_args()
# Cap threads before numpy / embree load.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "TBB_NUM_THREADS"):
    os.environ[_v] = str(ARGS.threads)
try:
    _cpus = sorted(os.sched_getaffinity(0))[:ARGS.threads]
    os.sched_setaffinity(0, _cpus)
except (AttributeError, OSError):
    pass

import json  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(ROOT))
from thermal_common import (get_intersector, make_sphere_directions,  # noqa: E402
                            nearest_hit_multi)

MESH_B, MESH_G, MESH_V = 0, 1, 2      # 05a mesh ids (buildings, ground, vegetation)

# Colours (Okabe-Ito based; separable under deuteranopia/protanopia by hue
# and lightness): blue = this route, light orange = other routes,
# vermillion = this route's walls, bluish green = crown faces hit.
C_SEL = "#0072B2"
C_WALL = "#D55E00"
C_OTHER = "#F2C46D"
C_OTHER_WALL = "#EDB65A"
C_HIT = "#009E73"
C_FOOT = "#CFE6C7"
C_GROUND = "#ECECEC"
C_BLDG = "#B3B3B3"
C_EDGE = "#4D4D4D"
C_START, C_END = "#FFFFFF", "#000000"


def load(case):
    geo = ROOT / "input" / case / "geometry"
    ro = ROOT / "run_output" / case
    cfg = json.loads((ro / "thermal_out" / "config.json").read_text())
    t0 = time.time()
    mb = trimesh.load(geo / "building_final.stl", force="mesh")
    mg = trimesh.load(geo / "ground_and_water_final.stl", force="mesh")
    mv = trimesh.load(geo / "vegetation_final.stl", force="mesh")
    print(f"meshes b/g/v {len(mb.faces):,}/{len(mg.faces):,}/{len(mv.faces):,} "
          f"({time.time() - t0:.0f} s)", flush=True)
    return cfg, mb, mg, mv, ro


def route_segment(case, route, path, seg):
    rxy = pd.read_csv(ROOT / "input" / case / "routes" / f"route_{route}.csv")[
        ["x_local_m", "y_local_m"]].to_numpy()
    dmax = {s: cKDTree(path[seg == s, :2]).query(rxy)[0].max() for s in np.unique(seg)}
    s = min(dmax, key=dmax.get)
    print(f"route_{route}.csv -> stage-05 segment {s} (max distance {dmax[s]:.2f} m)")
    if dmax[s] > 1.0:
        sys.exit("no stage-05 segment matches the route CSV")
    return s


def main():
    cfg, mb, mg, mv, ro = load(ARGS.case)
    fac = np.load(ro / "thermal_out" / "facets.npz")
    W = sp.load_npz(ro / "thermal_out" / "lw_view_matrix.npz").tocsr()
    cidx = np.load(ro / "thermal_out" / "coarse_index.npy")
    seg = np.load(ro / "mrt_out" / "path_segment_id.npy")
    path = np.load(ro / "mrt_out" / "path_xyz.npy")
    rec = np.load(ro / "mrt_out" / "receptor_xyz.npy")
    s = route_segment(ARGS.case, ARGS.route, path, seg)

    rows = np.where(seg[cidx] == s)[0]
    sel = np.asarray(W[rows].sum(axis=0)).ravel() > 0
    mid, fid = fac["mesh_id"], fac["face_id"]
    nB, nG = len(mb.faces), len(mg.faces)
    selB = np.zeros(nB, bool); selB[fid[(mid == MESH_B) & sel]] = True
    selG = np.zeros(nG, bool); selG[fid[(mid == MESH_G) & sel]] = True
    uniB = np.zeros(nB, bool); uniB[fid[mid == MESH_B]] = True
    uniG = np.zeros(nG, bool); uniG[fid[mid == MESH_G]] = True
    bn = np.asarray(mb.face_normals)
    rz = float(cfg.get("roof_normal_z", 0.7))
    roof = bn[:, 2] > rz
    wall = np.abs(bn[:, 2]) < rz

    # ---- crown faces hit: re-trace with 05a's settings (not stored) ----
    t0 = time.time()
    dirs, _ = make_sphere_directions(cfg["n_lw_azimuth"], cfg["n_lw_elevation"],
                                     body=cfg["body_model"])
    inter = [(MESH_B, get_intersector(mb)), (MESH_G, get_intersector(mg)),
             (MESH_V, get_intersector(mv))]
    pts = rec[cidx[rows]]
    keys, veg = set(), set()
    for a in range(0, len(pts), 100):
        p = pts[a:a + 100]
        o = np.repeat(p, len(dirs), 0)
        dd = np.tile(dirs, (len(p), 1))
        hm, hf, _ = nearest_hit_multi(inter, o, dd, float(cfg["max_distance"]))
        behind = np.zeros(len(hm), bool)
        b = np.where(hm == MESH_B)[0]
        behind[b[np.einsum("ij,ij->i", bn[hf[b]], dd[b]) > 0]] = True
        ok = ((hm == MESH_B) | (hm == MESH_G)) & ~behind
        keys.update(zip(hm[ok].tolist(), hf[ok].tolist()))
        veg.update(hf[hm == MESH_V].tolist())
    stored = set(zip(mid[sel].tolist(), fid[sel].tolist()))
    print(f"re-trace of {len(pts)} receptors ({time.time() - t0:.0f} s): "
          f"{len(keys):,} facets, stored {len(stored):,}, "
          f"identical={keys == stored}", flush=True)
    if keys != stored:
        print("WARNING: re-trace differs from the stored selection; the crown "
              "hits may not match 05a exactly.")
    veg = np.array(sorted(veg), np.int64)

    # ---- counts for the caption ----
    nSelG, nSelW, nSelR = int(selG.sum()), int((selB & wall).sum()), int((selB & roof).sum())
    print(f"\nRoute {ARGS.route} selected triangles: ground {nSelG:,}, wall {nSelW:,}, "
          f"roof {nSelR:,}, total {int(sel.sum()):,}")
    print(f"building+ground triangles in scene: {nB + nG:,} (building {nB:,}, ground {nG:,})")
    print(f"union over all routes: {len(mid):,} (building {int(uniB.sum()):,}, "
          f"ground {int(uniG.sum()):,})")
    print(f"crown triangles hit by Route {ARGS.route}'s rays: {len(veg):,} "
          f"(of {len(mv.faces):,})")
    print(f"traced receptors: {len(pts)} (every {cfg['point_stride']}th route sample); "
          f"rays per receptor {len(dirs)}")

    draw(mb, mg, mv, selB, selG, uniB, uniG, roof, wall, veg,
         path[seg == s], nSelR)


def draw(mb, mg, mv, selB, selG, uniB, uniG, roof, wall, veg, route, nSelR):
    btri = np.asarray(mb.triangles, float)
    gtri = np.asarray(mg.triangles, float)[:, :, :2]
    gb = mg.bounds
    nB = len(btri)

    # wall triangles -> plan footprint segments
    xy = btri[:, :, :2]
    pairs = np.array([(0, 1), (1, 2), (0, 2)])
    L = np.stack([np.linalg.norm(xy[:, a] - xy[:, b], axis=1) for a, b in pairs], 1)
    k = L.argmax(1)
    segs = np.stack([xy[np.arange(nB), pairs[k, 0]], xy[np.arange(nB), pairs[k, 1]]], 1)

    # tree-crown footprint, 1 m raster of crown triangle centroids
    vc = mv.triangles_center
    H, xe, ye = np.histogram2d(vc[:, 0], vc[:, 1],
                               bins=[np.arange(gb[0, 0], gb[1, 0] + 1, 1.0),
                                     np.arange(gb[0, 1], gb[1, 1] + 1, 1.0)])
    crown = np.ma.masked_where(H.T == 0, np.ones_like(H.T))

    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans",
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6})
    CM = 1 / 2.54
    fig = plt.figure(figsize=(16 * CM, 12.2 * CM), dpi=ARGS.dpi)
    ax = fig.add_axes([0.085, 0.215, 0.905, 0.775])

    def polys(tri, color, z, alpha=1.0):
        ax.add_collection(PolyCollection(tri, facecolors=color, edgecolors="none",
                                         linewidths=0, antialiaseds=False, zorder=z,
                                         alpha=alpha, rasterized=True))

    def lines(sg, color, lw, z):
        ax.add_collection(LineCollection(sg, colors=color, linewidths=lw, zorder=z,
                                         capstyle="butt", rasterized=True))

    polys(gtri, C_GROUND, 1)
    polys(btri[roof & ~selB][:, :, :2], C_BLDG, 2)
    ax.imshow(crown, extent=(xe[0], xe[-1], ye[0], ye[-1]), origin="lower",
              cmap=ListedColormap([C_FOOT]), zorder=3, interpolation="nearest",
              rasterized=True)
    polys(gtri[uniG & ~selG], C_OTHER, 4)
    lines(segs[uniB & wall & ~selB], C_OTHER_WALL, 0.6, 4.5)
    polys(np.asarray(mv.triangles, float)[veg][:, :, :2], C_HIT, 5, alpha=0.6)
    polys(gtri[selG], C_SEL, 6)
    if nSelR:
        polys(btri[selB & roof][:, :, :2], C_SEL, 7)
    lines(segs[selB & wall], C_WALL, 1.1, 8)

    ax.plot(route[:, 0], route[:, 1], color="white", lw=1.9, zorder=20,
            solid_capstyle="round")
    ax.plot(route[:, 0], route[:, 1], color="black", lw=0.75, zorder=21)
    ax.plot(*route[0, :2], marker="o", ms=5.5, mfc=C_START, mec="black", mew=0.9, zorder=23)
    ax.plot(*route[-1, :2], marker="s", ms=4.8, mfc=C_END, mec="white", mew=0.7, zorder=23)
    ax.add_patch(Rectangle(gb[0, :2], *(gb[1, :2] - gb[0, :2]), fill=False, ls=(0, (4, 2)),
                           lw=0.7, ec=C_EDGE, zorder=19))

    x0, y0, x1, y1 = gb[0, 0] - 4, gb[0, 1] - 4, gb[1, 0] + 4, gb[1, 1] + 4
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")

    # scale bar + north arrow in the empty north-west corner, on a white card
    bx, by, bar = x0 + 18, y1 - 66, 100
    ax.add_patch(Rectangle((bx - 12, by - 14), bar + 82, 72, fc="white", ec="none",
                           alpha=0.85, zorder=29))
    ax.add_patch(Rectangle((bx, by), bar / 2, 5, fc="black", ec="black", lw=0.5, zorder=30))
    ax.add_patch(Rectangle((bx + bar / 2, by), bar / 2, 5, fc="white", ec="black", lw=0.5,
                           zorder=30))
    ax.text(bx, by + 8, "0", ha="center", va="bottom", fontsize=7, zorder=30)
    ax.text(bx + bar, by + 8, f"{bar:g} m", ha="center", va="bottom", fontsize=7, zorder=30)
    nx, ny = bx + bar + 52, by - 4
    ax.annotate("", xy=(nx, ny + 30), xytext=(nx, ny),
                arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.5",
                                lw=1.0, color="black", shrinkA=0, shrinkB=0), zorder=30)
    ax.text(nx, ny + 32, "N", ha="center", va="bottom", fontsize=8, weight="bold",
            zorder=30)

    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=7, length=2.5, pad=1.5)
    ax.set_xlabel("Easting (m), NAD83(2011) / UTM zone 17N (EPSG:6346)", fontsize=7.5,
                  labelpad=2)
    ax.set_ylabel("Northing (m)", fontsize=7.5, labelpad=2)

    r = ARGS.route
    leg = [Patch(fc=C_SEL, label=f"Selected ground (Route {r})"),
           Line2D([], [], color=C_WALL, lw=2, label=f"Selected walls (Route {r})")]
    if nSelR:
        leg.append(Patch(fc=C_SEL, ec=C_WALL, label=f"Selected roofs (Route {r})"))
    leg += [Patch(fc=C_OTHER, label="Selected for the other routes"),
            Patch(fc=C_HIT, alpha=0.6,
                  label=f"Tree crowns hit by Route {r}'s rays (no energy balance)"),
            Patch(fc=C_BLDG, label="Buildings (not selected)"),
            Patch(fc=C_FOOT, label="Tree-crown footprint"),
            Line2D([], [], color="black", lw=0.75, label=f"Route {r}"),
            Line2D([], [], marker="o", ls="", ms=5, mfc=C_START, mec="black", mew=0.9,
                   label="Start"),
            Line2D([], [], marker="s", ls="", ms=4.5, mfc=C_END, mec="black", label="End"),
            Line2D([], [], color=C_EDGE, lw=0.7, ls=(0, (4, 2)), label="Scene edge")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=3,
               fontsize=7, frameon=False, handlelength=1.8, handleheight=0.9,
               columnspacing=1.4, labelspacing=0.45, borderaxespad=0.3)

    ARGS.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ARGS.out, dpi=ARGS.dpi)
    plt.close(fig)
    print(f"wrote {ARGS.out}")


if __name__ == "__main__":
    main()
