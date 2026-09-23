#!/usr/bin/env python3
"""Study-area map of the six Lisbon validation routes (paper figure).

Panel (a): overview of the Lisbon municipality with the Tagus estuary and the
six route locations. Panels (b)-(g): one panel per route with OSM building
footprints, the daytime measurement walk (solid) and the night-time walk
(dashed), start marker and a 100 m scale bar.

Inputs (read only):
  input/lisbonN/routes/route_1.csv  daytime track actually simulated
  input/lisbonN/routes/route_2.csv  night-time track actually simulated
  input/lisbonN/osm/lisbon_complete_osm.gpkg  OSM buildings + query domain
  Municipality outline and Tagus coastline: fetched live from OSM with osmnx
  (cache disabled). Falls back to the convex extent of the six case domains
  (no coastline) when the network or osmnx is unavailable.

Route tracks: Silva et al., Zenodo doi:10.5281/zenodo.15516995 (CC BY 4.0).
Building footprints (c) OpenStreetMap contributors (ODbL).

Usage (from the project root):
  python3 paper_lisbon_map.py [--out-dir "/home/harshin/files/fastUTEC paper"]
Writes lisbon_study_area.png and lisbon_study_area.pdf; nothing else.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from shapely.geometry import Point, box
from shapely.ops import polygonize, unary_union

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
CRS = "EPSG:3763"
DEFAULT_OUT = Path("/home/harshin/files/fastUTEC paper")

ROUTE_NAMES = {
    1: "Belém",
    2: "Rato–Chiado",
    3: "Marquês de Pombal–Gulbenkian",
    4: "University rectory–Quinta das Conchas",
    5: "Roma/Areeiro–Anjos",
    6: "Parque das Nações",
}
# Panel titles wrapped to fit the narrow bottom-row panels.
TITLE_NAMES = {
    3: "Marquês de Pombal–\nGulbenkian",
    4: "University rectory–\nQuinta das Conchas",
    5: "Roma/Areeiro–\nAnjos",
    6: "Parque das\nNações",
}
# Okabe-Ito (colour-blind safe), yellow omitted.
COLORS = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73",
          4: "#D55E00", 5: "#0072B2", 6: "#CC79A7"}

WATER = "#dbe7f0"
LAND_OUT = "#f3f3f1"
LAND_IN = "#e4e4e0"
BUILDING = "#d0d0d0"
INK = "#222222"

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 7,
    "font.family": "DejaVu Sans", "figure.dpi": 150,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


# ----------------------------------------------------------------- data ---
def load_track(case: int, route: int) -> gpd.GeoSeries:
    df = pd.read_csv(ROOT / f"input/lisbon{case}/routes/route_{route}.csv")
    pts = gpd.GeoSeries(gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326").to_crs(CRS)
    return pts, float(df.cumdist_m.iloc[-1]), str(df.timestamp_local.iloc[0])


def load_buildings(case: int) -> tuple[gpd.GeoDataFrame, object]:
    f = ROOT / f"input/lisbon{case}/osm/lisbon_complete_osm.gpkg"
    g = gpd.read_file(f, layer="raw_complete_osm_features")
    cols = [c for c in ("building", "building:part") if c in g.columns]
    mask = np.zeros(len(g), bool)
    for c in cols:
        mask |= g[c].notna().to_numpy()
    g = g[mask & g.geom_type.isin(["Polygon", "MultiPolygon"]).to_numpy()]
    dom = gpd.read_file(f, layer="query_domain_projected").to_crs(CRS).geometry.iloc[0]
    return g.to_crs(CRS), dom


def fetch_context(domains):
    """Municipality outline + water polygon in CRS, or fallback."""
    try:
        import osmnx as ox
        ox.settings.use_cache = False
        ox.settings.log_console = False
        ox.settings.requests_timeout = 90
        muni = ox.geocode_to_gdf("Lisboa, Portugal").to_crs(CRS).geometry.iloc[0]
        # Window in WGS84 around the municipality for the coastline query.
        w, s, e, n = gpd.GeoSeries([muni], crs=CRS).to_crs(4326).total_bounds
        pad = 0.03
        bbox = (w - pad, s - pad, e + pad, n + pad)
        coast = ox.features_from_bbox(bbox, {"natural": "coastline"})
        win = gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(CRS).iloc[0]
        lines = [g.boundary if g.geom_type in ("Polygon", "MultiPolygon") else g
                 for g in coast.to_crs(CRS).geometry]
        merged = unary_union(lines + [win.boundary])
        pieces = list(polygonize(merged))
        # Seed points in the Tagus estuary (lon, lat) classify water pieces.
        seeds = gpd.GeoSeries([Point(-9.15, 38.675), Point(-9.10, 38.70),
                               Point(-9.07, 38.74)], crs=4326).to_crs(CRS)
        water = unary_union([p for p in pieces if any(p.contains(s) for s in seeds)])
        water = water.intersection(win)
        return muni, water, True
    except Exception as exc:  # network / osmnx failure -> fallback
        print(f"[paper_lisbon_map] OSM context unavailable ({exc}); using case extents.")
        hull = unary_union(domains).convex_hull.buffer(800)
        return hull, None, False


# ------------------------------------------------------------- drawing ---
def scalebar(ax, length_m, label, loc="lower left", lw=2.0, fs=7):
    """Scale bar with its label on a translucent white box."""
    x0, x1 = ax.get_xlim()
    bar_h = lw / 72 * (x1 - x0) / (ax.get_position().width * ax.figure.get_figwidth())
    sb = AnchoredSizeBar(ax.transData, length_m, label, loc=loc, pad=0.35,
                         borderpad=0.5, sep=2, frameon=True, color=INK,
                         size_vertical=bar_h, label_top=True,
                         fontproperties={"size": fs})
    sb.patch.set(facecolor="white", edgecolor="none", alpha=0.85)
    sb.set_zorder(20)
    ax.add_artist(sb)


def pick_corner(ax, pts):
    """Lower corner with fewer track points nearby (keeps the bar off the route)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = x1 - x0, y1 - y0
    low = pts.y.to_numpy() < y0 + 0.15 * dy
    left = (pts.x.to_numpy() < x0 + 0.35 * dx) & low
    right = (pts.x.to_numpy() > x1 - 0.35 * dx) & low
    return "lower left" if left.sum() < right.sum() else "lower right"


def north_arrow(ax, x=0.93, y=0.86, size=0.08):
    ax.annotate("", xy=(x, y + size), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.6",
                                color=INK, lw=1.0), zorder=20)
    ax.text(x, y + size + 0.01, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8, fontweight="bold", color=INK, zorder=20)


def fit_limits(bounds, box_w, box_h, pad_frac=0.06):
    """Limits covering bounds (+pad) with the aspect of the axes box."""
    x0, y0, x1, y1 = bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0), (y1 - y0)
    w, h = w * (1 + 2 * pad_frac), h * (1 + 2 * pad_frac)
    ar = box_w / box_h
    if w / h > ar:
        h = w / ar
    else:
        w = h * ar
    return (cx - w / 2, cx + w / 2), (cy - h / 2, cy + h / 2)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#666666")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    tracks, info, bld, doms = {}, {}, {}, {}
    for c in range(1, 7):
        day, lday, tday = load_track(c, 1)
        night, lnight, tnight = load_track(c, 2)
        tracks[c] = (day, night)
        info[c] = (lday, tday, lnight, tnight)
        bld[c], doms[c] = load_buildings(c)
    muni, water, have_osm = fetch_context(list(doms.values()))

    # ---- layout (inches) -------------------------------------------------
    FW, FH = 7.0, 6.55
    fig = plt.figure(figsize=(FW, FH))

    def add(x, y, w, h):  # inches from lower-left
        return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

    top_y, top_h = 3.02, 3.30
    ax_a = add(0.02, top_y, 3.30, top_h)
    right_x = 3.45
    right_w = FW - right_x - 0.02
    boxes = {
        1: (right_x, top_y + 1.93, right_w, 1.17),
        2: (right_x, top_y, 1.70, 1.66),
    }
    bw, bh, gap = 1.685, 2.80, 0.087
    for k, c in enumerate((3, 4, 5, 6)):
        boxes[c] = (0.02 + k * (bw + gap), 0.02, bw, bh)

    # ---- panel (a) ---------------------------------------------------------
    allpts = unary_union([tracks[c][0].union_all() for c in tracks])
    if have_osm:
        mb = gpd.GeoSeries([muni]).total_bounds
        ext = (mb[0], mb[1] - 1500, mb[2], mb[3])
    else:
        ext = muni.bounds
    xl, yl = fit_limits(ext, 3.30, top_h, pad_frac=0.03)
    ax_a.set_facecolor(LAND_OUT if have_osm else "white")
    if water is not None and not water.is_empty:
        gpd.GeoSeries([water]).plot(ax=ax_a, color=WATER, lw=0, zorder=1)
    gpd.GeoSeries([muni]).plot(ax=ax_a, facecolor=LAND_IN if have_osm else "none",
                               edgecolor="#555555", lw=0.8, zorder=2)
    if have_osm and water is not None:
        # repaint water over the municipal polygon where it extends offshore
        gpd.GeoSeries([water.intersection(muni)]).plot(ax=ax_a, color=WATER,
                                                       lw=0, zorder=2.5)
        gpd.GeoSeries([muni.boundary]).plot(ax=ax_a, color="#555555", lw=0.8,
                                            zorder=3)
    for c in range(1, 7):
        day = tracks[c][0]
        ax_a.plot(day.x, day.y, color=COLORS[c], lw=1.4, zorder=5,
                  solid_capstyle="round")
    # Numbered labels offset from each route so the tracks stay visible.
    offsets = {1: (0, 900), 2: (-900, 500), 3: (-1000, 0), 4: (-900, 300),
               5: (900, 0), 6: (-900, 0)}
    for c in range(1, 7):
        day = tracks[c][0]
        cx, cy = float(day.x.mean()), float(day.y.mean())
        ox_, oy_ = offsets[c]
        ax_a.annotate(str(c), xy=(cx, cy), xytext=(cx + ox_, cy + oy_),
                      ha="center", va="center", fontsize=7.5, fontweight="bold",
                      color="white", zorder=7,
                      bbox=dict(boxstyle="circle,pad=0.25", fc=COLORS[c],
                                ec=INK, lw=0.5),
                      arrowprops=dict(arrowstyle="-", color=INK, lw=0.5,
                                      shrinkA=0, shrinkB=2))
    ax_a.set_xlim(*xl)
    ax_a.set_ylim(*yl)
    ax_a.set_aspect("equal")
    clean(ax_a)
    scalebar(ax_a, 2000, "2 km", loc="lower right")
    north_arrow(ax_a, x=0.07, y=0.84)
    if have_osm:
        ax_a.text(0.42, 0.07, "Tagus estuary", transform=ax_a.transAxes,
                  fontsize=7, style="italic", color="#4a6b85", ha="center")
        ax_a.text(0.60, 0.60, "Lisbon", transform=ax_a.transAxes,
                  fontsize=8, color="#555555", ha="center")
    ax_a.set_title("(a)  Lisbon municipality", loc="left", pad=3)

    # ---- panels (b)-(g) ---------------------------------------------------
    letters = "bcdefg"
    for c in range(1, 7):
        x, y, w, h = boxes[c]
        th = 0.17 if c <= 2 else 0.33  # title room (two lines, bottom row)
        ax = add(x, y, w, h - th)
        day, night = tracks[c]
        b = day.union_all().union(night.union_all()).bounds
        xl, yl = fit_limits(b, w, h - th, pad_frac=0.05)
        ax.set_facecolor("white")
        clip = box(xl[0], yl[0], xl[1], yl[1])
        bb = bld[c].clip(clip)
        bb.plot(ax=ax, color=BUILDING, lw=0, zorder=1)
        ax.plot(day.x, day.y, color=COLORS[c], lw=1.8, zorder=4,
                solid_capstyle="round", solid_joinstyle="round")
        ax.plot(night.x, night.y, color=INK, lw=0.7, ls=(0, (2.5, 2)),
                zorder=5)
        ax.plot(day.x.iloc[0], day.y.iloc[0], marker="o", ms=5,
                mfc="white", mec=INK, mew=1.0, zorder=6, ls="none")
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        clean(ax)
        both = pd.concat([day, night])
        scalebar(ax, 100, "100 m", loc=pick_corner(ax, both), lw=1.8)
        ax.set_title(f"({letters[c-1]})  {c}  {TITLE_NAMES.get(c, ROUTE_NAMES[c])}",
                     loc="left", pad=3)

    # ---- legend in the free slot right of (c) -----------------------------
    lx = boxes[2][0] + boxes[2][2] + 0.1
    ax_l = add(lx, top_y, FW - lx - 0.02, 1.66)
    ax_l.axis("off")
    handles = [
        Line2D([], [], color="#777777", lw=1.8, label="Daytime walk\n(colour = route)"),
        Line2D([], [], color=INK, lw=0.7, ls=(0, (2.5, 2)), label="Night-time walk"),
        Line2D([], [], marker="o", ms=5, mfc="white", mec=INK, mew=1.0,
               ls="none", label="Start (daytime)"),
        plt.Rectangle((0, 0), 1, 1, fc=BUILDING, ec="none", label="Building (OSM)"),
    ]
    ax_l.legend(handles=handles, loc="center left", frameon=False,
                handlelength=2.2, labelspacing=0.9, borderaxespad=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext_ in ("png", "pdf"):
        fig.savefig(args.out_dir / f"lisbon_study_area.{ext_}", dpi=300)
    plt.close(fig)

    for c in range(1, 7):
        lday, tday, lnight, tnight = info[c]
        print(f"route {c} {ROUTE_NAMES[c]:40s} day {lday:6.0f} m ({tday[:16]})  "
              f"night {lnight:6.0f} m ({tnight[:16]})")
    print("wrote", args.out_dir / "lisbon_study_area.png")


if __name__ == "__main__":
    main()
