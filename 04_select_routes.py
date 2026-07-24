"""
04_select_routes.py -- pick the candidate start->end routes ONCE, up front,
so the expensive MRT ray tracing runs along only those routes instead of the
entire pedestrian network.

Previously the pipeline ray-traced MRT over EVERY edge of the OSM pedestrian
network (05/05a/05b/05*), and only at the very last stage (08/09) did it pick
a handful of edge-disjoint start->end routes to actually compare. That meant
almost all of the radiation computation was spent on network points no route
ever used. This stage flips the order: choose the routes first (a cheap,
graph-only operation), then hand ONLY those route polylines to the MRT stage.

Inputs:
    --graphml pedestrian_network.graphml   (from extract_osm_pedestrian_network.py)

Outputs (into --output-dir):
    route_polylines.pkl   -- the selected routes as polylines, in the exact
                             format 05_mrt_network_raytrace.py consumes, so MRT
                             is computed ONLY along these routes.
    selected_routes.pkl   -- the same routes plus start/end/connectivity
                             metadata, loaded back by stages 08 and 09 so they
                             score exactly the routes MRT was computed for.

Run:
    python3 04_select_routes.py \
        --graphml osm_paths/pedestrian_network.graphml \
        --output-dir osm_paths/ \
        --n-routes 3 \
        --start-latlon 25.7585 -80.3760 --end-latlon 25.7532 -80.3735
"""

import argparse
from pathlib import Path

import osmnx as ox

from route_selection import select_routes, save_route_polylines, save_selected_routes


def parse_args():
    p = argparse.ArgumentParser(description="Select start->end routes before MRT")
    p.add_argument("--graphml", required=True, help="pedestrian_network.graphml")
    p.add_argument("--output-dir", required=True,
                    help="Where to write route_polylines.pkl and selected_routes.pkl")

    p.add_argument("--n-routes", type=int, default=3,
                    help="Number of edge-disjoint start->end routes to compare (default: 3)")
    p.add_argument("--route-sample-spacing-m", type=float, default=1.0,
                    help="Vertex spacing along each reconstructed route, meters. The MRT "
                         "stage re-samples these polylines at its own finer ds-path, so "
                         "this only sets the stored route resolution (default: 1.0)")

    # Explicit endpoints (optional). Provide BOTH start and end to pin them;
    # otherwise the two opposite bbox corners are used automatically.
    p.add_argument("--start-latlon", nargs=2, type=float, default=None,
                    metavar=("LAT", "LON"),
                    help="Route START as latitude longitude (overrides auto corner).")
    p.add_argument("--end-latlon", nargs=2, type=float, default=None,
                    metavar=("LAT", "LON"),
                    help="Route END as latitude longitude (overrides auto corner).")
    p.add_argument("--start-xy", nargs=2, type=float, default=None,
                    metavar=("X", "Y"),
                    help="Route START in the LOCAL (origin-shifted) frame, meters.")
    p.add_argument("--end-xy", nargs=2, type=float, default=None,
                    metavar=("X", "Y"),
                    help="Route END in the LOCAL frame, meters.")

    p.add_argument("--local-origin-x", type=float, default=0.0,
                    help="Origin-shift X applied when the network was built (must match "
                         "extract_osm_pedestrian_network.py so lat/lon endpoints line up).")
    p.add_argument("--local-origin-y", type=float, default=0.0,
                    help="Origin-shift Y (see --local-origin-x).")
    p.add_argument("--project-crs", default="EPSG:6346",
                    help="Projected CRS of the network/local frame "
                         "(default EPSG:6346 = NAD83(2011) UTM 17N, Miami).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[routes] Loading routable network...")
    G_multi = ox.load_graphml(args.graphml)

    print(f"[routes] Selecting up to {args.n_routes} edge-disjoint routes...")
    selection = select_routes(
        G_multi, n_routes=args.n_routes,
        ds=args.route_sample_spacing_m,
        project_crs=args.project_crs,
        origin=(args.local_origin_x, args.local_origin_y),
        start_latlon=args.start_latlon, end_latlon=args.end_latlon,
        start_xy=args.start_xy, end_xy=args.end_xy,
    )

    poly_path = out_dir / "route_polylines.pkl"
    routes_path = out_dir / "selected_routes.pkl"
    save_route_polylines(poly_path, selection["routes"])
    save_selected_routes(routes_path, selection)

    total_pts = sum(len(r["xy"]) for r in selection["routes"])
    print(f"[routes] Wrote {poly_path} ({len(selection['routes'])} routes, "
          f"{total_pts} vertices at {args.route_sample_spacing_m} m spacing)")
    print(f"[routes] Wrote {routes_path} (routes + endpoint metadata)")
    print(f"[routes_result] n_routes={len(selection['routes'])} "
          f"start={selection['start_node']} end={selection['end_node']} "
          f"connectivity={selection['connectivity']} output_dir={out_dir}")


if __name__ == "__main__":
    main()
