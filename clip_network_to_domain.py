#!/usr/bin/env python3
"""clip_network_to_domain.py -- drop pedestrian-network edges that leave the
reconstructed scene, so generated routes stay where geometry exists.

The FIU campus scene is a single LiDAR tile. One OSM footway (456 m) swung up
to 88 m south of the tile before rejoining the campus, and the edge-disjoint
route generator used it for Route 1, which then crossed 349 m with no ground,
buildings or trees beneath or around it. This script removes every edge whose
geometry is not covered by the domain rectangle (widened by --tolerance-m, so
a path that grazes the tile edge by a fraction of a metre is kept) and writes
a new GraphML; generate_route.py is then run on that file.

    python3 clip_network_to_domain.py \
        --graphml run_output/MMC/osm_paths/pedestrian_network.graphml \
        --output run_output/MMC/osm_paths/pedestrian_network_domain.graphml \
        --bounds 562027.7 2848417.0 563202.1 2849228.8 --tolerance-m 1.0
    python3 generate_route.py --graphml <output> --output-dir input/MMC/routes \
        --n-routes 3 --overwrite
    python3 generate_route.py --graphml <output> --output-dir input/MMC/routes \
        --import-geojson verification_route_refactor/fourth_route.geojson
"""
from __future__ import annotations

import argparse

import osmnx as ox
from shapely.geometry import LineString, box


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graphml", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bounds", type=float, nargs=4, required=True,
                    metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="domain rectangle in the graph's projected CRS")
    ap.add_argument("--tolerance-m", type=float, default=1.0)
    args = ap.parse_args()

    x0, y0, x1, y1 = args.bounds
    t = args.tolerance_m
    domain = box(x0 - t, y0 - t, x1 + t, y1 + t)
    graph = ox.load_graphml(args.graphml)
    outside = []
    for u, v, k, data in graph.edges(keys=True, data=True):
        geom = data.get("geometry") or LineString(
            [(graph.nodes[u]["x"], graph.nodes[u]["y"]),
             (graph.nodes[v]["x"], graph.nodes[v]["y"])])
        if not domain.covers(geom):
            outside.append((u, v, k))
            print(f"  removing {u}->{v}: {geom.length:.0f} m, "
                  f"{geom.difference(domain).length:.0f} m outside")
    graph.remove_edges_from(outside)
    ox.save_graphml(graph, args.output)
    print(f"removed {len(outside)} directed edge(s); wrote {args.output}")


if __name__ == "__main__":
    main()
