"""
extract_osm_pedestrian_network.py -- pull the walkable pedestrian network
from OpenStreetMap for the exact footprint of your LIDAR tile, reproject
it into the same coordinate system as your point cloud / STL geometry,
and export it as simple polylines ready to feed into the MRT/shading
ray-tracing pipeline's sample_polyline() function.

IMPORTANT -- run this on a machine with normal internet access. It needs
to reach the Overpass API (overpass-api.de) to query OpenStreetMap; it
will NOT work from a sandboxed/restricted network.

The bounding box below was derived directly from your LIDAR tile index
(fl2021_miami_dade_J1413971_tileindex.shp), converted from its native
NAD83(2011) UTM Zone 17N CRS to lat/lon:
    UTM17N bounds:  X 562027-563203, Y 2848416-2849229
    Lat/lon bounds: lat 25.752457-25.759748, lon -80.381509--80.369745

Run:
    python3 extract_osm_pedestrian_network.py --output-dir osm_paths/

Output:
    osm_paths/pedestrian_network.graphml   -- full routable graph (for
                                               future shortest-path work)
    osm_paths/pedestrian_edges.geojson     -- edge geometries, UTM17N meters
    osm_paths/path_polylines.npz           -- flattened polylines as a list
                                               of Nx2 numpy arrays (UTM17N
                                               meters), ready for
                                               sample_polyline()
    osm_paths/network_preview.png          -- quick visual sanity check
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt

# Bounding box derived from your LIDAR tile index (see docstring above).
# (left, bottom, right, top) in lat/lon, EPSG:4326 -- required format for
# osmnx 2.x's graph_from_bbox.
DEFAULT_BBOX_LATLON = (-80.381509, 25.752457, -80.369745, 25.759748)

# Your point-cloud/STL data's coordinate system (NAD83(2011) UTM Zone 17N).
TARGET_CRS = "EPSG:6346"


def parse_args():
    p = argparse.ArgumentParser(description="Extract FIU MMC pedestrian network from OSM")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("LEFT_LON", "BOTTOM_LAT", "RIGHT_LON", "TOP_LAT"),
                    help="Override the default bbox (left, bottom, right, top in lat/lon). "
                         "Default: derived from your LIDAR tile index.")
    p.add_argument("--network-type", default="walk",
                    choices=["walk", "all", "all_public"],
                    help="osmnx network type. 'walk' includes footway/path/pedestrian/"
                         "steps/residential-with-sidewalk-access etc. (default: walk)")
    p.add_argument("--local-origin-x", type=float, default=0.0,
                    help="If your STL/point-cloud data was RECENTERED to a local origin "
                         "(rather than kept in raw UTM meters), subtract that origin's "
                         "UTM X here so the path network lines up. Default 0.0 = assumes "
                         "your data is still in raw UTM17N meters.")
    p.add_argument("--local-origin-y", type=float, default=0.0,
                    help="Same as --local-origin-x but for the Y (northing) coordinate.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = tuple(args.bbox) if args.bbox else DEFAULT_BBOX_LATLON
    print(f"[osm] Querying Overpass API for bbox {bbox} (network_type={args.network_type}) ...")
    print("[osm] This requires internet access to overpass-api.de -- if this hangs or "
          "fails, check your network connection, not this script's logic.")

    G = ox.graph_from_bbox(bbox=bbox, network_type=args.network_type)
    print(f"[osm] Retrieved graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    print(f"[osm] Reprojecting to {TARGET_CRS} (matching your LIDAR data's CRS) ...")
    G_proj = ox.project_graph(G, to_crs=TARGET_CRS)

    graphml_path = out_dir / "pedestrian_network.graphml"
    ox.save_graphml(G_proj, str(graphml_path))
    print(f"[osm] Saved routable graph: {graphml_path}")

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)

    geojson_path = out_dir / "pedestrian_edges.geojson"
    edges_gdf.to_file(str(geojson_path), driver="GeoJSON")
    print(f"[osm] Saved edge geometries: {geojson_path}")

    # Flatten to simple polylines in your LOCAL coordinate frame (origin-shifted
    # if you provided --local-origin-x/y), ready for sample_polyline().
    origin = np.array([args.local_origin_x, args.local_origin_y])
    polylines = []
    highway_tags = []
    for _, row in edges_gdf.iterrows():
        geom = row.geometry
        coords = np.array(geom.coords) - origin  # drop Z if present, apply origin shift
        coords = coords[:, :2]
        polylines.append(coords)
        highway_tags.append(row.get("highway", "unknown"))

    npz_path = out_dir / "path_polylines.npz"
    # np.savez can't directly store a ragged list of arrays -- use pickle via
    # allow_pickle for this variable-length case.
    with open(out_dir / "path_polylines.pkl", "wb") as f:
        pickle.dump({"polylines": polylines, "highway_tags": highway_tags}, f)
    print(f"[osm] Saved {len(polylines)} polylines (local coords, origin={tuple(origin)}): "
          f"{out_dir / 'path_polylines.pkl'}")

    # Quick visual sanity check
    fig, ax = plt.subplots(figsize=(10, 10))
    for coords in polylines:
        ax.plot(coords[:, 0], coords[:, 1], "-", color="steelblue", linewidth=1)
    ax.set_aspect("equal")
    ax.set_xlabel("Local X [m]")
    ax.set_ylabel("Local Y [m]")
    ax.set_title(f"FIU MMC pedestrian network ({len(polylines)} segments)\n"
                 f"CRS: {TARGET_CRS}, origin shift: {tuple(origin)}")
    fig.tight_layout()
    preview_path = out_dir / "network_preview.png"
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    print(f"[osm] Saved preview plot: {preview_path}")

    print("\n[osm] highway tag breakdown:")
    from collections import Counter
    # Some OSM ways have a list-valued highway tag (e.g. ['footway','steps']) --
    # not hashable as-is, so normalize to a tuple (or the tag itself if already
    # a plain string) before counting.
    normalized_tags = [tuple(t) if isinstance(t, list) else t for t in highway_tags]
    for tag, count in Counter(normalized_tags).most_common():
        print(f"  {tag}: {count}")

    print(f"\n[osm_result] n_nodes={len(G.nodes)} n_edges={len(G.edges)} "
          f"n_polylines={len(polylines)} output_dir={out_dir}")


if __name__ == "__main__":
    main()
