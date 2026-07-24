"""
route_selection.py -- shared, MRT-independent selection of edge-disjoint
corner-to-corner routes through the pedestrian network.

This logic used to be duplicated inside 08_route_thermal_stress.py and
09_route_thermal_stress_jos3.py (and they had drifted -- 09 lacked the
explicit lat/lon endpoint support that 08 had). It now lives here so that:

  * a dedicated up-front stage (04_select_routes.py) can pick the routes
    ONCE, right after the OSM network is built, and hand ONLY those route
    polylines to the MRT ray tracer -- so MRT is computed along the handful
    of candidate routes instead of the entire pedestrian network (the whole
    point of this refactor: a large speedup), and
  * stages 08 and 09 load those SAME routes back, guaranteeing the routes
    they score are exactly the ones MRT was computed for.

Route finding is graph-only (network geometry + connectivity); it does not
depend on any radiation result, which is precisely why it can run first.
"""

import pickle
from pathlib import Path

import numpy as np
import networkx as nx


# ============================================================
# Route finding (graph-only)
# ============================================================
def find_best_corner_pair(G_simple, pos, corner_a_xy, corner_b_xy, k_needed,
                          search_n=20, check_top=8):
    """Pick a start/end node pair near two opposite corners whose graph
    edge-connectivity can actually support k_needed edge-disjoint routes."""
    cand_a = sorted(G_simple.nodes(), key=lambda n: (pos[n][0]-corner_a_xy[0])**2 + (pos[n][1]-corner_a_xy[1])**2)[:search_n]
    cand_b = sorted(G_simple.nodes(), key=lambda n: (pos[n][0]-corner_b_xy[0])**2 + (pos[n][1]-corner_b_xy[1])**2)[:search_n]
    best = None
    for a in cand_a[:check_top]:
        for b in cand_b[:check_top]:
            if a == b:
                continue
            try:
                conn = nx.edge_connectivity(G_simple, a, b)
            except nx.NetworkXError:
                continue
            score = (conn >= k_needed, conn)
            if best is None or score > best[0]:
                best = (score, a, b, conn)
    if best is None:
        raise RuntimeError("Could not find any valid corner pair in this graph.")
    return best[1], best[2], best[3]


def reconstruct_route_xy(G_multi, node_path, ds=1.0):
    """Walk the actual edge geometries (not straight node-to-node lines) to
    get a densely-sampled (x,y) polyline for the route."""
    all_xy = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        edge_data = G_multi.get_edge_data(u, v)
        if edge_data is None:
            edge_data = G_multi.get_edge_data(v, u)
        data = list(edge_data.values())[0]
        if "geometry" in data:
            coords = np.array(data["geometry"].coords)
        else:
            coords = np.array([[G_multi.nodes[u]["x"], G_multi.nodes[u]["y"]],
                                [G_multi.nodes[v]["x"], G_multi.nodes[v]["y"]]])
        # ensure orientation matches u -> v
        if np.linalg.norm(coords[0] - [G_multi.nodes[u]["x"], G_multi.nodes[u]["y"]]) > \
           np.linalg.norm(coords[-1] - [G_multi.nodes[u]["x"], G_multi.nodes[u]["y"]]):
            coords = coords[::-1]
        all_xy.append(coords)
    full = np.vstack(all_xy)
    # densify to ~ds spacing
    seg_lens = np.linalg.norm(np.diff(full, axis=0), axis=1)
    cumlen = np.concatenate(([0], np.cumsum(seg_lens)))
    total = cumlen[-1]
    svals = np.arange(0, total, ds)
    dense = np.empty((len(svals), 2))
    j = 0
    for i, s in enumerate(svals):
        while j < len(seg_lens) - 1 and s > cumlen[j + 1]:
            j += 1
        frac = 0 if seg_lens[j] == 0 else (s - cumlen[j]) / seg_lens[j]
        dense[i] = full[j] + frac * (full[j + 1] - full[j])
    return dense, total


def _nearest_node(pos, pt_local):
    node_ids = list(pos.keys())
    P = np.array([pos[n] for n in node_ids])
    d = np.hypot(P[:, 0] - pt_local[0], P[:, 1] - pt_local[1])
    return node_ids[int(np.argmin(d))]


def resolve_endpoints(pos, n_routes, project_crs, origin,
                      start_latlon=None, end_latlon=None,
                      start_xy=None, end_xy=None):
    """Turn optional lat/lon or local-xy endpoint requests into graph nodes.

    Returns (start_node, end_node, connectivity_or_None). When both endpoints
    are pinned we resolve them to the nearest nodes; otherwise the caller
    should fall back to automatic corner selection.
    """
    G_nodes = list(pos.keys())

    def latlon_to_local(lat, lon):
        from pyproj import Transformer
        tf = Transformer.from_crs("EPSG:4326", project_crs, always_xy=True)
        X, Y = tf.transform(lon, lat)          # always_xy -> (lon, lat)
        return np.array([X, Y]) - np.asarray(origin)

    start_pt = end_pt = None
    if start_latlon is not None:
        start_pt = latlon_to_local(*start_latlon)
    elif start_xy is not None:
        start_pt = np.array(start_xy, dtype=float)
    if end_latlon is not None:
        end_pt = latlon_to_local(*end_latlon)
    elif end_xy is not None:
        end_pt = np.array(end_xy, dtype=float)

    if start_pt is not None and end_pt is not None:
        return _nearest_node(pos, start_pt), _nearest_node(pos, end_pt), None
    return None, None, None


def select_routes(G_multi, n_routes, ds=1.0, project_crs="EPSG:6346",
                  origin=(0.0, 0.0), start_latlon=None, end_latlon=None,
                  start_xy=None, end_xy=None, verbose=True):
    """Select up to n_routes edge-disjoint routes and return their dense (x,y)
    polylines plus endpoint metadata.

    Returns a dict:
        {"routes": [{"route_id", "xy", "length_m"}...],
         "start_node", "end_node", "connectivity", "pos"}
    """
    G_simple = nx.Graph(G_multi)  # simple undirected for connectivity analysis
    pos = {n: (float(d["x"]), float(d["y"])) for n, d in G_multi.nodes(data=True)}
    if verbose:
        print(f"  {G_simple.number_of_nodes()} nodes, {G_simple.number_of_edges()} edges")

    start_node, end_node, connectivity = resolve_endpoints(
        pos, n_routes, project_crs, origin,
        start_latlon=start_latlon, end_latlon=end_latlon,
        start_xy=start_xy, end_xy=end_xy,
    )

    if start_node is not None and end_node is not None:
        connectivity = nx.edge_connectivity(G_simple, start_node, end_node)
        if verbose:
            print("  Using EXPLICIT endpoints:")
            print(f"    Start: node {start_node} at {pos[start_node]}")
            print(f"    End:   node {end_node} at {pos[end_node]}")
            print(f"    Edge connectivity: {connectivity}")
        if connectivity < n_routes and verbose:
            print(f"    WARNING: only {connectivity} edge-disjoint routes exist "
                  f"between these endpoints (< requested {n_routes}); "
                  f"will return {connectivity}.")
    else:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        if verbose:
            print(f"  Searching for a corner pair supporting {n_routes} "
                  f"edge-disjoint routes...")
        start_node, end_node, connectivity = find_best_corner_pair(
            G_simple, pos, (min(xs), min(ys)), (max(xs), max(ys)), n_routes
        )
        if verbose:
            print(f"    Start: node {start_node} at {pos[start_node]}")
            print(f"    End:   node {end_node} at {pos[end_node]}")
            print(f"    Edge connectivity: {connectivity} "
                  f"({'>= requested' if connectivity >= n_routes else 'LESS than requested'} {n_routes})")

    node_paths = list(nx.edge_disjoint_paths(G_simple, start_node, end_node))[:n_routes]
    if verbose:
        print(f"  Found {len(node_paths)} routes")

    routes = []
    for i, np_path in enumerate(node_paths):
        xy, length = reconstruct_route_xy(G_multi, np_path, ds=ds)
        routes.append({"route_id": i + 1, "xy": xy, "length_m": float(length)})
        if verbose:
            print(f"    Route {i+1}: {length:.0f} m, {len(xy)} sample points")

    return {"routes": routes, "start_node": start_node, "end_node": end_node,
            "connectivity": int(connectivity), "pos": pos}


# ============================================================
# Persistence
# ============================================================
def routes_to_polylines(routes):
    """Return the selected routes as a list of Nx2 arrays -- the exact format
    05_mrt_network_raytrace.py's sample_polyline() consumes, so the MRT stage
    can ray-trace along ONLY these routes."""
    return [np.asarray(r["xy"], dtype=float) for r in routes]


def save_route_polylines(path, routes):
    """Write the selected routes in path_polylines.pkl format for the MRT stage."""
    polylines = routes_to_polylines(routes)
    highway_tags = [f"route_{r['route_id']}" for r in routes]
    with open(path, "wb") as f:
        pickle.dump({"polylines": polylines, "highway_tags": highway_tags}, f)


def save_selected_routes(path, selection):
    """Persist the full route selection (geometries + endpoint metadata) so
    stages 08/09 load the SAME routes MRT was computed for."""
    payload = {
        "routes": [{"route_id": r["route_id"],
                    "xy": np.asarray(r["xy"], dtype=float),
                    "length_m": float(r["length_m"])} for r in selection["routes"]],
        "start_node": selection["start_node"],
        "end_node": selection["end_node"],
        "start_xy": tuple(selection["pos"][selection["start_node"]]),
        "end_xy": tuple(selection["pos"][selection["end_node"]]),
        "connectivity": int(selection["connectivity"]),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_selected_routes(path):
    """Load routes saved by save_selected_routes()."""
    with open(path, "rb") as f:
        return pickle.load(f)
