import sys; sys.path.insert(0, "/media/harshin/data_drive/route based UTCI")
import numpy as np, pandas as pd, osmnx as ox, networkx as nx
from shapely.geometry import box, LineString, Point
import route_selection as rs
X0,X1,Y0,Y1 = 562027.7, 563202.1, 2848417.0, 2849228.8
M = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0           # inward margin, m
dom = box(X0 + M, Y0 + M, X1 - M, Y1 - M)
G = ox.load_graphml("run_output/MMC/osm_paths/pedestrian_network.graphml")
H = G.copy()
drop = [n for n, d in H.nodes(data=True) if not dom.covers(Point(float(d["x"]), float(d["y"])))]
H.remove_nodes_from(drop)
bad = []
for u, v, k, d in H.edges(keys=True, data=True):
    geom = d.get("geometry") or LineString([(H.nodes[u]["x"], H.nodes[u]["y"]), (H.nodes[v]["x"], H.nodes[v]["y"])])
    if not dom.covers(geom): bad.append((u, v, k))
H.remove_edges_from(bad)
print(f"margin {M} m: removed {len(drop)} nodes, {len(bad)} edges partly outside")
old = {r: pd.read_csv(f"input/MMC/routes/route_{r}.csv") for r in (1, 2, 3)}
s = old[1][["x_local_m", "y_local_m"]].iloc[0].tolist(); e = old[1][["x_local_m", "y_local_m"]].iloc[-1].tolist()
sel = rs.select_routes(H, 3, ds=1.0, start_xy=s, end_xy=e, verbose=True)
for r in sel["routes"]:
    xy = r["xy"]
    best = min((1, 2, 3), key=lambda k: abs(len(old[k]) - len(xy)))
    same = {k: (len(old[k]) == len(xy) and np.allclose(old[k][["x_local_m", "y_local_m"]].values, xy, atol=0.01)) for k in (1, 2, 3)}
    print(f"  new route {r['route_id']}: {r['length_m']:.1f} m; identical to old: {[k for k,v in same.items() if v]}")
if len(sys.argv) > 2:
    import pickle; pickle.dump(sel, open(sys.argv[2], "wb"))
