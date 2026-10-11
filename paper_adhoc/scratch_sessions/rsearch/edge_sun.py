"""Direct-beam transmission on every edge of the domain-clipped campus network,
07:00-19:00 every 10 min, with the pipeline's own ray-tracing functions."""
import importlib.util, sys, time, pickle
import numpy as np, pandas as pd, osmnx as ox, trimesh
from shapely.geometry import LineString
P = "/media/harshin/data_drive/route based UTCI"; sys.path.insert(0, P)
spec = importlib.util.spec_from_file_location("s05", f"{P}/05_mrt_network_raytrace.py")
s05 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s05)
OUT = "/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/rsearch"
DS, H, K_DIR = 2.0, 1.1, 0.45
G = ox.load_graphml(f"{P}/run_output/MMC/osm_paths/pedestrian_network_domain.graphml")
edges, pts, owner = [], [], []
for u, v, k, d in G.edges(keys=True, data=True):
    if (v, u) in [(e[0], e[1]) for e in edges[-3:]]: pass
    geom = d.get("geometry") or LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
    xy = s05.sample_polyline(np.asarray(geom.coords)[:, :2], DS)
    if len(xy) == 0: xy = np.asarray(geom.coords)[:1, :2]
    edges.append((u, v, k, float(geom.length))); owner.append(np.full(len(xy), len(edges) - 1)); pts.append(xy)
xy = np.vstack(pts); owner = np.concatenate(owner)
print(f"{len(edges)} directed edges, {len(xy)} sample points"); sys.stdout.flush()
geo = f"{P}/input/MMC/geometry"
gi = s05.get_intersector(s05.load_mesh(f"{geo}/ground_and_water_final.stl"))
bi = s05.get_intersector(s05.load_mesh(f"{geo}/building_final.stl"))
vi = s05.get_intersector(s05.load_mesh(f"{geo}/vegetation_final.stl"))
z = s05.ground_height_lookup(xy, gi) + H
xyz = np.column_stack([xy, z])
t = pd.read_csv(f"{P}/run_output/MMC/mrt_facet_out/times.csv")
hh = pd.to_datetime(t.time).dt.hour + pd.to_datetime(t.time).dt.minute / 60
sel = (hh >= 7) & (hh <= 19.5) & (t.elevation_deg > 0)
T = t[sel].reset_index(drop=True); hours = hh[sel].values
tau = np.zeros((len(T), len(xyz)), np.float32); t0 = time.time()
for i, r in T.iterrows():
    sv = s05.sun_vector_enu(r.azimuth_deg, r.elevation_deg)
    tau[i] = s05.direct_solar_transmission_batched(xyz + sv * 0.0, sv, bi, vi, K_DIR, 20000)
    if i % 10 == 0: print(f"  {hours[i]:.2f} h done ({time.time()-t0:.0f} s)"); sys.stdout.flush()
pickle.dump(dict(edges=edges, owner=owner, xy=xy, hours=hours, tau=tau,
                 elev=T.elevation_deg.values, dni=T.DNI_Wm2.values), open(f"{OUT}/edge_sun.pkl", "wb"))
print("saved")
