"""Does colourised-LiDAR brightness separate OSM ground classes in the order literature albedos predict?"""
import glob, json, numpy as np, pandas as pd, laspy
from scipy.spatial import cKDTree
P = "/media/harshin/data_drive/route based UTCI"
tiles = sorted(glob.glob(f"{P}/validation data/Liseben/LAZ/*.laz"))
B = {t: (lambda h: (h.mins, h.maxs))(laspy.open(t).header) for t in tiles}
out = []
for c in range(1, 7):
    o = json.load(open(f"{P}/input/lisbon{c}/case.json"))["coordinates"]
    d = pd.read_csv(f"{P}/run_output/lisbon{c}/osm_ground_materials/route_point_ground_material_diagnostics.csv").iloc[::5]
    X = d.x.values + o["local_origin_x"]; Y = d.y.values + o["local_origin_y"]
    x0, x1, y0, y1 = X.min() - 2, X.max() + 2, Y.min() - 2, Y.max() + 2
    pts = []
    for t, (mn, mx) in B.items():
        if mx[0] < x0 or mn[0] > x1 or mx[1] < y0 or mn[1] > y1: continue
        L = laspy.read(t); m = (L.x >= x0) & (L.x <= x1) & (L.y >= y0) & (L.y <= y1) & (np.asarray(L.classification) == 2)
        vis = (np.asarray(L.red, float) + np.asarray(L.green, float) + np.asarray(L.blue, float)) / 3 / 65535
        pts.append(np.column_stack([np.asarray(L.x)[m], np.asarray(L.y)[m], (0.55 * vis + 0.45 * np.asarray(L.nir, float) / 65535)[m], np.asarray(L.intensity, float)[m] / 65535]))
    A = np.vstack(pts); tr = cKDTree(A[:, :2])
    for x, y, mat in zip(X, Y, d.downward_hit_material):
        i = tr.query_ball_point([x, y], 1.0)
        if len(i) >= 3: out.append((c, mat, np.median(A[i, 2]), np.median(A[i, 3])))
o = pd.DataFrame(out, columns=["case", "material", "bb", "inten"])
lit = {"asphalt_road": 0.12, "asphalt_pedestrian": 0.14, "paving_stone_pedestrian": 0.24, "pedestrian_crossing": 0.26,
       "concrete_pedestrian": 0.30, "calcada_limestone": 0.38, "bare_ground": None, "grass_lawn": 0.23}
g = o.groupby("material").agg(n=("bb", "size"), bb_med=("bb", "median"), bb_iqr=("bb", lambda s: s.quantile(.75) - s.quantile(.25)), inten_med=("inten", "median"))
g["library_albedo"] = [lit.get(k) for k in g.index]
print(g[g.n >= 30].sort_values("bb_med").round(3).to_string())
