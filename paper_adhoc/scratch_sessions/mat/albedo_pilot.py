"""Pilot: does the colourised Lisbon LiDAR predict the measured effective ground albedo
(K-up / K-down) along the walks better than the model's material map?  Diagnostic only."""
import glob, numpy as np, pandas as pd, laspy
from scipy.spatial import cKDTree
P = "/media/harshin/data_drive/route based UTCI"
H = 1.1          # sensor height above ground (m)
R = 10.0         # footprint radius (m); h^2/R^4 weighting captures ~99 % within 10 m
tiles = sorted(glob.glob(f"{P}/validation data/Liseben/LAZ/*.laz"))
bounds = {}
for t in tiles:
    with laspy.open(t) as r: bounds[t] = (r.header.mins, r.header.maxs)

rows = []
for c in range(1, 7):
    d = pd.read_csv(f"{P}/run_output/lisbon{c}/validation/mrt_lisbon/radiant_flux_comparison_points.csv")
    d = d[(d.period == "day")].copy()
    ghi = d.trec_atmospheric_ghi_wm2
    d = d[(d.measured_swin_wm2 > 0.5 * ghi) & (d.sensor_shortwave_down_Wm2 > 0.5 * ghi)]
    if d.empty: continue
    x0, x1 = d.x_projected_m.min() - R, d.x_projected_m.max() + R
    y0, y1 = d.y_projected_m.min() - R, d.y_projected_m.max() + R
    pts = []
    for t, (mn, mx) in bounds.items():
        if mx[0] < x0 or mn[0] > x1 or mx[1] < y0 or mn[1] > y1: continue
        las = laspy.read(t)
        m = ((las.x >= x0) & (las.x <= x1) & (las.y >= y0) & (las.y <= y1)
             & np.isin(np.asarray(las.classification), [2, 3, 9]))
        if m.any():
            pts.append(np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m],
                                        np.asarray(las.red, float)[m], np.asarray(las.green, float)[m],
                                        np.asarray(las.blue, float)[m], np.asarray(las.nir, float)[m],
                                        np.asarray(las.intensity, float)[m]]))
    A = np.vstack(pts); tree = cKDTree(A[:, :2])
    for _, s in d.iterrows():
        idx = tree.query_ball_point([s.x_projected_m, s.y_projected_m], R)
        if len(idx) < 20: continue
        q = A[idx]; rho2 = (q[:, 0] - s.x_projected_m) ** 2 + (q[:, 1] - s.y_projected_m) ** 2
        w = H ** 2 / (H ** 2 + rho2) ** 2; w /= w.sum()
        vis = (q[:, 2] + q[:, 3] + q[:, 4]) / 3 / 65535
        rows.append(dict(case=c, meas=s.measured_swout_wm2 / s.measured_swin_wm2,
                         model=s.sensor_shortwave_up_Wm2 / s.sensor_shortwave_down_Wm2,
                         vis=(w * vis).sum(), nir=(w * q[:, 5] / 65535).sum(),
                         bb=(w * (0.55 * vis + 0.45 * q[:, 5] / 65535)).sum(),
                         inten=(w * q[:, 6] / 65535).sum(), n=len(idx)))
o = pd.DataFrame(rows)
o.to_csv("albedo_pilot_points.csv", index=False)
print("samples", len(o), "per case", o.case.value_counts().sort_index().to_dict())
print("measured albedo mean %.3f sd %.3f" % (o.meas.mean(), o.meas.std()))
for k in ("model", "vis", "nir", "bb", "inten"):
    r = np.corrcoef(o.meas, o[k])[0, 1]
    rc = [np.corrcoef(g.meas, g[k])[0, 1] for _, g in o.groupby("case") if len(g) > 20]
    print(f"{k:6s} pooled r = {r:+.2f}   per-walk r = {', '.join(f'{x:+.2f}' for x in rc)}")
