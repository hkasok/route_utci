import sys, json, time
sys.path.insert(0, "/media/harshin/data_drive/route based UTCI")
import os; os.chdir("/media/harshin/data_drive/route based UTCI")
import numpy as np, pandas as pd
from importlib import import_module
from pathlib import Path
from scipy.spatial import cKDTree
from dataclasses import replace
from generate_route import load_routes_directory
from microclimate_field import EnvironmentField
from weather_provider import WeatherProvider
s9 = import_module("09_route_thermal_stress_jos3")
pj = import_module("paper_jos3_figures")
root = Path(".")
routes = {r["route_id"]: r for r in load_routes_directory(root/"input/MMC/routes")}
mrt_dir = root/"run_output/MMC/mrt_facet_out"
xyz = np.load(mrt_dir/"path_xyz.npy")
tdf = pd.read_csv(mrt_dir/"times.csv", parse_dates=["time"])
th = np.array([t.hour+t.minute/60 for t in tdf["time"]])
mrt = s9.MrtField(xyz, np.load(mrt_dir/"tmrt_matrix_C.npy"), th, np.load(mrt_dir/"direct_transmission_matrix.npy", mmap_mode="r"))
tree = cKDTree(xyz[:, :2])
env = EnvironmentField(WeatherProvider(csv_path="input/MMC/weather/weather.csv"))
class Scaled:
    def __init__(s, e, f): s.e, s.f = e, f
    def sample(s, p, h):
        x = s.e.sample(p, h)
        return replace(x, wind_speed_ms=x.wind_speed_ms*s.f)
prov = json.load(open("run_output/MMC/viz/route_jos3/clothing_provenance.json"))
clo = {int(r["route_id"]): r["segment_clo"] for r in prov["routes"]}
subj = s9.Subject(1.72, 74.0, 30, "male", 15.0, 2.59, 0.0)
f = np.log(1.1/0.01)/np.log(10/0.01)
dep = float(sys.argv[1]) if len(sys.argv) > 1 else 13.0
for rid in (1, 3):
    xy, d = pj.resample(np.asarray(routes[rid]["xy"]), 2.0)
    _, nn = tree.query(xy)
    arr = dep + d/1.3/3600
    par = s9.jos3_protocol.walking_activity_ratio(subj.make_model(), 1.3, 74.0)
    out = []
    for fac in (1.0, f):
        e = env if fac == 1.0 else Scaled(env, fac)
        w = s9.simulate_walk(xy, arr, nn, mrt, e, subj, clo[rid], par, 10, "t")
        n = s9.simulate_walk(xy, arr, nn, mrt, e, subj, clo[rid], par, 10, "t", radiation_neutral=True)
        out.append((w["final_tcore_rise_c"], n["final_tcore_rise_c"]))
    print(rid, dep, "v10: tot %.4f rad %.4f | v1.1: tot %.4f rad %.4f" % (out[0][0], out[0][0]-out[0][1], out[1][0], out[1][0]-out[1][1]), flush=True)
