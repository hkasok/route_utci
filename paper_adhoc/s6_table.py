"""Rows of Supplementary Table S6 from the facade-test validation summaries."""
import json, re
from pathlib import Path
R = Path("/media/harshin/data_drive/route based UTCI")
P = Path("/home/harshin/files/fastUTEC paper")
def ldown(table):
    t = table.read_text()
    m = re.search(r"Longwave down & day & \d+ & [\d.]+ & [\d.]+ & ([+-][\d.]+) & [\d.]+ \\\\\s*\n & night & \d+ & [\d.]+ & [\d.]+ & ([+-][\d.]+)", t)
    return float(m.group(1)), float(m.group(2))
rows = [("As reported", P / "validation_summary.json", P / "table_radiometer.tex")]
for v, label in (("wall_m3", "Walls -3 K"), ("wall_m6", "Walls -6 K"), ("albedo_060", "Facade albedo 0.60"),
                 ("conv_13", "Globe convection x1.3"), ("conv_15", "Globe convection x1.5")):
    rows.append((label, R / "facade_tests" / v / "fig" / "validation_summary.json", R / "facade_tests" / v / "fig" / "table_radiometer.tex"))
for label, js, tab in rows:
    d = json.loads(js.read_text()); g = d["globe"]
    ld = ldown(tab)
    print(f"{label:24s} {g['day']['mbe']:+.2f} {g['day']['rmse']:.2f} {g['night']['mbe']:+.2f} "
          f"{d['globe_flux_budget']['day']['excess_Wm2']:.1f} {ld[0]:+.1f} {ld[1]:+.1f}")
