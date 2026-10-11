#!/usr/bin/env bash
# Ranking Monte Carlo (main-text Table propagation) and its sensitivity runs
# (Supplementary Table mc_sensitivity), with inputs re-derived from the
# validation (11 Oct 2026):
#   flip rate   1 - sun/shade agreement                       (validation_summary.json)
#   load offset 0 .. -(daytime globe excess)                  (validation_summary.json)
#   correlation length: paper_adhoc/calibrate_flip_correlation.py
set -euo pipefail
cd "$(dirname "$0")/.."
FLIP=${FLIP:?}; CORR=${CORR:?}; OFF=${OFF:?}
PAPER="/home/harshin/files/fastUTEC paper"
B=run_output/MMC/postprocessing
SCR=geometry_rebuild_20261011/logs/mc; mkdir -p "$SCR"
echo "== published run: flip $FLIP, correlation $CORR m, offset $OFF..0 ($(date +%T))"
python3 paper_ranking_uncertainty.py --flip-rate $FLIP --correlation-length-m $CORR \
    --load-offset-range $OFF 0 --workers 6 > "$SCR/published.log" 2>&1
rm -rf "$B/ranking_uncertainty_published"; cp -a "$B/ranking_uncertainty" "$B/ranking_uncertainty_published"
run() { name=$1; shift
    echo "== $name ($(date +%T))"; mkdir -p "$SCR/$name"
    python3 paper_ranking_uncertainty.py --flip-rate $FLIP --load-offset-range $OFF 0 --workers 6 \
        --paper-dir "$SCR/$name" "$@" > "$SCR/$name/run.log" 2>&1
    rm -rf "$B/ranking_uncertainty_$name"; cp -a "$B/ranking_uncertainty" "$B/ranking_uncertainty_$name"
}
run n2000 --correlation-length-m $CORR --draws 2000
run corr10m --correlation-length-m 10
run corr40m --correlation-length-m 40
rm -rf "$B/ranking_uncertainty"; cp -a "$B/ranking_uncertainty_published" "$B/ranking_uncertainty"
python3 - "$CORR" <<'PY'
import json, sys
corr = float(sys.argv[1]); c = f"{corr:g}"
B = "run_output/MMC/postprocessing"
runs = ((f"500, {c} m (reported)", "ranking_uncertainty_published"), (f"2000, {c} m", "ranking_uncertainty_n2000"),
        ("500, 10 m", "ranking_uncertainty_corr10m"), ("500, 40 m", "ranking_uncertainty_corr40m"))
L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"Draws, & Best route / & \multicolumn{2}{c}{R3$-$R1, 95\,\% band (\si{\celsius})} & Largest \\",
     r"\cmidrule(lr){3-4}",
     r"correlation & ranking kept & 16:00 & 17:00 & shift (\si{\celsius}) \\", r"\midrule"]
for lab, d in runs:
    S = json.load(open(f"{B}/{d}/summary.json")); s = S["summary"]; mx = 0
    for v in s.values():
        for g in v["pairs"].values():
            q = g["gap_percentiles_c"]; mx = max(mx, abs(g["baseline_gap_c"] - q[0]), abs(q[-1] - g["baseline_gap_c"]))
    pb = min(v["p_best_unchanged"] for v in s.values()); pr = min(v["p_ranking_unchanged"] for v in s.values())
    kept = f"{100*pb:.0f}\\,\\% / {100*pr:.0f}\\,\\%"
    b = lambda dep: "[{:.3f}, {:.3f}]".format(s[dep]["pairs"]["1-3"]["gap_percentiles_c"][0], s[dep]["pairs"]["1-3"]["gap_percentiles_c"][-1])
    L.append(f"{lab} & {kept} & {b('16.0')} & {b('17.0')} & {mx:.3f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open("/home/harshin/files/fastUTEC paper/table_mc_sensitivity.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
PY
echo "== done ($(date +%T))"
