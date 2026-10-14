#!/usr/bin/env python3
"""Analytic checks of route_measures.py (route comfort and risk measures)."""
import numpy as np

import route_measures as rm

checks = []


def check(name, ok, detail=""):
    checks.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


t = np.linspace(0.0, 20.0, 2001)
check("walk mean of a constant", abs(rm.walk_mean(np.full_like(t, 37.0), t) - 37.0) < 1e-12)
check("walk mean of a ramp is its midpoint", abs(rm.walk_mean(30 + t, t) - 40.0) < 1e-9)
# unequal spacing: 10 min at 30 then 1 min at 40, sampled densely only at the end
tu = np.r_[0.0, 10.0, np.linspace(10.001, 11.0, 50)]
vu = np.r_[30.0, 30.0, np.full(50, 40.0)]
m = rm.walk_mean(vu, tu)
check("walk mean weights time, not points", abs(m - (300 + 40 * 0.999 + 35 * 0.001) / 11.0) < 1e-6,
      f"{m:.4f} vs point mean {vu.mean():.2f}")

step = np.where(t >= 5.0, 1.0, 0.0)
lag = rm.lagged(step, t, tau_min=2.5)
k = np.searchsorted(t, 7.5)
check("lag reaches 1 - 1/e one time constant after a step",
      abs(lag[k] - (1 - np.exp(-1))) < 0.01, f"{lag[k]:.4f}")
check("lag starts at the first value", lag[0] == step[0])

ramp = 30.0 + t                         # crosses 38 at t = 8
check("minutes at/above a crossed threshold", abs(rm.minutes_at_or_above(ramp, t, 38.0) - 12.0) < 1e-9)
check("minutes at/above when always above", abs(rm.minutes_at_or_above(ramp, t, 0.0) - 20.0) < 1e-12)
check("minutes at/above when never above", rm.minutes_at_or_above(ramp, t, 99.0) == 0.0)

pulse = np.where((t >= 10.0) & (t <= 15.0), 40.0, 30.0)
w, idx = rm.worst_window(pulse, t, 3.0)
check("worst 3-min stretch inside a 5-min pulse", abs(w - 40.0) < 1e-6 and 13.0 <= t[idx] <= 15.0,
      f"{w:.3f} ending at {t[idx]:.2f} min")
w2, _ = rm.worst_window(np.where((t >= 10.0) & (t < 11.0), 40.0, 30.0), t, 3.0)
check("a 1-min pulse is diluted by the 3-min window", abs(w2 - (30 + 10 / 3)) < 0.02, f"{w2:.3f}")
w3, i3 = rm.worst_window(pulse[:100], t[:100], 3.0)
check("walk shorter than the window gives nan", np.isnan(w3) and i3 == -1)

d = 1.3 * 60 * t
out = rm.utci_route_measures(pulse, t, d)
check("route measures bundle", abs(out["walk_utci_c"] - (30 + 10 * 5 / 20)) < 0.01
      and abs(out["worst_3min_utci_c"] - 40) < 1e-6
      and 0 < out["very_strong_stress_lagged_min"] < 5.0,
      f"walk UTCI {out['walk_utci_c']:.2f}, lagged very-strong {out['very_strong_stress_lagged_min']:.2f} min")

print(f"\n{sum(checks)}/{len(checks)} checks passed")
raise SystemExit(0 if all(checks) else 1)
