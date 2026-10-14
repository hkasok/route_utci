#!/usr/bin/env python3
"""Analytic checks of route_measures.py (route comfort and heat-strain measures)."""
import numpy as np

import route_measures as rm

checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


t = np.linspace(0.0, 20.0, 2001)
check("walk mean of a constant", abs(rm.walk_mean(np.full_like(t, 37.0), t) - 37.0) < 1e-12)
check("walk mean of a ramp is its midpoint", abs(rm.walk_mean(30 + t, t) - 40.0) < 1e-9)
tu = np.r_[0.0, 10.0, np.linspace(10.001, 11.0, 50)]
vu = np.r_[30.0, 30.0, np.full(50, 40.0)]
m = rm.walk_mean(vu, tu)
check("walk mean weights time, not points", abs(m - (300 + 40 * 0.999 + 35 * 0.001) / 11.0) < 1e-6,
      f"{m:.4f} vs point mean {vu.mean():.2f}")

ramp = 30.0 + t                         # crosses 38 at t = 8
check("minutes at/above a crossed threshold", abs(rm.minutes_at_or_above(ramp, t, 38.0) - 12.0) < 1e-9)
check("minutes at/above when always above", abs(rm.minutes_at_or_above(ramp, t, 0.0) - 20.0) < 1e-12)
check("minutes at/above when never above", rm.minutes_at_or_above(ramp, t, 99.0) == 0.0)

# backward mean: constant series with a different 'before' value ramps over the window
const = np.full_like(t, 40.0)
bm = rm.backward_mean(const, t, 3.0, before=26.0)
k15, k3, k10 = (np.searchsorted(t, x) for x in (1.5, 3.0, 10.0))
check("backward mean blends the prior state over the first window",
      abs(bm[0] - 26.0) < 1e-12 and abs(bm[k15] - 33.0) < 1e-9 and abs(bm[k3] - 40.0) < 1e-9
      and abs(bm[k10] - 40.0) < 1e-9, f"{bm[0]:.2f}, {bm[k15]:.2f}, {bm[k3]:.2f}")
bm_ramp = rm.backward_mean(ramp, t, 3.0, before=26.0)
check("backward mean of a ramp lags by half the window", abs(bm_ramp[k10] - (40.0 - 1.5)) < 1e-9,
      f"{bm_ramp[k10]:.4f}")
pulse = np.where((t >= 10.0) & (t <= 15.0), 40.0, 30.0)
bp = rm.backward_mean(pulse, t, 3.0, before=30.0)
check("a 5-min pulse reaches its full value in the 3-min mean", abs(bp.max() - 40.0) < 1e-6)
short = np.where((t >= 10.0) & (t < 11.0), 40.0, 30.0)
bs = rm.backward_mean(short, t, 3.0, before=30.0)
check("a 1-min pulse is diluted to a third", abs(bs.max() - (30 + 10 / 3)) < 0.02, f"{bs.max():.3f}")

d = 1.3 * 60 * t
out, exp = rm.utci_route_measures(pulse, t, d, start_utci_c=26.0)
check("route measures bundle", abs(out["walk_utci_c"] - 32.5) < 0.01
      and abs(out["worst_3min_utci_c"] - 40.0) < 1e-6
      and abs(out["very_strong_stress_min"] - (15.0 + 3 * 0.2 - 10.0 - 3 * 0.8)) < 0.01,
      f"walk UTCI {out['walk_utci_c']:.2f}, t38 {out['very_strong_stress_min']:.3f} min")
hot = np.full_like(t, 40.0)
o2, _ = rm.utci_route_measures(hot, t, d, start_utci_c=26.0)
check("a walk starting indoors is not counted hot before the window fills",
      abs(o2["very_strong_stress_min"] - (20.0 - 3.0 * 12.0 / 14.0)) < 0.01,
      f"{o2['very_strong_stress_min']:.3f} min")

print(f"\n{sum(checks)}/{len(checks)} checks passed")
raise SystemExit(0 if all(checks) else 1)
