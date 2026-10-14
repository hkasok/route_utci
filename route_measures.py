"""Route-level comfort and heat-risk measures (stages 08 and 09).

A walk is a time series sampled at the walker's arrival times, so every
measure here is a time integral, not a mean over route points (points are
not equally spaced in time on recorded walks with stops).

Comfort, from UTCI along the walk:
  * walk UTCI -- the time-weighted mean UTCI over the walk;
  * minutes of very strong heat stress -- time with the lagged UTCI at or
    above 38 C, where the lag (first order, 2.5 min) stands for the
    2-3 min over which thermal sensation follows exposure while walking
    (Lau, Shi & Ng 2019), so that brief sunlit gaps do not count as
    sustained discomfort;
  * worst sustained stretch -- the highest 3-min running mean of UTCI and
    where on the route it ends.
Comfort and risk, from JOS-3 (stage 09):
  * minutes with whole-body skin wettedness above 0.36, the onset of
    general discomfort measured during walking (Fukazawa & Havenith 2009);
  * peak core-temperature rise including a window after arrival, because
    the core keeps warming after the walker stops;
  * heat stored by the body at arrival, sum over nodes of heat capacity
    times temperature change.
"""
from __future__ import annotations

import numpy as np

UTCI_VERY_STRONG_C = 38.0
SENSATION_LAG_MIN = 2.5
WORST_WINDOW_MIN = 3.0
WETTEDNESS_DISCOMFORT = 0.36


def _times(t_min):
    t = np.asarray(t_min, dtype=float)
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) < 0):
        raise ValueError("time must be a non-decreasing 1-D series of >= 2 values")
    return t


def walk_mean(values, t_min):
    """Time-weighted (trapezoid) mean of a series over the walk."""
    t = _times(t_min)
    v = np.asarray(values, dtype=float)
    span = t[-1] - t[0]
    if span <= 0:
        return float(np.mean(v))
    return float(np.trapz(v, t) / span)


def lagged(values, t_min, tau_min=SENSATION_LAG_MIN):
    """First-order lag of a series, starting at its first value."""
    t = _times(t_min)
    v = np.asarray(values, dtype=float)
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        a = 1.0 - np.exp(-(t[i] - t[i - 1]) / tau_min)
        out[i] = out[i - 1] + a * (v[i] - out[i - 1])
    return out


def minutes_at_or_above(values, t_min, threshold):
    """Time (min) for which a piecewise-linear series is >= threshold."""
    t = _times(t_min)
    v = np.asarray(values, dtype=float) - threshold
    total = 0.0
    for i in range(1, len(v)):
        dt = t[i] - t[i - 1]
        a, b = v[i - 1], v[i]
        if a >= 0 and b >= 0:
            total += dt
        elif a >= 0 or b >= 0:          # crossing: linear interpolation
            total += dt * (max(a, b) / abs(b - a))
    return float(total)


def worst_window(values, t_min, window_min=WORST_WINDOW_MIN):
    """Highest running mean over ``window_min`` and the index where it ends.

    Returns (nan, -1) if the walk is shorter than the window."""
    t = _times(t_min)
    v = np.asarray(values, dtype=float)
    if t[-1] - t[0] < window_min:
        return float("nan"), -1
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))))
    ends = np.flatnonzero(t - t[0] >= window_min - 1e-9)
    start_t = t[ends] - window_min
    cum_start = np.interp(start_t, t, cum)
    means = (cum[ends] - cum_start) / window_min
    k = int(np.argmax(means))
    return float(means[k]), int(ends[k])


def utci_route_measures(utci_c, t_min, cumdist_m):
    """The three UTCI-based comfort measures for one walk."""
    lag = lagged(utci_c, t_min)
    worst, end_idx = worst_window(utci_c, t_min)
    return {
        "walk_utci_c": walk_mean(utci_c, t_min),
        "very_strong_stress_lagged_min": minutes_at_or_above(
            lag, t_min, UTCI_VERY_STRONG_C),
        "worst_3min_utci_c": worst,
        "worst_3min_end_distance_m": (float(np.asarray(cumdist_m)[end_idx])
                                      if end_idx >= 0 else float("nan")),
    }
