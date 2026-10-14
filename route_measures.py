"""Route-level comfort and heat-strain measures (stages 08 and 09).

A walk is a time series sampled at the walker's arrival times, so every
measure here is a time integral, not a mean over route points (points are
not equally spaced in time on recorded walks with stops).

From UTCI along the walk (stage 08):
  * walk UTCI -- the time-weighted mean UTCI over the walk;
  * experienced UTCI -- the backward running mean of UTCI over the preceding
    3 min, because thermal sensation while walking is associated with the
    preceding 2-3 min of exposure (Lau, Shi & Ng 2019). The minutes before
    departure are spent indoors in the conditions JOS-3 starts from, so the
    window is filled with the UTCI of those conditions until the walker has
    been outside for 3 min;
  * worst stretch -- the highest experienced UTCI and where it occurs;
  * minutes of very strong heat stress -- time with experienced UTCI at or
    above 38 C, the lower bound of that UTCI category.
From JOS-3 (stage 09):
  * minutes with whole-body skin wettedness at or above 0.36; general
    discomfort was not induced below this value in subjects walking at
    4.5 km/h (Fukazawa & Havenith 2009);
  * peak core-temperature rise over the walk and a recovery window after
    arrival, because the core keeps warming after the walker stops;
  * heat stored by the body at arrival, the sum over nodes of heat capacity
    times temperature change since departure.
"""
from __future__ import annotations

import numpy as np

UTCI_VERY_STRONG_C = 38.0
SENSATION_WINDOW_MIN = 3.0
WETTEDNESS_DISCOMFORT = 0.36
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


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
    return float(_trapezoid(v, t) / span)


def backward_mean(values, t_min, window_min=SENSATION_WINDOW_MIN, before=None):
    """Mean of a piecewise-linear series over the preceding ``window_min``.

    Before the first sample the series takes the constant value ``before``
    (the state the walker comes from); if ``before`` is None, the window is
    truncated at the first sample instead."""
    t = _times(t_min)
    v = np.asarray(values, dtype=float)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))))
    start = t - window_min
    inside = start >= t[0]
    out = np.empty_like(v)
    cs = np.interp(start[inside], t, cum)
    out[inside] = (cum[inside] - cs) / window_min
    early = ~inside
    if before is None:
        span = np.maximum(t[early] - t[0], 1e-12)
        out[early] = np.where(t[early] > t[0], cum[early] / span, v[0])
    else:
        out[early] = (cum[early] + before * (t[0] - start[early])) / window_min
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


def utci_route_measures(utci_c, t_min, cumdist_m, start_utci_c):
    """UTCI-based measures for one walk; ``start_utci_c`` is the UTCI of the
    conditions the walker leaves (the JOS-3 indoor start)."""
    experienced = backward_mean(utci_c, t_min, before=start_utci_c)
    k = int(np.argmax(experienced))
    return {
        "walk_utci_c": walk_mean(utci_c, t_min),
        "worst_3min_utci_c": float(experienced[k]),
        "worst_3min_end_distance_m": float(np.asarray(cumdist_m)[k]),
        "very_strong_stress_min": minutes_at_or_above(
            experienced, t_min, UTCI_VERY_STRONG_C),
        "start_utci_c": float(start_utci_c),
    }, experienced
