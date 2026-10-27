"""weather_from_sensors.main() with the diurnal fallback also covering an
inverted anchor pair (day window cooler than the night anchor), as for lisbon5/_w2."""
import sys
import numpy as np
sys.path.insert(0, ".")
import weather_from_sensors as w

def fit_allow_inverted(anchor_hours, anchor_temperature_c, *, sunrise, sunset,
                       peak_lag_h=1.5, night_decay=3.0):
    hours = np.asarray(anchor_hours, float); values = np.asarray(anchor_temperature_c, float)
    phi = w.diurnal_shape(hours, sunrise=sunrise, sunset=sunset,
                          peak_lag_h=peak_lag_h, night_decay=night_decay)
    if len(hours) < 2 or np.ptp(phi) < 1e-3:
        return None
    design = np.column_stack([1.0 - phi, phi])
    sol, *_ = np.linalg.lstsq(design, values, rcond=None)
    mn, mx = float(sol[0]), float(sol[1])
    res = design @ sol - values
    return {"t_min_c": mn, "t_max_c": mx, "diurnal_range_c": mx - mn,
            "sunrise_h": float(sunrise), "sunset_h": float(sunset),
            "peak_lag_h": float(peak_lag_h), "night_decay": float(night_decay),
            "anchor_hours": [float(h) for h in hours],
            "anchor_temperature_c": [float(v) for v in values],
            "anchor_rmse_c": float(np.sqrt(np.mean(res ** 2))),
            "shape_reference": "Parton & Logan (1981) diurnal form"}

w.fit_diurnal_anchors = fit_allow_inverted
sys.argv = ["weather_from_sensors.py"] + sys.argv[1:]
raise SystemExit(w.main())
