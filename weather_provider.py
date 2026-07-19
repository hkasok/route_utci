"""
weather_provider.py -- one consistent source of time-varying meteorology
for the route-stress stages (08 UTCI exposure, 09 JOS-3).

Two modes:

  * REAL DATA (preferred): pass --weather-csv pointing at a file with an
    hour column and any of air_temp_C / rh_pct / wind_ms. Each variable is
    linearly interpolated to the walker's actual arrival time (with 24 h
    wrap), exactly the way Tmrt already is. Columns you omit fall back to
    the parametric defaults below, so a CSV with only air temperature is
    fine.

  * PARAMETRIC (fallback, unchanged behavior): no CSV -> air temperature
    follows the diurnal cosine (mean/amplitude/peak-hour) and RH + wind
    are the constant CLI values. This reproduces the previous behavior
    bit-for-bit when --weather-csv is not supplied.

CSV format (header required; column names are case-insensitive; extra
columns ignored). The time column may be named 'hour' (0-24 decimal) OR
'time' (a parseable timestamp, from which the decimal hour is derived):

    hour,air_temp_C,rh_pct,wind_ms
    0,26.1,82,2.1
    1,25.8,84,1.9
    ...
    13,32.4,58,3.6
    ...

Only the hours you provide are needed; values are interpolated between
them and wrapped at 24 h, so a walk crossing any hour boundary is handled.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _decimal_hours_from_time(series):
    t = pd.to_datetime(series)
    return (t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0).to_numpy()


REQUIRED_VARS = ("air_temp_C", "rh_pct", "wind_ms")


class WeatherProvider:
    def __init__(self, csv_path=None,
                 air_temp_mean_c=29.0, air_temp_amp_c=4.0, air_temp_peak_hour=15.0,
                 rh_pct=70.0, wind_ms=3.1, strict=False):
        # parametric fallbacks (also fill any column missing from the CSV)
        self.mean_c = air_temp_mean_c
        self.amp_c = air_temp_amp_c
        self.peak_hour = air_temp_peak_hour
        self.const_rh = rh_pct
        self.const_wind = wind_ms
        self.strict = strict
        self.csv_path = str(csv_path) if csv_path is not None else None

        self.have_csv = csv_path is not None
        self._hours = None
        self._ta = self._rh = self._wind = None
        self.columns_from_csv = []

        # STRICT MODE: a missing CSV is a hard error, never a silent fallback.
        # Cross-model comparisons are invalidated by unmatched forcing, and a
        # forcing mismatch is invisible in the UTCI output, so it must fail
        # loudly at startup rather than quietly change the answer.
        if strict and not self.have_csv:
            raise ValueError(
                "--require-weather-csv was set but no --weather-csv was given. "
                "Refusing to fall back to parametric weather.")

        if self.have_csv:
            if not Path(csv_path).is_file():
                msg = (f"weather CSV not found: {csv_path}")
                if strict:
                    raise FileNotFoundError(
                        msg + "\nRefusing to fall back to parametric weather "
                              "because --require-weather-csv was set.")
                raise FileNotFoundError(msg)
            df = pd.read_csv(csv_path)
            lower = {c.lower(): c for c in df.columns}
            if "hour" in lower:
                self._hours = df[lower["hour"]].to_numpy(dtype=float)
            elif "time" in lower:
                self._hours = _decimal_hours_from_time(df[lower["time"]])
            else:
                raise ValueError(
                    "weather CSV must have an 'hour' (0-24) or 'time' column")
            order = np.argsort(self._hours)
            self._hours = self._hours[order]
            if len(self._hours) < 2:
                raise ValueError("weather CSV needs at least 2 rows to interpolate")

            def col(*names):
                for n in names:
                    if n in lower:
                        return df[lower[n]].to_numpy(dtype=float)[order]
                return None

            self._ta = col("air_temp_c", "air_temp", "tdb_c", "tdb", "ta_c", "ta")
            self._rh = col("rh_pct", "rh", "relative_humidity_pct", "relative_humidity")
            self._wind = col("wind_ms", "wind", "v_ms", "wind_speed_ms", "v")
            for name, arr in (("air_temp_C", self._ta), ("rh_pct", self._rh),
                              ("wind_ms", self._wind)):
                if arr is not None:
                    if not np.isfinite(arr).all():
                        raise ValueError(
                            f"weather CSV column '{name}' contains missing or "
                            f"non-numeric values")
                    self.columns_from_csv.append(name)

            missing = [c for c in REQUIRED_VARS if c not in self.columns_from_csv]
            if missing and strict:
                raise ValueError(
                    f"weather CSV {csv_path} is missing required column(s): "
                    f"{', '.join(missing)}. Under --require-weather-csv every "
                    f"UTCI driver must come from the CSV; silently substituting "
                    f"a parametric value for one variable is exactly the bug "
                    f"this flag exists to prevent.")
            if missing:
                print(f"  WARNING: weather CSV supplies "
                      f"{', '.join(self.columns_from_csv) or 'nothing'}; "
                      f"falling back to parametric values for "
                      f"{', '.join(missing)}.")

    # -- each accessor returns a value for a scalar or array of hours --
    def _interp(self, hour, table):
        h = np.asarray(hour, dtype=float) % 24.0
        # np.interp with period handles the 24 h wrap for monotonic hours
        return np.interp(h, self._hours, table, period=24.0)

    def air_temp_c(self, hour):
        if self.have_csv and self._ta is not None:
            return self._interp(hour, self._ta)
        h = np.asarray(hour, dtype=float)
        return self.mean_c + self.amp_c * np.cos(
            2.0 * np.pi * (h - self.peak_hour) / 24.0)

    def rh_pct(self, hour):
        if self.have_csv and self._rh is not None:
            return self._interp(hour, self._rh)
        return np.full(np.shape(hour), self.const_rh, dtype=float) \
            if np.ndim(hour) else self.const_rh

    def wind_ms(self, hour):
        if self.have_csv and self._wind is not None:
            return self._interp(hour, self._wind)
        return np.full(np.shape(hour), self.const_wind, dtype=float) \
            if np.ndim(hour) else self.const_wind

    def describe(self):
        if not self.have_csv:
            return (f"parametric weather: air_temp cosine "
                    f"(mean {self.mean_c} C, amp {self.amp_c} C, peak "
                    f"{self.peak_hour}h), RH {self.const_rh}% const, "
                    f"wind {self.const_wind} m/s const")
        got = ", ".join(self.columns_from_csv) if self.columns_from_csv else "none"
        span = f"{self._hours.min():.1f}-{self._hours.max():.1f} h"
        missing = [c for c in ("air_temp_C", "rh_pct", "wind_ms")
                   if c not in self.columns_from_csv]
        fb = f"; fallback for: {', '.join(missing)}" if missing else ""
        return (f"CSV weather ({len(self._hours)} rows, {span}); "
                f"columns used: {got}{fb}")

    def source_of(self, var):
        """'csv' or 'parametric' for one of REQUIRED_VARS."""
        return "csv" if var in self.columns_from_csv else "parametric"

    def provenance(self):
        """Machine-readable record of where every UTCI driver came from.
        Written to disk next to the results so a later model-vs-model
        comparison can prove the two runs were forced identically."""
        return {
            "weather_csv": self.csv_path,
            "strict": self.strict,
            **{f"source_{v}": self.source_of(v) for v in REQUIRED_VARS},
            "all_from_csv": all(v in self.columns_from_csv
                                for v in REQUIRED_VARS),
            "parametric_air_temp_mean_c": self.mean_c,
            "parametric_air_temp_amp_c": self.amp_c,
            "parametric_air_temp_peak_hour": self.peak_hour,
            "parametric_rh_pct": self.const_rh,
            "parametric_wind_ms": self.const_wind,
        }

    def forcing_at(self, hour):
        """(Ta, RH, wind) actually used at the given hour(s)."""
        h = np.asarray(hour, dtype=float)
        return (np.asarray(self.air_temp_c(h), dtype=float) * np.ones_like(h),
                np.asarray(self.rh_pct(h), dtype=float) * np.ones_like(h),
                np.asarray(self.wind_ms(h), dtype=float) * np.ones_like(h))


def add_weather_args(parser):
    """Attach the shared weather CLI flags to an argparse parser. Both 08
    and 09 call this so their weather interface is identical."""
    parser.add_argument("--weather-csv", default=None,
                        help="CSV of real weather with an 'hour' (or 'time') "
                             "column and any of air_temp_C / rh_pct / wind_ms. "
                             "Interpolated to each point's arrival time. "
                             "Omitted columns fall back to the parametric "
                             "defaults below; no CSV = fully parametric "
                             "(previous behavior).")
    parser.add_argument("--air-temp-mean-c", type=float, default=29.0)
    parser.add_argument("--air-temp-amp-c", type=float, default=4.0)
    parser.add_argument("--air-temp-peak-hour", type=float, default=15.0)
    parser.add_argument("--relative-humidity-pct", type=float, default=70.0)
    parser.add_argument("--wind-speed-ms", type=float, default=3.1)
    parser.add_argument("--require-weather-csv", action="store_true",
                        help="Hard-fail instead of falling back to parametric "
                             "weather. Use this for any run that will be "
                             "compared against another model (e.g. SOLWEIG): "
                             "a silent forcing fallback shifts UTCI by O(1.5 "
                             "degC) with no visible symptom.")


def provider_from_args(args):
    return WeatherProvider(
        csv_path=args.weather_csv,
        strict=getattr(args, "require_weather_csv", False),
        air_temp_mean_c=args.air_temp_mean_c,
        air_temp_amp_c=args.air_temp_amp_c,
        air_temp_peak_hour=args.air_temp_peak_hour,
        rh_pct=args.relative_humidity_pct,
        wind_ms=args.wind_speed_ms)
