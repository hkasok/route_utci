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

WIND HEIGHT: the UTCI polynomial takes wind at 10 m above ground; the
UTCI-Fiala model derives body-level wind internally, and the operational
procedure (Broede et al. 2012, Eq. 3) converts a wind measured at height x
with  va = va_x * log(10/0.01) / log(x/0.01).  Supplying pedestrian-level
wind unconverted understates ventilation and overstates heat stress. An
optional constant column ``wind_height_m`` records the height of
``wind_ms``; it defaults to 10 m, the meteorological standard, so a CSV
without it behaves exactly as before. Only UTCI uses the converted value;
JOS-3 and the globe take the wind at the body.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from physical_checks import check_air_temp_c, check_rh_pct, check_wind_ms

# UTCI reference height and the 0.01 m term of the height-conversion profile
# prescribed by the UTCI operational procedure (Broede et al. 2012, Eq. 3).
UTCI_REFERENCE_HEIGHT_M = 10.0
UTCI_PROFILE_ROUGHNESS_M = 0.01


def utci_reference_wind(speed_ms, height_m):
    """Refer a wind measured at ``height_m`` to the UTCI 10 m reference."""
    if not height_m > UTCI_PROFILE_ROUGHNESS_M:
        raise ValueError(f"wind height {height_m} m must exceed the UTCI "
                         f"profile roughness {UTCI_PROFILE_ROUGHNESS_M} m")
    factor = (np.log(UTCI_REFERENCE_HEIGHT_M / UTCI_PROFILE_ROUGHNESS_M)
              / np.log(height_m / UTCI_PROFILE_ROUGHNESS_M))
    return np.asarray(speed_ms, dtype=float) * factor


def _decimal_hours_from_time(series):
    t = pd.to_datetime(series)
    return (t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0).to_numpy()


REQUIRED_VARS = ("air_temp_C", "rh_pct", "wind_ms")


class WeatherProvider:
    def __init__(self, csv_path=None,
                 air_temp_mean_c=29.0, air_temp_amp_c=4.0, air_temp_peak_hour=15.0,
                 rh_pct=70.0, wind_ms=3.1, strict=False,
                 wind_height_m=UTCI_REFERENCE_HEIGHT_M):
        # parametric fallbacks (also fill any column missing from the CSV)
        # Validate the parametric constants too -- they are used verbatim
        # whenever the CSV is absent or lacks a column, so a bad --relative-
        # humidity-pct (e.g. 0.7 meaning 70%) must fail just as loudly.
        check_air_temp_c([air_temp_mean_c - abs(air_temp_amp_c),
                          air_temp_mean_c + abs(air_temp_amp_c)],
                         "parametric air temperature (mean +/- amplitude)")
        check_rh_pct(rh_pct, "parametric --relative-humidity-pct")
        check_wind_ms(wind_ms, "parametric --wind-speed-ms")

        self.mean_c = air_temp_mean_c
        self.amp_c = air_temp_amp_c
        self.peak_hour = air_temp_peak_hour
        self.const_rh = rh_pct
        self.const_wind = wind_ms
        self.wind_height_m = float(wind_height_m)
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
            heights = col("wind_height_m")
            if heights is not None:
                if not (np.isfinite(heights).all()
                        and np.allclose(heights, heights[0])):
                    raise ValueError(
                        "weather CSV column 'wind_height_m' must be one "
                        "constant finite height")
                self.wind_height_m = float(heights[0])
            # OPTIONAL time-varying INLET (free-stream) wind for the step-3
            # potential-flow field. It is a different quantity from wind_ms:
            # wind_ms is the wind a person experiences, while this is the
            # boundary speed the flow field is scaled by. Absent = the
            # historical behaviour, where wind_ms serves as both.
            self._wind_inlet = col("wind_inlet_ms", "inlet_wind_ms",
                                   "reference_wind_ms")
            # FREE-STREAM reference wind: the measured cart wind lifted through
            # the urban canopy profile to a height above the roughness
            # sublayer. This is what an "a + b*U" convection correlation was
            # calibrated against -- an undisturbed approach velocity, not the
            # sheltered in-canopy value. Distinct from wind_inlet_ms, which is
            # a pedestrian-height boundary speed for the 2-D flow solve.
            self._wind_freestream = col("wind_freestream_ms",
                                        "free_stream_wind_ms")
            if self._wind_freestream is not None:
                if not np.isfinite(self._wind_freestream).all() \
                        or np.any(self._wind_freestream < 0):
                    raise ValueError(
                        "weather CSV column 'wind_freestream_ms' must be "
                        f"finite and non-negative: {csv_path}")
                self.columns_from_csv.append("wind_freestream_ms")
            if self._wind_inlet is not None:
                if not np.isfinite(self._wind_inlet).all() \
                        or np.any(self._wind_inlet < 0):
                    raise ValueError(
                        f"weather CSV column 'wind_inlet_ms' must be finite and "
                        f"non-negative: {csv_path}")
                self.columns_from_csv.append("wind_inlet_ms")
            # Validate UNITS and physical range at the source, so a bad CSV
            # fails here once instead of silently corrupting every stage
            # (RH as a 0-1 fraction is the classic one -- see physical_checks).
            ctx = f"weather CSV {csv_path}"
            checkers = {"air_temp_C": check_air_temp_c, "rh_pct": check_rh_pct,
                        "wind_ms": check_wind_ms}
            for name, arr in (("air_temp_C", self._ta), ("rh_pct", self._rh),
                              ("wind_ms", self._wind)):
                if arr is not None:
                    if not np.isfinite(arr).all():
                        raise ValueError(
                            f"weather CSV column '{name}' contains missing or "
                            f"non-numeric values")
                    checkers[name](arr, f"{ctx} column '{name}'")
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

    def utci_wind_10m_ms(self, hour):
        """``wind_ms`` referred to the UTCI 10 m reference height."""
        return utci_reference_wind(self.wind_ms(hour), self.wind_height_m)

    def has_free_stream_wind(self):
        """True when the CSV supplies a canopy-lifted free-stream series."""
        return self.have_csv and self._wind_freestream is not None

    def free_stream_wind_ms(self, hour):
        """Free-stream reference wind for convection correlations.

        Falls back to :meth:`wind_ms` when absent, which reproduces the
        historical behaviour of driving convection with the sheltered wind.
        """
        if self.has_free_stream_wind():
            return self._interp(hour, self._wind_freestream)
        return self.wind_ms(hour)

    def has_inlet_wind(self):
        """True when the CSV supplies a separate inlet (free-stream) series."""
        return self.have_csv and self._wind_inlet is not None

    def inlet_wind_ms(self, hour):
        """Time-varying inlet wind for the potential-flow field.

        Falls back to :meth:`wind_ms` when the CSV has no inlet column, which
        reproduces the historical single-series behaviour exactly.
        """
        if self.has_inlet_wind():
            return self._interp(hour, self._wind_inlet)
        return self.wind_ms(hour)

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
            "wind_height_m": self.wind_height_m,
            "utci_wind_factor": float(utci_reference_wind(
                1.0, self.wind_height_m)),
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
