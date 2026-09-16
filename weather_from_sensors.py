#!/usr/bin/env python3
"""Build TREC-Route weather and solar forcing from mobile sensor measurements.

WHY
---
A mobile campaign measures the meteorology the walker actually experienced,
which is a better boundary condition than a generic station file. But raw
mobile data cannot be used as forcing directly, for two reasons this module
handles explicitly:

1. **The radiometer walks through shade.** Its shortwave series is
   ``GHI_atmosphere(t) x local_shade(t)``, and the ray tracer already computes
   the shade. Replaying the measured series as forcing would apply shade
   twice. Only the *upper envelope* of the measurements -- the samples taken
   in the open -- constrains the atmosphere, and that envelope is what this
   module fits, as a time-dependent series rather than one session constant.

2. **The walk is short; the model day is not.** The Lisbon campaigns cover
   roughly 15:00-22:45. The pipeline integrates a full 24 h and repeats it for
   several diurnal spin-up cycles, so the hours outside the walk still set the
   surface temperatures the walk starts from. With a measured window only,
   ``numpy.interp(..., period=24)`` draws a straight chord across the missing
   17 hours, which for these cases implies a pre-dawn air temperature of
   23-26 degC -- a night that never cools. This module reconstructs those
   hours instead, and labels every row as measured or reconstructed.

WHAT IS PRODUCED
----------------
``weather/<name>.csv``
    hour, air_temp_C, rh_pct, wind_ms, plus ``source`` (measured |
    reconstructed) and dew-point diagnostics. Consumed by WeatherProvider.
``config/<name>_solar_components.csv``
    hour, DNI_Wm2, DHI_Wm2, GHI_Wm2, cloud_fraction. Consumed through the
    existing ``components_csv`` radiation-forcing mode, so the shortwave and
    the cloud-dependent sky longwave stay on ONE convention.
``config/<name>.json``
    The radiation-forcing config pointing at that components file.
``weather/<name>_provenance.json``
    Every fitted parameter, envelope diagnostic and reconstruction choice.

METHOD
------
Solar: for each daytime sample, ``ratio = SWin / GHI_clear`` removes the solar
geometry and leaves atmospheric transmission. The unshaded envelope of that
ratio is extracted robustly (``robust_unshaded_ratio``): a WIDE rolling high
quantile detects shade, because shade is a multiplicative reduction and a wide
window is unlikely to be shaded end to end, and a NARROW rolling median of the
surviving samples then estimates the level, because once shade is gone the
median is robust against upward spikes too. Detection and estimation use
different statistics on purpose -- a narrow median used for detection descends
into any shaded stretch approaching its own width, at which point the dip stops
looking like a dip and nothing is flagged, silently.

The envelope is bounded by a CLEAN-SKY ceiling derived from a low Linke
turbidity, not by an arbitrary constant. If the fitted envelope exceeds the
climatological clear sky, the clear-sky BASELINE is refitted to the turbidity
that reproduces it, because the cloud model can only attenuate: routing an
above-one clearness through it saturates at zero cloud and silently clamps the
forcing back to climatological clear sky. Only the remaining, genuinely
attenuating part is inverted through the pipeline's own
:func:`radiation_forcing.apply_cloud_adjustment`, so nothing here invents a
second radiation convention.

Anything the unshaded sensor reads ABOVE a clean sky is reported as residual
instrument bias and is NOT absorbed into the forcing; injecting it would add
non-physical energy to every surface energy balance in the domain.

Meteorology: inside the measured windows, binned medians of the sensor series.
Outside them, air temperature follows the Parton-Logan diurnal form (sine from
sunrise to a lagged afternoon peak, exponential decay overnight) whose Tmin and
Tmax are solved so the curve passes through the median of each measured
window. A campaign is a handful of short walks, not a diurnal record, so the
two shape constants are documented model choices while the two amplitude
parameters come from the data. Humidity is carried by holding the measured dew
point -- physically far better than interpolating relative humidity through a
temperature swing -- and recomputing RH. Wind holds the measured median. Every
reconstructed row is labelled as such.

Note that when a case's night walk was recorded on a different date from its
day walk, the two anchors come from different days; that is recorded in the
provenance rather than hidden.

The reconstruction is an explicit, inspectable modelling choice for hours
nobody measured. It is not presented as data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from radiation_forcing import apply_cloud_adjustment
import wind_profile as wp

ROOT = Path(__file__).resolve().parent
MEASUREMENT_RELATIVE = Path("measurements/experimental_measurements_all_routes.csv")


def saturation_vapour_pressure_hpa(temperature_c):
    """Magnus formula, matching the convention used in sensitivity.py."""
    t = np.asarray(temperature_c, dtype=float)
    return 6.112 * np.exp(17.67 * t / (t + 243.5))


def relative_humidity_from_dewpoint(temperature_c, dewpoint_c):
    rh = 100.0 * (saturation_vapour_pressure_hpa(dewpoint_c)
                  / saturation_vapour_pressure_hpa(temperature_c))
    return np.clip(rh, 1.0, 100.0)


def dewpoint_from_relative_humidity(temperature_c, rh_pct):
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    vapour = saturation_vapour_pressure_hpa(temperature_c) * rh / 100.0
    ratio = np.log(vapour / 6.112)
    return 243.5 * ratio / (17.67 - ratio)


def load_case_measurements(case_dir: Path) -> pd.DataFrame:
    """Measured samples with a local decimal hour, sorted in time."""
    path = case_dir / MEASUREMENT_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"case has no mobile measurements: {path}")
    frame = pd.read_csv(path)
    stamp = pd.to_datetime(frame["timestamp_local_refined"], format="ISO8601")
    frame = frame.assign(
        hour=(stamp.dt.hour + stamp.dt.minute / 60.0
              + stamp.dt.second / 3600.0 + stamp.dt.microsecond / 3.6e9),
        local_date=stamp.dt.date)
    return frame.sort_values("hour").reset_index(drop=True)


def clear_sky_series(hours, *, latitude, longitude, timezone_name, date):
    """pvlib Ineichen clear sky and solar elevation at the given local hours."""
    base = pd.Timestamp(date)
    stamps = pd.DatetimeIndex(
        [base + pd.Timedelta(hours=float(h)) for h in np.asarray(hours, float)]
    ).tz_localize(timezone_name, ambiguous=True, nonexistent="shift_forward")
    location = pvlib.location.Location(latitude=latitude, longitude=longitude,
                                       tz=timezone_name)
    solar = pvlib.solarposition.get_solarposition(stamps, latitude, longitude)
    clear = location.get_clearsky(stamps, model="ineichen")
    return (clear["dni"].to_numpy(float), clear["dhi"].to_numpy(float),
            clear["ghi"].to_numpy(float),
            solar["apparent_elevation"].to_numpy(float))


def clear_sky_at_turbidity(hours, *, latitude, longitude, timezone_name, date,
                           linke_turbidity: float):
    """Full Ineichen components at an EXPLICIT Linke turbidity.

    Same call as ``clear_sky_series`` but with the turbidity pinned instead of
    taken from the monthly climatology, so a day that was genuinely cleaner than
    the monthly mean can be represented as what it was -- a brighter clear sky --
    rather than being forced through a cloud model that can only darken.
    """
    base = pd.Timestamp(date)
    stamps = pd.DatetimeIndex(
        [base + pd.Timedelta(hours=float(h)) for h in np.asarray(hours, float)]
    ).tz_localize(timezone_name, ambiguous=True, nonexistent="shift_forward")
    location = pvlib.location.Location(latitude=latitude, longitude=longitude,
                                       tz=timezone_name)
    solar = pvlib.solarposition.get_solarposition(stamps, latitude, longitude)
    turbidity = pd.Series(float(linke_turbidity), index=stamps)
    clear = location.get_clearsky(stamps, model="ineichen",
                                  linke_turbidity=turbidity)
    return (clear["dni"].to_numpy(float), clear["dhi"].to_numpy(float),
            clear["ghi"].to_numpy(float),
            solar["apparent_elevation"].to_numpy(float))


def clean_sky_ceiling(hours, *, latitude, longitude, timezone_name, date,
                      linke_turbidity: float) -> np.ndarray:
    """Clear-sky GHI under a CLEAN atmosphere -- the physical upper bound.

    pvlib's default Ineichen call uses a monthly Linke-turbidity CLIMATOLOGY,
    which for Lisbon in June is about 3.6. A genuinely clean maritime day has a
    lower turbidity and therefore a genuinely higher clear-sky irradiance, so a
    measured value modestly above the climatological clear sky is not
    automatically an instrument fault -- it can simply be a cleaner day than the
    monthly mean.

    This gives the ceiling a PHYSICAL derivation instead of an arbitrary
    constant: at Linke ~2.0 the atmosphere is close to Rayleigh-limited, and
    nothing horizontal can exceed that for a sustained period. Measurements
    above it are instrument error (mount tilt, calibration), not weather, and
    must not be absorbed into a domain-wide solar forcing.
    """
    base = pd.Timestamp(date)
    stamps = pd.DatetimeIndex(
        [base + pd.Timedelta(hours=float(h)) for h in np.asarray(hours, float)]
    ).tz_localize(timezone_name, ambiguous=True, nonexistent="shift_forward")
    location = pvlib.location.Location(latitude=latitude, longitude=longitude,
                                       tz=timezone_name)
    turbidity = pd.Series(float(linke_turbidity), index=stamps)
    return location.get_clearsky(
        stamps, model="ineichen", linke_turbidity=turbidity)["ghi"].to_numpy(float)


def robust_unshaded_ratio(hours, ratio, *, window_samples: int,
                          shade_tolerance: float, iterations: int = 3,
                          reference_window_multiple: float = 4.0,
                          reference_quantile: float = 0.90) -> dict:
    """Separate unshaded from shaded samples, robustly, in both directions.

    THE PROBLEM THIS SOLVES
    -----------------------
    A cart walking a city route spends much of its time in shade. The signal we
    want -- what the ATMOSPHERE was doing over the whole domain -- is carried
    only by the samples taken in the open. Estimating it with a fixed high
    quantile inside fixed time windows fails whenever a window happens to be
    mostly shaded: the quantile then sits on the brightest SHADED sample and
    drags the whole forcing down.

    THE METHOD, AND WHY IT IS TWO STATISTICS AND NOT ONE
    ----------------------------------------------------
    Detection and estimation want different statistics, and using one for both
    is what breaks.

    * DETECTION uses a rolling high QUANTILE over a WIDE window. Shade is a
      multiplicative reduction, so the unshaded level is an upper envelope; a
      wide window is very unlikely to be shaded end to end. A narrow rolling
      MEDIAN cannot do this job: once a shaded stretch approaches the window
      width the median descends into the shade with it, the dip stops looking
      like a dip, and nothing is flagged. That failure is silent and it is why
      the reference here is deliberately wide and high rather than local and
      central.

    * ESTIMATION uses a rolling MEDIAN of the surviving unshaded samples. Once
      shade is removed, the median is the right summary because it is robust in
      BOTH directions -- a lone upward spike from cloud-edge enhancement or a
      lurch of the mount cannot set the forcing any more than a dip can.

    A sample is shaded when it falls more than ``shade_tolerance`` BELOW the
    reference, in relative terms. The reference is then rebuilt from the
    surviving samples and the test repeated, which lets the estimate climb out
    of a stretch that started mostly shaded.

    Only DOWNWARD excursions are called shade. An upward excursion is not shade
    and is handled separately by the physical clean-sky ceiling -- rejecting it
    here as an outlier would quietly discard evidence about the instrument.
    """
    hours = np.asarray(hours, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    if hours.size != ratio.size:
        raise ValueError("hours and ratio must have equal length")
    if hours.size == 0:
        return {"unshaded_ratio": ratio.copy(), "envelope": ratio.copy(),
                "shaded": np.zeros(0, dtype=bool), "n_unshaded": 0,
                "iterations": 0, "reference": ratio.copy()}
    window = max(3, int(window_samples) | 1)
    wide = max(window, int(window * float(reference_window_multiple)) | 1)

    def _fill(values):
        values = np.asarray(values, dtype=float)
        if not np.isnan(values).any():
            return values
        valid = ~np.isnan(values)
        if not valid.any():
            return np.full_like(values, np.nan)
        return np.interp(np.arange(len(values)), np.flatnonzero(valid),
                         values[valid])

    working = pd.Series(ratio, dtype=float)
    shaded = np.zeros(ratio.shape, dtype=bool)
    reference = np.full(ratio.shape, np.nan)
    for _ in range(max(1, int(iterations))):
        reference = _fill(working.rolling(
            wide, center=True, min_periods=1).quantile(reference_quantile))
        if np.isnan(reference).all():
            break
        previous = shaded.copy()
        shaded = (ratio < reference * (1.0 - float(shade_tolerance))) & np.isfinite(ratio)
        working = pd.Series(np.where(shaded, np.nan, ratio), dtype=float)
        if np.array_equal(previous, shaded):
            break

    envelope = _fill(working.rolling(window, center=True,
                                    min_periods=1).median().to_numpy(float))
    if np.isnan(envelope).all():
        envelope = _fill(reference)
    return {
        "unshaded_ratio": np.where(shaded, np.nan, ratio),
        "envelope": envelope,
        "reference": reference,
        "shaded": shaded,
        "n_unshaded": int((~shaded).sum()),
        "iterations": int(max(1, int(iterations))),
        "reference_window_samples": int(wide),
        "reference_quantile": float(reference_quantile),
    }


def solar_clearness_envelope(measurements: pd.DataFrame, *, latitude, longitude,
                             timezone_name, date, window_min: float,
                             quantile: float, smooth_windows: int,
                             minimum_clear_ghi: float,
                             maximum_clearness: float,
                             estimator: str = "robust_median",
                             robust_window_samples: int = 31,
                             shade_tolerance: float = 0.12,
                             ceiling_linke_turbidity: float = 2.0) -> dict:
    """Fit the upper envelope of mobile SWin as atmospheric clearness k(t).

    ``k = SWin / GHI_clear`` for samples the sensor took in the open. Shaded
    samples sit below the envelope and are simply not selected by the rolling
    high quantile, which is why no shade mask is needed here.
    """
    day = measurements.dropna(subset=["SWin"]).copy()
    _dni, _dhi, ghi_clear, elevation = clear_sky_series(
        day["hour"].to_numpy(float), latitude=latitude, longitude=longitude,
        timezone_name=timezone_name, date=date)
    day["ghi_clear"] = ghi_clear
    day["elevation_deg"] = elevation
    usable = day[(day["ghi_clear"] > minimum_clear_ghi)
                 & (day["SWin"] >= 0.0)].copy()
    if len(usable) < 10:
        raise ValueError(
            "not enough daytime samples above the clear-sky threshold to fit a "
            "solar envelope; lower --minimum-clear-ghi or check the case")
    usable["ratio"] = usable["SWin"] / usable["ghi_clear"]

    # ------------------------------------------------------------------
    # PHYSICAL CEILING
    #
    # Derived, not chosen: the clear-sky irradiance a CLEAN atmosphere would
    # give at this site and time. Anything sustained above it is instrument
    # error rather than weather. The previous hard clamp at clearness 1.0 threw
    # away the whole difference between the monthly turbidity climatology and a
    # genuinely clean day -- about +7% here -- and that is why the fitted
    # forcing sat at the climatological clear sky (~875 W/m2) while the sensor
    # read ~1000 in the open.
    # ------------------------------------------------------------------
    ceiling_ghi = clean_sky_ceiling(
        usable["hour"].to_numpy(float), latitude=latitude, longitude=longitude,
        timezone_name=timezone_name, date=date,
        linke_turbidity=ceiling_linke_turbidity)
    ceiling_ratio = ceiling_ghi / np.maximum(usable["ghi_clear"].to_numpy(float), 1.0)
    effective_ceiling = np.maximum(float(maximum_clearness), ceiling_ratio)

    if estimator == "robust_median":
        robust = robust_unshaded_ratio(
            usable["hour"].to_numpy(float), usable["ratio"].to_numpy(float),
            window_samples=robust_window_samples,
            shade_tolerance=shade_tolerance)
        raw_envelope = robust["envelope"]
        above_ceiling = raw_envelope > effective_ceiling
        clearness = np.minimum(raw_envelope, effective_ceiling)
        # Resample the per-sample envelope onto regular windows so the returned
        # shape matches the legacy contract that build_meteorology consumes.
        step = window_min / 60.0
        edges = np.arange(usable["hour"].min(), usable["hour"].max() + step, step)
        hours_used = usable["hour"].to_numpy(float)
        centres, values, counts = [], [], []
        for start, stop in zip(edges[:-1], edges[1:]):
            block = (hours_used >= start) & (hours_used < stop)
            if not block.any():
                continue
            centres.append(0.5 * (start + stop))
            values.append(float(np.nanmedian(clearness[block])))
            counts.append(int(block.sum()))
        if len(centres) < 3:
            raise ValueError("solar envelope needs at least three populated windows")
        centres = np.asarray(centres, float)
        smoothed = pd.Series(values).rolling(
            max(1, int(smooth_windows)), center=True,
            min_periods=1).median().to_numpy()
        return {
            "hours": centres,
            "clearness": smoothed,
            "raw_clearness": np.asarray(values, float),
            "samples_per_window": np.asarray(counts, int),
            "n_usable_samples": int(len(usable)),
            "measured_swin_max_wm2": float(usable["SWin"].max()),
            "clear_ghi_max_wm2": float(usable["ghi_clear"].max()),
            "raw_clearness_min": float(np.min(values)),
            "raw_clearness_median": float(np.median(values)),
            "raw_clearness_max": float(np.max(values)),
            "clamped_windows": int(np.sum(np.asarray(values) > np.max(effective_ceiling))),
            "estimator": "robust_median",
            "shade_tolerance": float(shade_tolerance),
            "robust_window_samples": int(robust_window_samples),
            "shaded_sample_fraction": float(robust["shaded"].mean()),
            "n_unshaded_samples": int(robust["n_unshaded"]),
            "ceiling_linke_turbidity": float(ceiling_linke_turbidity),
            "ceiling_clearness_median": float(np.median(effective_ceiling)),
            "samples_above_physical_ceiling_fraction": float(above_ceiling.mean()),
            "unshaded_ratio_median": float(np.nanmedian(robust["unshaded_ratio"])),
            "clean_sky_ghi_mean_wm2": float(np.mean(ceiling_ghi)),
            # How far the UNSHADED sensor sits above what even a clean sky can
            # deliver. This is the part of the sensor-versus-model gap that is
            # NOT weather and NOT shading, and it is reported rather than
            # absorbed: injecting it would add non-physical energy to every
            # surface energy balance in the domain.
            "residual_instrument_bias_fraction": float(
                np.nanmedian(robust["unshaded_ratio"])
                / max(float(np.median(effective_ceiling)), 1e-9) - 1.0),
        }

    step = window_min / 60.0
    edges = np.arange(usable["hour"].min(), usable["hour"].max() + step, step)
    centres, values, counts = [], [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        block = usable[(usable["hour"] >= start) & (usable["hour"] < stop)]
        if block.empty:
            continue
        centres.append(0.5 * (start + stop))
        values.append(float(block["ratio"].quantile(quantile)))
        counts.append(int(len(block)))
    if len(centres) < 3:
        raise ValueError("solar envelope needs at least three populated windows")
    centres = np.asarray(centres, float)
    envelope = pd.Series(values).rolling(
        max(1, int(smooth_windows)), center=True, min_periods=1).median().to_numpy()
    clamped = np.clip(envelope, 0.0, float(maximum_clearness))
    return {
        "hours": centres,
        "clearness": clamped,
        "raw_clearness": np.asarray(values, float),
        "samples_per_window": np.asarray(counts, int),
        "n_usable_samples": int(len(usable)),
        "measured_swin_max_wm2": float(usable["SWin"].max()),
        "clear_ghi_max_wm2": float(usable["ghi_clear"].max()),
        "raw_clearness_min": float(np.min(values)),
        "raw_clearness_median": float(np.median(values)),
        "raw_clearness_max": float(np.max(values)),
        "clamped_windows": int(np.sum(np.asarray(values) > maximum_clearness)),
        "estimator": "window_quantile",
    }


def clearness_to_cloud_fraction(clearness, dni_clear, dhi_clear, elevation_deg,
                                *, tolerance: float = 1e-4) -> np.ndarray:
    """Invert the pipeline's own cloud adjustment for each timestep.

    ``apply_cloud_adjustment`` is monotone decreasing in cloud fraction, so a
    bisection recovers the cloud fraction whose adjusted GHI reproduces the
    fitted clearness. Going through that function (rather than writing an
    independent attenuation) keeps shortwave and the cloud-dependent sky
    longwave on one convention.
    """
    clearness = np.asarray(clearness, dtype=float)
    dni_clear = np.asarray(dni_clear, dtype=float)
    dhi_clear = np.asarray(dhi_clear, dtype=float)
    elevation_deg = np.asarray(elevation_deg, dtype=float)
    ghi_clear = np.maximum(
        dni_clear * np.sin(np.deg2rad(np.maximum(elevation_deg, 0.0))) + dhi_clear,
        1e-9)
    target = np.clip(clearness, 0.0, None) * ghi_clear

    low = np.zeros_like(target)
    high = np.ones_like(target)
    for _ in range(40):
        mid = 0.5 * (low + high)
        _dni, _dhi, ghi = apply_cloud_adjustment(
            dni_clear, dhi_clear, elevation_deg, mid)
        too_bright = ghi > target
        low = np.where(too_bright, mid, low)
        high = np.where(too_bright, high, mid)
        if np.all(high - low < tolerance):
            break
    cloud = 0.5 * (low + high)
    # Above clear sky no cloud fraction can help; report zero cloud there.
    return np.where(clearness >= 1.0, 0.0, np.clip(cloud, 0.0, 1.0))


def clearness_inversion_report(clearness, cloud, dni_clear, dhi_clear,
                               elevation_deg, *, tolerance: float = 0.02) -> dict:
    """How faithfully the cloud fraction reproduces the fitted clearness.

    The pipeline's scalar cloud adjustment has a limited dynamic range: even
    at cloud fraction 1 it retains a quarter of the direct beam and multiplies
    the diffuse by 2.2, so it cannot represent heavy overcast, and at low sun
    the diffuse boost can push GHI above clear sky. That is a property of the
    established convention, not of the envelope fit. Rather than shipping a
    forcing that silently disagrees with the fit, this reports where the two
    part company so a cloudy campaign fails loudly instead of quietly.
    """
    _dni, _dhi, ghi = apply_cloud_adjustment(dni_clear, dhi_clear,
                                             elevation_deg, cloud)
    ghi_clear = np.asarray(dni_clear, float) * np.sin(np.deg2rad(
        np.maximum(np.asarray(elevation_deg, float), 0.0))) + np.asarray(dhi_clear, float)
    daytime = ghi_clear > 50.0
    if not daytime.any():
        return {"daytime_samples": 0, "maximum_absolute_error": 0.0,
                "unreachable_samples": 0, "representable": True}
    achieved = ghi[daytime] / np.maximum(ghi_clear[daytime], 1e-9)
    requested = np.asarray(clearness, float)[daytime]
    error = np.abs(achieved - requested)
    return {
        "daytime_samples": int(daytime.sum()),
        "maximum_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "unreachable_samples": int((error > tolerance).sum()),
        "minimum_representable_clearness": float(achieved.min()),
        "representable": bool(error.max() <= tolerance),
        "note": ("the scalar cloud adjustment cannot reach clearness much "
                 "below ~0.85 of clear sky; requested values below that are "
                 "not represented in the written forcing"),
    }


def estimate_inlet_wind(measurements: pd.DataFrame, wind_field_dir: Path, *,
                        bin_min: float, quantile: float,
                        minimum_amplification: float,
                        minimum_speed_ms: float, maximum_inlet_ms: float) -> dict:
    """Invert measured pedestrian wind to the potential-flow inlet speed.

    The anemometer has the same problem as the radiometer: it walks through
    sheltered and accelerated places, so its reading is not the boundary
    condition. Here, though, the model supplies the correction directly. The
    step-3 field is linear in the inlet speed, so at any point

        local_speed(x, y, t) = U_inlet(t) * amplification(x, y)

    and the amplification at the sensor's own position is already solved. Each
    sample therefore gives one estimate ``U_inlet = WS / amplification``, and a
    robust statistic over the samples in a time bin gives the series.

    Samples are restricted to places where the amplification is near the free
    stream. Potential flow has no wakes, so in a lee it over-predicts the local
    speed and would push the inverted inlet too low; open ground is where the
    inviscid field is most trustworthy.

    IMPORTANT -- what this quantity is: the inlet speed that, through THIS
    inviscid field, reproduces the measured pedestrian-level wind. It is
    self-consistent with the model that consumes it, and it is NOT a 10 m
    meteorological wind. Potential flow has no boundary layer, so its
    "free stream" already sits at pedestrian level.
    """
    from pedestrian_flow_field import PedestrianFlowField

    field = PedestrianFlowField(wind_field_dir)
    usable = measurements.dropna(subset=["WS", "x_local_m", "y_local_m"]).copy()
    usable = usable[usable["WS"] >= minimum_speed_ms]
    if usable.empty:
        raise ValueError("no measured wind samples above the speed floor")
    amplification = field.amplification(usable["x_local_m"].to_numpy(float),
                                        usable["y_local_m"].to_numpy(float))
    usable["amplification"] = amplification

    # Prefer open ground; relax only if the walk never crossed any.
    thresholds = [minimum_amplification, 0.5, 0.3, 0.1]
    for threshold in thresholds:
        open_ground = usable[usable["amplification"] >= threshold]
        if len(open_ground) >= 20:
            break
    else:
        raise ValueError(
            "the walk has too few samples in open ground to invert an inlet "
            "wind; lower --minimum-amplification or check the wind field")
    open_ground = open_ground.copy()
    open_ground["inlet"] = open_ground["WS"] / open_ground["amplification"]

    step = bin_min / 60.0
    open_ground["bin"] = np.floor(open_ground["hour"] / step) * step + 0.5 * step
    binned = open_ground.groupby("bin").agg(
        inlet_ms=("inlet", lambda values: float(values.quantile(quantile))),
        n_samples=("inlet", "size"),
        mean_amplification=("amplification", "mean")).reset_index()
    binned["inlet_ms"] = binned["inlet_ms"].clip(0.0, maximum_inlet_ms)
    return {
        "hours": binned["bin"].to_numpy(float),
        "inlet_ms": binned["inlet_ms"].to_numpy(float),
        "samples_per_bin": binned["n_samples"].to_numpy(int),
        "amplification_threshold_used": float(threshold),
        "n_open_ground_samples": int(len(open_ground)),
        "n_samples_total": int(len(usable)),
        "measured_speed_range_ms": [float(usable["WS"].min()),
                                    float(usable["WS"].max())],
        "measured_speed_mean_ms": float(usable["WS"].mean()),
        "inlet_range_ms": [float(binned["inlet_ms"].min()),
                           float(binned["inlet_ms"].max())],
        "inlet_mean_ms": float(binned["inlet_ms"].mean()),
        "mean_amplification_at_used_samples": float(
            open_ground["amplification"].mean()),
        "wind_field": str(Path(wind_field_dir).resolve()),
        "field_reference_speed_ms": field.reference_speed_ms,
        "quantile": float(quantile),
        "caveat": ("inlet speed consistent with the inviscid step-3 field, not "
                   "a 10 m meteorological wind; a cart-mounted anemometer also "
                   "carries walking-induced apparent wind, so short-bin values "
                   "are noisy and a robust statistic is used"),
    }


def diurnal_shape(hours, *, sunrise, sunset, peak_lag_h=1.5, night_decay=3.0):
    """Normalised diurnal temperature shape phi(t) in [0, 1].

    Parton-Logan form (Parton & Logan 1981, *A model for diurnal variation in
    soil and air temperature*, Agricultural Meteorology): a sine from sunrise
    to a peak lagging solar noon, then exponential decay through the night to
    the sunrise minimum. Temperature is then the linear blend
    ``T = Tmin + (Tmax - Tmin) * phi``, so two measured anchors determine Tmin
    and Tmax exactly.

    The two shape constants are documented model choices, not fitted values;
    the campaign windows are far too short to constrain them.
    """
    hours = np.atleast_1d(np.asarray(hours, dtype=float))
    daylength = max(sunset - sunrise, 1e-6)
    nightlength = max(24.0 - daylength, 1e-6)
    phi = np.zeros_like(hours)
    day = (hours >= sunrise) & (hours <= sunset)
    phi[day] = np.sin(np.pi * (hours[day] - sunrise) / (daylength + 2.0 * peak_lag_h))
    phi_sunset = float(np.sin(np.pi * (sunset - sunrise)
                              / (daylength + 2.0 * peak_lag_h)))
    night = ~day
    elapsed = np.where(hours[night] > sunset, hours[night] - sunset,
                       hours[night] + 24.0 - sunset)
    phi[night] = phi_sunset * np.exp(-night_decay * elapsed / nightlength)
    return np.clip(phi, 0.0, 1.0)


def fit_diurnal_anchors(anchor_hours, anchor_temperature_c, *, sunrise, sunset,
                        peak_lag_h=1.5, night_decay=3.0):
    """Solve Tmin/Tmax so the diurnal shape passes through the measured anchors.

    ``T = Tmin (1 - phi) + Tmax phi`` is linear in (Tmin, Tmax), so any two or
    more anchors give a least-squares solution. One anchor cannot separate the
    two, and returns None.
    """
    hours = np.asarray(anchor_hours, dtype=float)
    values = np.asarray(anchor_temperature_c, dtype=float)
    phi = diurnal_shape(hours, sunrise=sunrise, sunset=sunset,
                        peak_lag_h=peak_lag_h, night_decay=night_decay)
    if len(hours) < 2 or np.ptp(phi) < 1e-3:
        return None
    design = np.column_stack([1.0 - phi, phi])
    solution, *_ = np.linalg.lstsq(design, values, rcond=None)
    minimum, maximum = float(solution[0]), float(solution[1])
    if not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
        return None
    residual = design @ solution - values
    return {"t_min_c": minimum, "t_max_c": maximum,
            "diurnal_range_c": maximum - minimum,
            "sunrise_h": float(sunrise), "sunset_h": float(sunset),
            "peak_lag_h": float(peak_lag_h), "night_decay": float(night_decay),
            "anchor_hours": [float(h) for h in hours],
            "anchor_temperature_c": [float(v) for v in values],
            "anchor_rmse_c": float(np.sqrt(np.mean(residual ** 2))),
            "shape_reference": "Parton & Logan (1981) diurnal form"}


def build_meteorology(measurements: pd.DataFrame, *, latitude, longitude,
                      timezone_name, date, bin_min: float, reconstruct: bool,
                      minimum_range_c: float = 4.0,
                      maximum_range_c: float = 25.0,
                      default_range_c: float = 10.0) -> pd.DataFrame:
    """Full-day weather series: measured where measured, reconstructed elsewhere."""
    frame = measurements.dropna(subset=["AirTemp"]).copy()
    step = bin_min / 60.0
    frame["bin"] = np.floor(frame["hour"] / step) * step + 0.5 * step
    grouped = frame.groupby("bin").agg(
        air_temp_C=("AirTemp", "median"),
        rh_pct=("HRel", "median"),
        wind_ms=("WS", "median"),
        dewpoint_C=("DewP", "median"),
        n_samples=("AirTemp", "size")).reset_index().rename(columns={"bin": "hour"})
    grouped["source"] = "measured"
    if not reconstruct:
        return grouped.sort_values("hour").reset_index(drop=True)

    # Sunrise/sunset bound the diurnal shape.
    probe = np.arange(0.0, 24.0, 0.05)
    _d, _f, _g, elevation = clear_sky_series(
        probe, latitude=latitude, longitude=longitude,
        timezone_name=timezone_name, date=date)
    daylight = probe[elevation > 0]
    sunrise = float(daylight.min()) if len(daylight) else 6.0
    sunset = float(daylight.max()) if len(daylight) else 20.0

    # A campaign is a few short walks, not a diurnal record. Anchor the shape
    # on the median of each contiguous measured window rather than on every
    # bin, so a 40-minute walk contributes one well-determined point.
    measured_hours = grouped["hour"].to_numpy(float)
    breaks = np.where(np.diff(measured_hours) > 4 * step)[0]
    segments = np.split(np.arange(len(grouped)), breaks + 1)
    anchors = [(float(grouped.iloc[block]["hour"].median()),
                float(grouped.iloc[block]["air_temp_C"].median()))
               for block in segments if len(block)]
    fit = fit_diurnal_anchors([a for a, _t in anchors], [t for _a, t in anchors],
                              sunrise=sunrise, sunset=sunset)
    # Two short walks can sit at similar points of the diurnal shape -- and when
    # the night walk was recorded on a different date they need not belong to
    # the same day at all. Either way the amplitude is then poorly determined,
    # and an implausible range (lisbon5 solved to 0.6 K) must not become
    # forcing. Fall back to the warmest anchor plus a documented range.
    if fit is not None and not (minimum_range_c <= fit["diurnal_range_c"]
                                <= maximum_range_c):
        rejected = dict(fit)
        warmest_hour, warmest_value = max(anchors, key=lambda item: item[1])
        phi_anchor = float(diurnal_shape(
            [warmest_hour], sunrise=sunrise, sunset=sunset,
            peak_lag_h=rejected["peak_lag_h"],
            night_decay=rejected["night_decay"])[0])
        maximum = warmest_value + (1.0 - phi_anchor) * default_range_c
        fit = {"t_min_c": float(maximum - default_range_c),
               "t_max_c": float(maximum),
               "diurnal_range_c": float(default_range_c),
               "sunrise_h": float(sunrise), "sunset_h": float(sunset),
               "peak_lag_h": rejected["peak_lag_h"],
               "night_decay": rejected["night_decay"],
               "anchor_hours": rejected["anchor_hours"],
               "anchor_temperature_c": rejected["anchor_temperature_c"],
               "anchor_rmse_c": None,
               "shape_reference": rejected["shape_reference"],
               "amplitude_source": (
                   f"default range {default_range_c:g} K anchored on the "
                   f"warmest measured window; the least-squares range "
                   f"{rejected['diurnal_range_c']:.2f} K fell outside the "
                   f"plausible band [{minimum_range_c:g}, {maximum_range_c:g}] K"),
               "rejected_least_squares_fit": rejected}
    elif fit is not None:
        fit["amplitude_source"] = "least squares through the measured windows"

    measured_bins = set(np.round(measured_hours, 6))
    missing = [float(h) for h in np.arange(0.0, 24.0, step) + 0.5 * step
               if round(float(h), 6) not in measured_bins]
    if not missing or fit is None:
        grouped.attrs["diurnal_fit"] = fit
        grouped.attrs["measured_windows"] = anchors
        return grouped.sort_values("hour").reset_index(drop=True)

    # Dew point is conserved far better than relative humidity through a
    # temperature swing, so hold the measured dew point and recompute RH.
    held_dewpoint = float(grouped["dewpoint_C"].median())
    median_wind = float(grouped["wind_ms"].median())
    phi = diurnal_shape(missing, sunrise=sunrise, sunset=sunset,
                        peak_lag_h=fit["peak_lag_h"],
                        night_decay=fit["night_decay"])
    temperatures = fit["t_min_c"] + (fit["t_max_c"] - fit["t_min_c"]) * phi
    dewpoints = np.minimum(held_dewpoint, temperatures)
    rows = pd.DataFrame({
        "hour": missing, "air_temp_C": temperatures,
        "rh_pct": relative_humidity_from_dewpoint(temperatures, dewpoints),
        "wind_ms": median_wind, "dewpoint_C": dewpoints,
        "n_samples": 0, "source": "reconstructed"})
    combined = pd.concat([grouped, rows], ignore_index=True)
    combined = combined.sort_values("hour").reset_index(drop=True)
    combined.attrs["diurnal_fit"] = fit
    combined.attrs["measured_windows"] = anchors
    combined.attrs["sunrise_hour"] = sunrise
    combined.attrs["sunset_hour"] = sunset
    return combined


def _case_buildings_stl(case_dir: Path) -> Path | None:
    """The case's building mesh, from its manifest, for canopy morphology."""
    manifest = case_dir / "case.json"
    if not manifest.is_file():
        return None
    files = json.loads(manifest.read_text(encoding="utf-8")).get("files", {})
    relative = files.get("buildings_stl")
    if not relative:
        return None
    path = case_dir / relative
    return path if path.is_file() else None


def build_case_inputs(case_dir: Path, *, name: str, bin_min: float,
                      window_min: float, quantile: float, smooth_windows: int,
                      minimum_clear_ghi: float, maximum_clearness: float,
                      reconstruct: bool, activate: bool,
                      longwave_from_sensor: bool,
                      wind_field_dir: Path | None = None,
                      buildings_stl: Path | None = None,
                      sensor_wind_height_m: float = 1.0,
                      pedestrian_height_m: float = 1.1,
                      free_stream_height_m: float | None = None,
                      inlet_quantile: float = 0.5,
                      minimum_amplification: float = 0.8,
                      minimum_speed_ms: float = 0.05,
                      maximum_inlet_ms: float = 25.0,
                      minimum_range_c: float = 4.0,
                      maximum_range_c: float = 25.0,
                      default_range_c: float = 10.0,
                      estimator: str = "robust_median",
                      robust_window_samples: int = 31,
                      shade_tolerance: float = 0.12,
                      ceiling_linke_turbidity: float = 2.0) -> dict:
    """Write sensor-derived weather and solar forcing for one case."""
    case_dir = Path(case_dir)
    manifest_path = case_dir / "case.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    location = manifest["location"]
    latitude = float(location["latitude"])
    longitude = float(location["longitude"])
    timezone_name = location["timezone"]
    date = manifest["simulation_defaults"]["date"]

    measurements = load_case_measurements(case_dir)
    weather = build_meteorology(
        measurements, latitude=latitude, longitude=longitude,
        timezone_name=timezone_name, date=date, bin_min=bin_min,
        reconstruct=reconstruct, minimum_range_c=minimum_range_c,
        maximum_range_c=maximum_range_c, default_range_c=default_range_c)

    # Inlet wind: invert the measured pedestrian wind through the solved
    # potential-flow amplification (linearity), so 05b scales the field by a
    # time-varying boundary speed instead of one constant.
    inlet = None
    inlet_note = "not estimated"
    if wind_field_dir is not None and Path(wind_field_dir).is_dir():
        try:
            inlet = estimate_inlet_wind(
                measurements, Path(wind_field_dir), bin_min=bin_min,
                quantile=inlet_quantile,
                minimum_amplification=minimum_amplification,
                minimum_speed_ms=minimum_speed_ms,
                maximum_inlet_ms=maximum_inlet_ms)
            inlet_note = "inverted from measured wind via step-3 amplification"
        except (ValueError, FileNotFoundError) as exc:
            inlet_note = f"not estimated: {exc}"
    elif wind_field_dir is not None:
        inlet_note = (f"not estimated: no solved wind field at {wind_field_dir}; "
                      "run pipeline step 3 for this case first")

    envelope = solar_clearness_envelope(
        measurements, latitude=latitude, longitude=longitude,
        timezone_name=timezone_name, date=date, window_min=window_min,
        quantile=quantile, smooth_windows=smooth_windows,
        minimum_clear_ghi=minimum_clear_ghi,
        maximum_clearness=maximum_clearness,
        estimator=estimator, robust_window_samples=robust_window_samples,
        shade_tolerance=shade_tolerance,
        ceiling_linke_turbidity=ceiling_linke_turbidity)
    if envelope.get("estimator") == "robust_median":
        print(f"  Solar envelope (robust rolling median, window "
              f"{envelope['robust_window_samples']} samples, shade tolerance "
              f"{envelope['shade_tolerance']:.0%}):")
        print(f"    shaded samples rejected      : "
              f"{envelope['shaded_sample_fraction']:.1%}")
        print(f"    unshaded clearness (median)  : "
              f"{envelope['unshaded_ratio_median']:.3f} x climatological clear sky")
        print(f"    clean-sky ceiling (Linke {envelope['ceiling_linke_turbidity']:.1f}): "
              f"{envelope['ceiling_clearness_median']:.3f}")
        residual = envelope["residual_instrument_bias_fraction"]
        if residual > 0.02:
            print(f"    NOTE: the unshaded sensor sits {residual:.1%} above what "
                  f"even a clean sky can deliver.")
            print(f"          That residual is instrument bias (mount tilt or "
                  f"calibration), not weather, so it is NOT")
            print(f"          added to the domain forcing. Lower "
                  f"--ceiling-linke-turbidity only if the sensor")
            print(f"          calibration has been independently verified.")

    # Solar components on a regular full-day grid.
    grid = np.arange(0.0, 24.0, bin_min / 60.0)
    dni_clear, dhi_clear, ghi_clear, elevation = clear_sky_series(
        grid, latitude=latitude, longitude=longitude,
        timezone_name=timezone_name, date=date)
    clearness = np.interp(grid, envelope["hours"], envelope["clearness"],
                          left=envelope["clearness"][0],
                          right=envelope["clearness"][-1])

    # ------------------------------------------------------------------
    # BASELINE TURBIDITY
    #
    # The forcing path can only ATTENUATE: cloud fraction reduces irradiance and
    # nothing in it can represent an atmosphere CLEANER than the monthly Linke
    # climatology. So a fitted clearness above 1 was silently unreachable --
    # the cloud inversion saturated at zero cloud and the written forcing came
    # out at exactly the climatological clear sky, which is why the model sat
    # near 875 W/m2 while the unshaded sensor read ~1000.
    #
    # The fix is to move the BASELINE rather than to fight the cloud model: if
    # the day was cleaner than climatology, recompute the clear-sky components
    # at the turbidity that actually reproduces the fitted envelope, and let
    # cloud handle only genuine departures BELOW that baseline. The search is
    # bounded by ceiling_linke_turbidity, so the baseline can never be brighter
    # than a clean sky.
    # ------------------------------------------------------------------
    baseline_turbidity = None
    daylight = ghi_clear > 50.0
    excess = float(np.median(clearness[daylight])) if daylight.any() else 1.0
    if excess > 1.005:
        climatological = float(np.median(
            pvlib.clearsky.lookup_linke_turbidity(
                pd.DatetimeIndex([pd.Timestamp(date)]).tz_localize(
                    timezone_name, ambiguous=True, nonexistent="shift_forward"),
                latitude, longitude).to_numpy(float)))
        candidates = np.linspace(float(ceiling_linke_turbidity),
                                 climatological, 41)
        best, best_error = climatological, np.inf
        for candidate in candidates:
            trial = clean_sky_ceiling(
                grid, latitude=latitude, longitude=longitude,
                timezone_name=timezone_name, date=date,
                linke_turbidity=candidate)
            ratio = np.median(trial[daylight] / np.maximum(ghi_clear[daylight], 1.0))
            error = abs(ratio - excess)
            if error < best_error:
                best, best_error = float(candidate), float(error)
        baseline_turbidity = best
        dni_clear, dhi_clear, ghi_clear_new, elevation_new = (
            clear_sky_at_turbidity(grid, latitude=latitude, longitude=longitude,
                                   timezone_name=timezone_name, date=date,
                                   linke_turbidity=baseline_turbidity))
        achieved = float(np.median(ghi_clear_new[daylight]
                                   / np.maximum(ghi_clear[daylight], 1.0)))
        ghi_clear, elevation = ghi_clear_new, elevation_new
        # Clearness is now expressed against the NEW baseline, so what remains
        # for the cloud model is only the part it can actually represent.
        clearness = np.clip(clearness / max(achieved, 1e-9), 0.0, 1.0)
        print(f"    baseline turbidity refitted  : Linke {baseline_turbidity:.2f} "
              f"(climatology {climatological:.2f}) -> clear-sky brighter by "
              f"{achieved - 1.0:+.1%}")
    cloud = clearness_to_cloud_fraction(clearness, dni_clear, dhi_clear, elevation)
    inversion = clearness_inversion_report(clearness, cloud, dni_clear,
                                           dhi_clear, elevation)
    if not inversion["representable"]:
        print(f"  WARNING: {inversion['unreachable_samples']} timestep(s) "
              f"requested a clearness the scalar cloud adjustment cannot "
              f"reach (max error {inversion['maximum_absolute_error']:.3f}); "
              f"the written forcing is brighter than the fitted envelope. "
              f"{inversion['note']}")
    dni, dhi, ghi = apply_cloud_adjustment(dni_clear, dhi_clear, elevation, cloud)
    components = pd.DataFrame({
        "hour": grid, "DNI_Wm2": dni, "DHI_Wm2": dhi, "GHI_Wm2": ghi,
        "cloud_fraction": cloud, "fitted_clearness": clearness,
        "solar_elevation_deg": elevation})

    longwave_note = "not taken from the sensor"
    if longwave_from_sensor and "LWin" in measurements:
        # The up-facing pyrgeometer sees sky AND any wall above the horizon, so
        # the LOWER envelope is the closest available estimate of unobstructed
        # sky longwave. In a deep canyon even that retains some wall signal.
        lw = measurements.dropna(subset=["LWin"])
        step = window_min / 60.0
        binned = lw.assign(bin=np.floor(lw["hour"] / step) * step + 0.5 * step)
        low = binned.groupby("bin")["LWin"].quantile(1.0 - quantile)
        components["LWin_Wm2"] = np.interp(
            grid, low.index.to_numpy(float), low.to_numpy(float),
            left=float(low.iloc[0]), right=float(low.iloc[-1]))
        longwave_note = (f"lower envelope (quantile {1.0 - quantile:.2f}) of "
                         "measured LWin; retains wall signal in deep canyons")

    # ------------------------------------------------------------------
    # FREE-STREAM REFERENCE WIND FROM THE EXPERIMENT
    #
    # An "a + b*U" convection correlation was calibrated against the
    # UNDISTURBED approach velocity of a wind tunnel. Feeding it the sheltered
    # in-canopy wind the cart measured is a category error that under-predicts
    # the coefficient by roughly a factor of two.
    #
    # This lifts the measurement to a free-stream reference through the urban
    # canopy profile, using morphology derived from the case's OWN buildings.
    # It is a boundary condition and it comes from the experiment -- never from
    # the potential-flow solution it goes on to drive.
    #
    # A plain log law cannot do this: the 1 m sensor sits well below the
    # displacement height, inside the canopy where no log layer exists. Hence
    # the two-layer exponential/logarithmic profile in wind_profile.py.
    # ------------------------------------------------------------------
    morphology = None
    if buildings_stl is not None and Path(buildings_stl).is_file():
        try:
            fluid_fraction = None
            domain_area = None
            if wind_field_dir is not None and Path(wind_field_dir).is_dir():
                fraction_path = Path(wind_field_dir) / "cell_fluid_fraction.npy"
                metadata_path = Path(wind_field_dir) / "potential_flow_metadata.json"
                if fraction_path.is_file():
                    fluid_fraction = np.load(fraction_path)
                if metadata_path.is_file():
                    grid = json.loads(
                        metadata_path.read_text(encoding="utf-8"))["grid"]
                    domain_area = (grid["nx"] * grid["ny"]
                                   * grid["spacing_m"] ** 2)
            morphology = wp.morphology_from_geometry(
                str(buildings_stl), cell_fluid_fraction=fluid_fraction,
                domain_area_m2=domain_area)
            reference_height = (float(free_stream_height_m)
                                if free_stream_height_m
                                else 2.0 * morphology.building_height_m)
            measured = weather["wind_ms"].to_numpy(float)
            weather["wind_freestream_ms"] = wp.free_stream_speed(
                measured, sensor_wind_height_m, morphology, reference_height)
            print(f"  Canopy profile: {morphology.describe()}")
            print(f"    free-stream reference at {reference_height:.0f} m "
                  f"(2H): {weather['wind_freestream_ms'].mean():.2f} m/s "
                  f"from a measured {measured.mean():.2f} m/s at "
                  f"{sensor_wind_height_m:.1f} m "
                  f"(x{weather['wind_freestream_ms'].mean() / max(measured.mean(), 1e-9):.2f})")
            print(f"    -> McAdams h = 5.7 + 3.8U would be "
                  f"{5.7 + 3.8 * weather['wind_freestream_ms'].mean():.1f} "
                  "W/m2K on the free stream, against "
                  f"{5.7 + 3.8 * measured.mean():.1f} on the sheltered wind")
        except (wp.WindProfileError, OSError, ValueError) as error:
            print(f"  canopy wind profile skipped ({error})")
            morphology = None

    # The inlet series is written as its OWN column. `wind_ms` stays the
    # measured pedestrian-level wind so nothing that already consumes it
    # changes meaning; 05b picks up `wind_inlet_ms` when present.
    if inlet is not None:
        weather["wind_inlet_ms"] = np.interp(
            weather["hour"].to_numpy(float), inlet["hours"], inlet["inlet_ms"],
            left=inlet["inlet_ms"][0], right=inlet["inlet_ms"][-1])
    weather_path = case_dir / "weather" / f"{name}.csv"
    components_path = case_dir / "config" / f"{name}_solar_components.csv"
    config_path = case_dir / "config" / f"{name}.json"
    provenance_path = case_dir / "weather" / f"{name}_provenance.json"
    weather_path.parent.mkdir(parents=True, exist_ok=True)
    components_path.parent.mkdir(parents=True, exist_ok=True)

    weather.to_csv(weather_path, index=False)
    components.to_csv(components_path, index=False)
    config_path.write_text(json.dumps({
        "mode": "components_csv",
        "source_file": components_path.name,
        "comment": ("Time-dependent solar forcing fitted to the upper envelope "
                    "of the mobile shortwave measurements. Local shade is still "
                    "ray traced; only the atmosphere comes from here."),
    }, indent=2) + "\n", encoding="utf-8")

    fit = weather.attrs.get("diurnal_fit")
    measured_rows = int((weather["source"] == "measured").sum())
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": manifest.get("case_id"),
        "source_measurements": str((case_dir / MEASUREMENT_RELATIVE).resolve()),
        "location": {"latitude": latitude, "longitude": longitude,
                     "timezone": timezone_name, "date": date},
        "meteorology": {
            "bin_minutes": bin_min,
            "measured_rows": measured_rows,
            "reconstructed_rows": int(len(weather) - measured_rows),
            "measured_window_h": [float(weather.loc[weather["source"] == "measured",
                                                    "hour"].min()),
                                  float(weather.loc[weather["source"] == "measured",
                                                    "hour"].max())],
            "measured_windows_hour_and_temperature":
                weather.attrs.get("measured_windows"),
            "reconstruction": ("Parton-Logan diurnal shape with Tmin/Tmax solved "
                               "from the measured window medians; dew point held "
                               "and RH recomputed; wind held at the measured "
                               "median" if fit else "not applied (needs at least "
                               "two separated measured windows)"),
            "diurnal_fit": fit,
            "sunrise_hour": weather.attrs.get("sunrise_hour"),
            "sunset_hour": weather.attrs.get("sunset_hour"),
            "air_temperature_range_c": [float(weather["air_temp_C"].min()),
                                        float(weather["air_temp_C"].max())],
        },
        "solar": {
            "method": ("unshaded-envelope clearness of mobile SWin, inverted "
                       "through radiation_forcing.apply_cloud_adjustment on a "
                       "clear-sky baseline refitted for atmospheric clarity"),
            # Everything the estimator reported: which method ran, how much of
            # the walk it judged shaded, and how far the unshaded sensor sat
            # above a physically clean sky.
            "envelope_diagnostics": {key: value for key, value in envelope.items()
                                     if not isinstance(value, np.ndarray)},
            "window_minutes": window_min,
            "envelope_quantile": quantile,
            "smoothing_windows": smooth_windows,
            "minimum_clear_sky_ghi_wm2": minimum_clear_ghi,
            "maximum_clearness": maximum_clearness,
            "_canopy_wind_profile": (morphology.as_metadata()
                                     if morphology is not None else None),
            "baseline_linke_turbidity": baseline_turbidity,
            "baseline_turbidity_note": (
                "None means the climatological turbidity was kept. A value "
                "means the fitted envelope exceeded the climatological clear "
                "sky and the BASELINE was rebrightened to that turbidity, "
                "because the cloud model can only attenuate and would "
                "otherwise have clamped the forcing at climatological clear "
                "sky."),
            "usable_daytime_samples": envelope["n_usable_samples"],
            "raw_clearness_min": envelope["raw_clearness_min"],
            "raw_clearness_median": envelope["raw_clearness_median"],
            "raw_clearness_max": envelope["raw_clearness_max"],
            "windows_clamped_at_maximum": envelope["clamped_windows"],
            "measured_swin_max_wm2": envelope["measured_swin_max_wm2"],
            "clear_sky_ghi_max_wm2": envelope["clear_ghi_max_wm2"],
            "fitted_clearness_range": [float(clearness.min()), float(clearness.max())],
            "cloud_fraction_range": [float(cloud.min()), float(cloud.max())],
            "clearness_inversion": inversion,
            "note": ("Measured SWin above the clear-sky model is clamped at "
                     "maximum_clearness rather than absorbed into the forcing; "
                     "a persistent excess indicates sensor tilt, wall "
                     "reflection into the dome, or cloud-edge enhancement, "
                     "none of which are atmospheric transmission."),
        },
        # The series itself lives in the weather CSV; provenance keeps the
        # scalars that explain how it was derived.
        "inlet_wind": ({key: value for key, value in
                        dict(inlet, note=inlet_note).items()
                        if not isinstance(value, np.ndarray)}
                       if inlet is not None else {"note": inlet_note}),
        "longwave": longwave_note,
        "outputs": {"weather_csv": str(weather_path.resolve()),
                    "solar_components_csv": str(components_path.resolve()),
                    "radiation_forcing_config": str(config_path.resolve())},
        "activated_in_case_json": bool(activate),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n",
                               encoding="utf-8")

    if activate:
        manifest.setdefault("files", {})
        manifest["files"]["weather_csv"] = str(
            weather_path.relative_to(case_dir).as_posix())
        manifest["files"]["radiation_forcing_config"] = str(
            config_path.relative_to(case_dir).as_posix())
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TREC-Route weather and time-dependent solar forcing "
                    "from mobile sensor measurements")
    parser.add_argument("--sensor-wind-height-m", type=float, default=1.0,
                        help="Height of the cart anemometer above local ground. "
                             "The source data does not record it; this is the "
                             "assumed value and it is stamped into the "
                             "provenance.")
    parser.add_argument("--free-stream-height-m", type=float, default=None,
                        help="Reference height for the free-stream wind that "
                             "drives an a+b*U convection correlation. Default "
                             "2H (twice the canopy height), the usual blending "
                             "height at which flow loses memory of individual "
                             "buildings.")
    parser.add_argument("--no-free-stream-wind", action="store_true",
                        help="Skip the canopy-profile lift; no "
                             "wind_freestream_ms column is written and "
                             "convection falls back to the sheltered wind.")
    parser.add_argument("--case", action="append", default=None,
                        help="Case directory or name (repeatable). Default: "
                             "every input case that has mobile measurements.")
    parser.add_argument("--input-root", type=Path, default=ROOT / "input")
    parser.add_argument("--name", default="weather_from_sensors",
                        help="Base name for the generated files")
    parser.add_argument("--bin-minutes", type=float, default=10.0,
                        help="Output time resolution (default 10 min, matching "
                             "the pipeline's DT_MIN)")
    parser.add_argument("--window-minutes", type=float, default=10.0,
                        help="Rolling window for the shortwave upper envelope")
    parser.add_argument("--envelope-quantile", type=float, default=0.95,
                        help="High quantile taken within each window (default "
                             "0.95; a quantile rather than the maximum so one "
                             "spurious spike cannot set the forcing)")
    parser.add_argument("--smooth-windows", type=int, default=3,
                        help="Rolling median width applied to the envelope")
    parser.add_argument("--minimum-clear-ghi", type=float, default=80.0,
                        help="Ignore samples whose clear-sky GHI is below this; "
                             "ratios are meaningless at very low sun")
    parser.add_argument("--solar-estimator",
                        choices=["robust_median", "window_quantile"],
                        default="robust_median",
                        help="How the unshaded solar envelope is extracted. "
                             "'robust_median' (default) uses a centred rolling "
                             "median that rejects shade dips in both directions "
                             "and is not dragged down by a mostly shaded "
                             "window; 'window_quantile' is the previous "
                             "fixed-quantile-per-window behaviour.")
    parser.add_argument("--robust-window-samples", type=int, default=31,
                        help="Rolling-median width in SAMPLES for the robust "
                             "estimator (default 31; at 5-10 s sampling that is "
                             "roughly 3-5 minutes of walking).")
    parser.add_argument("--shade-tolerance", type=float, default=0.12,
                        help="A sample more than this fraction BELOW the local "
                             "rolling median is treated as shaded and excluded "
                             "from the solar envelope (default 0.12).")
    parser.add_argument("--ceiling-linke-turbidity", type=float, default=2.0,
                        help="Linke turbidity defining the CLEAN-atmosphere "
                             "clear-sky ceiling (default 2.0, near the "
                             "Rayleigh limit). The envelope may exceed the "
                             "climatological clear sky up to this bound, but "
                             "not beyond it: a sustained excess over a clean "
                             "sky is instrument error, not weather.")
    parser.add_argument("--maximum-clearness", type=float, default=1.0,
                        help="Cap on fitted clearness (default 1.0 = the "
                             "clear-sky model). Raising it above 1 adopts the "
                             "sensor's over-reading as atmospheric truth; do "
                             "that only as a deliberate sensitivity test.")
    parser.add_argument("--no-reconstruct-missing-hours", action="store_true",
                        help="Write only the measured window. The pipeline will "
                             "then interpolate a straight chord across the "
                             "unmeasured night, which is what this tool exists "
                             "to avoid.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "run_output",
                        help="Where solved step-3 wind fields live, used to "
                             "invert the measured wind to an inlet speed")
    parser.add_argument("--inlet-quantile", type=float, default=0.5,
                        help="Robust statistic per time bin for the inverted "
                             "inlet wind (default 0.5 = median)")
    parser.add_argument("--minimum-amplification", type=float, default=0.8,
                        help="Only invert samples where the potential-flow "
                             "amplification is at least this (default 0.8): "
                             "potential flow has no wakes, so a lee reading "
                             "would push the inlet too low")
    parser.add_argument("--maximum-inlet-ms", type=float, default=25.0,
                        help="Cap on the inverted inlet speed")
    parser.add_argument("--no-inlet-wind", action="store_true",
                        help="Skip the inlet-wind inversion entirely")
    parser.add_argument("--minimum-diurnal-range-c", type=float, default=4.0,
                        help="Reject a fitted diurnal range below this and use "
                             "the default range instead (default 4 K)")
    parser.add_argument("--maximum-diurnal-range-c", type=float, default=25.0,
                        help="Reject a fitted diurnal range above this (default 25 K)")
    parser.add_argument("--default-diurnal-range-c", type=float, default=10.0,
                        help="Diurnal range used when the measured windows "
                             "cannot determine it (default 10 K)")
    parser.add_argument("--longwave-from-sensor", action="store_true",
                        help="Also take downwelling longwave from the LOWER "
                             "envelope of measured LWin. Off by default: in a "
                             "canyon even the lower envelope retains wall "
                             "signal, so it is not pure sky longwave.")
    parser.add_argument("--no-activate", action="store_true",
                        help="Write the files but do not repoint case.json at "
                             "them (the originals are never overwritten).")
    return parser.parse_args()


def discover_cases(input_root: Path, requested) -> list[Path]:
    if requested:
        cases = []
        for item in requested:
            path = Path(item)
            if not path.is_dir():
                path = input_root / item
            if not (path / "case.json").is_file():
                raise FileNotFoundError(f"not a case directory: {path}")
            cases.append(path)
        return cases
    return sorted(
        entry for entry in input_root.iterdir()
        if entry.is_dir() and (entry / "case.json").is_file()
        and (entry / MEASUREMENT_RELATIVE).is_file())


def cases_with_measurements(input_root: Path) -> list[str]:
    """Names of the cases this tool can actually work on."""
    if not input_root.is_dir():
        return []
    return sorted(entry.name for entry in input_root.iterdir()
                  if entry.is_dir() and (entry / "case.json").is_file()
                  and (entry / MEASUREMENT_RELATIVE).is_file())


def main() -> int:
    args = parse_args()
    requested = discover_cases(args.input_root, args.case)
    # This tool derives forcing FROM a measurement campaign, so a case without
    # one is not an error in the case -- it simply has nothing to derive from.
    # Say that plainly instead of raising, because this runs from a UI button
    # where any case can be selected.
    cases = [case for case in requested
             if (case / MEASUREMENT_RELATIVE).is_file()]
    skipped = [case for case in requested if case not in cases]
    for case in skipped:
        print(f"[{case.name}] skipped: no mobile measurements at "
              f"{case / MEASUREMENT_RELATIVE}")
    if not cases:
        available = cases_with_measurements(args.input_root)
        print("\nNothing to do: this tool builds weather and solar forcing "
              "FROM a mobile measurement campaign, and none of the selected "
              "case(s) have one.")
        if available:
            print("Cases with measurements: " + ", ".join(available))
            print("Select one of those in Step 1, or pass --case <name>.")
        else:
            print(f"No case under {args.input_root} has "
                  f"{MEASUREMENT_RELATIVE}.")
        return 1
    print(f"Building sensor-derived forcing for {len(cases)} case(s): "
          f"{', '.join(case.name for case in cases)}")
    for case in cases:
        print(f"\n[{case.name}]")
        provenance = build_case_inputs(
            case, name=args.name, bin_min=args.bin_minutes,
            window_min=args.window_minutes, quantile=args.envelope_quantile,
            smooth_windows=args.smooth_windows,
            minimum_clear_ghi=args.minimum_clear_ghi,
            maximum_clearness=args.maximum_clearness,
            estimator=args.solar_estimator,
            robust_window_samples=args.robust_window_samples,
            shade_tolerance=args.shade_tolerance,
            ceiling_linke_turbidity=args.ceiling_linke_turbidity,
            reconstruct=not args.no_reconstruct_missing_hours,
            activate=not args.no_activate,
            longwave_from_sensor=args.longwave_from_sensor,
            wind_field_dir=(None if args.no_inlet_wind else
                            args.output_root / case.name / "pedestrian_wind"),
            buildings_stl=(None if args.no_free_stream_wind else
                           _case_buildings_stl(case)),
            sensor_wind_height_m=args.sensor_wind_height_m,
            free_stream_height_m=args.free_stream_height_m,
            inlet_quantile=args.inlet_quantile,
            minimum_amplification=args.minimum_amplification,
            maximum_inlet_ms=args.maximum_inlet_ms,
            minimum_range_c=args.minimum_diurnal_range_c,
            maximum_range_c=args.maximum_diurnal_range_c,
            default_range_c=args.default_diurnal_range_c)
        met = provenance["meteorology"]
        sol = provenance["solar"]
        window = met["measured_window_h"]
        print(f"  meteorology: {met['measured_rows']} measured rows "
              f"({window[0]:.2f}-{window[1]:.2f} h), "
              f"{met['reconstructed_rows']} reconstructed; "
              f"Ta {met['air_temperature_range_c'][0]:.1f}.."
              f"{met['air_temperature_range_c'][1]:.1f} C")
        if met["diurnal_fit"]:
            fit = met["diurnal_fit"]
            rmse = fit.get("anchor_rmse_c")
            print(f"    diurnal fit: Tmin {fit['t_min_c']:.1f} C at sunrise "
                  f"{fit['sunrise_h']:.2f} h, Tmax {fit['t_max_c']:.1f} C "
                  f"(range {fit['diurnal_range_c']:.1f} K, "
                  + (f"anchor RMSE {rmse:.2f} C" if rmse is not None
                     else "amplitude defaulted")
                  + f", {len(fit['anchor_hours'])} window(s))")
            if fit.get("rejected_least_squares_fit"):
                print(f"    NOTE: {fit['amplitude_source']}")
        print(f"  solar: clearness {sol['fitted_clearness_range'][0]:.2f}.."
              f"{sol['fitted_clearness_range'][1]:.2f}, cloud "
              f"{sol['cloud_fraction_range'][0]:.2f}.."
              f"{sol['cloud_fraction_range'][1]:.2f} from "
              f"{sol['usable_daytime_samples']} samples "
              f"({sol['windows_clamped_at_maximum']} window(s) clamped)")
        wind = provenance["inlet_wind"]
        if "inlet_range_ms" in wind:
            print(f"  inlet wind: {wind['inlet_range_ms'][0]:.2f}.."
                  f"{wind['inlet_range_ms'][1]:.2f} m/s (mean "
                  f"{wind['inlet_mean_ms']:.2f}) inverted from measured "
                  f"{wind['measured_speed_mean_ms']:.2f} m/s mean using "
                  f"{wind['n_open_ground_samples']} open-ground samples "
                  f"(amplification >= {wind['amplification_threshold_used']:.2f})")
        else:
            print(f"  inlet wind: {wind['note']}")
        print(f"  activated in case.json: {provenance['activated_in_case_json']}")
    print("\nRe-run the pipeline from step 4 for these cases to use the new "
          "forcing (the surface spin-up depends on the full-day weather).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
