"""Case-driven atmospheric radiation forcing for TREC-Route.

The module keeps atmospheric forcing separate from route geometry.  In
particular, radiation measured by a mobile station is *not* replayed point by
point as domain-wide forcing because it already contains local shade and
reflection.  The optional ``route_upper_envelope_cloud`` mode uses only the
upper envelope of the mobile shortwave observations to estimate one
case/session cloud attenuation.  Local shade is still calculated by the ray
tracer.

Configuration modes
-------------------
``clear_sky``
    Use pvlib clear sky plus the configured scalar cloud fraction.
``components_csv``
    Read independent DNI/DHI/GHI station forcing.  Optional LWin and cloud
    fraction columns are retained.
``route_upper_envelope_cloud``
    Compare high mobile SWin/clear-sky-GHI ratios at sufficiently high solar
    elevation, infer one bounded cloud fraction, and apply it to the complete
    model day.  This is a documented forcing sensitivity estimate, not a
    measured pointwise solar boundary condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RadiationForcing:
    """Resolved radiation arrays and their reproducibility metadata."""

    dni_wm2: np.ndarray
    dhi_wm2: np.ndarray
    ghi_wm2: np.ndarray
    cloud_fraction: np.ndarray
    lwin_wm2: np.ndarray
    source: str
    metadata: dict[str, Any]


def apply_cloud_adjustment(
    dni_clear: np.ndarray,
    dhi_clear: np.ndarray,
    elevation_deg: np.ndarray,
    cloud_fraction: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the established TREC-Route scalar cloud adjustment."""
    cloud = np.clip(np.asarray(cloud_fraction, dtype=float), 0.0, 1.0)
    elevation = np.asarray(elevation_deg, dtype=float)
    sin_el = np.sin(np.deg2rad(np.maximum(elevation, 0.0)))
    direct_factor = np.clip(1.0 - 0.75 * cloud ** 3.4, 0.0, 1.0)
    dni = np.asarray(dni_clear, dtype=float) * direct_factor
    lost_direct_horizontal = (
        np.asarray(dni_clear, dtype=float) * sin_el * (1.0 - direct_factor)
    )
    dhi = np.asarray(dhi_clear, dtype=float) * (1.0 + 1.2 * cloud) \
        + 0.6 * lost_direct_horizontal
    ghi = dni * sin_el + dhi
    night = elevation <= 0.0
    return (
        np.where(night, 0.0, dni),
        np.where(night, 0.0, dhi),
        np.where(night, 0.0, ghi),
    )


def _decimal_hours(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(
        index.hour + index.minute / 60.0 + index.second / 3600.0,
        dtype=float,
    )


def _periodic_interp(source_hours: np.ndarray, values: np.ndarray,
                     target_hours: np.ndarray) -> np.ndarray:
    order = np.argsort(source_hours)
    return np.interp(target_hours, source_hours[order], values[order], period=24.0)


def _column(frame: pd.DataFrame, aliases: tuple[str, ...], *, required: bool,
            label: str) -> np.ndarray | None:
    lower = {str(name).lower(): name for name in frame.columns}
    for alias in aliases:
        if alias.lower() in lower:
            values = pd.to_numeric(frame[lower[alias.lower()]], errors="coerce").to_numpy(float)
            if required and not np.isfinite(values).all():
                raise ValueError(f"{label} contains non-finite values")
            return values
    if required:
        raise ValueError(f"missing {label}; accepted columns: {', '.join(aliases)}")
    return None


def _read_component_csv(path: Path, model_times: pd.DatetimeIndex) -> RadiationForcing:
    if not path.is_file():
        raise FileNotFoundError(f"radiation component CSV not found: {path}")
    frame = pd.read_csv(path)
    lower = {str(name).lower(): name for name in frame.columns}
    if "hour" in lower:
        source_hours = pd.to_numeric(frame[lower["hour"]], errors="coerce").to_numpy(float)
    elif "time" in lower:
        parsed = pd.to_datetime(frame[lower["time"]], errors="raise")
        source_hours = np.asarray(
            parsed.dt.hour + parsed.dt.minute / 60.0 + parsed.dt.second / 3600.0,
            dtype=float,
        )
    else:
        raise ValueError("radiation component CSV requires an 'hour' or 'time' column")
    dni = _column(frame, ("DNI_Wm2", "DNI", "Kdir"), required=True, label="DNI")
    dhi = _column(frame, ("DHI_Wm2", "DHI", "Kdiff"), required=True, label="DHI")
    ghi = _column(frame, ("GHI_Wm2", "GHI", "Kdn"), required=True, label="GHI")
    if len(frame) < 2 or not np.isfinite(source_hours).all():
        raise ValueError("radiation component CSV needs at least two finite time rows")
    if min(np.nanmin(dni), np.nanmin(dhi), np.nanmin(ghi)) < 0:
        raise ValueError("radiation component CSV contains negative shortwave irradiance")
    target_hours = _decimal_hours(model_times)
    cloud = _column(frame, ("cloud_fraction", "cloud_cover_fraction"),
                    required=False, label="cloud fraction")
    lwin = _column(frame, ("LWin_Wm2", "LWin", "longwave_down_Wm2"),
                   required=False, label="downwelling longwave")
    cloud_out = (np.full(len(model_times), np.nan) if cloud is None else
                 _periodic_interp(source_hours, cloud, target_hours))
    lwin_out = (np.full(len(model_times), np.nan) if lwin is None else
                _periodic_interp(source_hours, lwin, target_hours))
    if np.isfinite(cloud_out).any() and (
            np.nanmin(cloud_out) < 0 or np.nanmax(cloud_out) > 1):
        raise ValueError("cloud fraction must lie in [0, 1]")
    if np.isfinite(lwin_out).any() and np.nanmin(lwin_out) < 0:
        raise ValueError("downwelling longwave must be non-negative")
    return RadiationForcing(
        _periodic_interp(source_hours, dni, target_hours),
        _periodic_interp(source_hours, dhi, target_hours),
        _periodic_interp(source_hours, ghi, target_hours),
        cloud_out, lwin_out, str(path.resolve()),
        {"mode": "components_csv", "source_file": str(path.resolve()),
         "n_source_rows": int(len(frame))},
    )


def _load_mobile_envelope(config: dict[str, Any], config_path: Path,
                          model_times: pd.DatetimeIndex,
                          clear_ghi: np.ndarray,
                          elevation_deg: np.ndarray,
                          clear_dni: np.ndarray,
                          clear_dhi: np.ndarray) -> RadiationForcing:
    source_value = config.get("source_file")
    if not source_value:
        raise ValueError("route_upper_envelope_cloud requires source_file")
    source = Path(source_value)
    if not source.is_absolute():
        source = (config_path.parent / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"mobile radiation diagnostic file not found: {source}")
    frame = pd.read_csv(source)
    time_column = str(config.get("time_column", "timestamp_local_refined"))
    sw_column = str(config.get("shortwave_column", "SWin"))
    if time_column not in frame or sw_column not in frame:
        raise ValueError(
            f"{source} requires configured columns {time_column!r} and {sw_column!r}")
    timestamps_utc = pd.to_datetime(
        frame[time_column], errors="coerce", utc=True, format="mixed")
    shortwave = pd.to_numeric(frame[sw_column], errors="coerce").to_numpy(float)
    model_local = (model_times if model_times.tz is not None
                   else model_times.tz_localize("UTC"))
    model_utc = model_local.tz_convert("UTC")
    timestamps_local = timestamps_utc.dt.tz_convert(model_local.tz)
    model_date = model_local[0].date()
    same_date = np.asarray(timestamps_local.dt.date == model_date)
    obs_hours = np.asarray(
        timestamps_local.dt.hour + timestamps_local.dt.minute / 60.0
        + timestamps_local.dt.second / 3600.0, dtype=float,
    )
    model_hours_local = _decimal_hours(model_local)
    clear_at_obs = _periodic_interp(model_hours_local, np.asarray(clear_ghi, float), obs_hours)
    elevation_at_obs = _periodic_interp(
        model_hours_local, np.asarray(elevation_deg, float), obs_hours)
    minimum_elevation = float(config.get("minimum_solar_elevation_deg", 10.0))
    minimum_samples = int(config.get("minimum_samples", 5))
    quantile = float(config.get("upper_quantile", 0.95))
    if not 0.5 <= quantile <= 1.0:
        raise ValueError("upper_quantile must lie in [0.5, 1.0]")
    valid = (same_date & np.isfinite(shortwave) & (shortwave >= 0)
             & np.isfinite(clear_at_obs) & (clear_at_obs > 50)
             & (elevation_at_obs >= minimum_elevation))
    warnings: list[str] = []
    if int(valid.sum()) < minimum_samples:
        fallback_cloud = float(config.get("fallback_cloud_fraction", 0.0))
        if not 0.0 <= fallback_cloud <= 1.0:
            raise ValueError("fallback_cloud_fraction must lie in [0, 1]")
        warning = (
            f"only {int(valid.sum())} usable mobile shortwave samples on "
            f"{model_date}; at least {minimum_samples} are required. Using "
            f"fallback cloud fraction {fallback_cloud:.3f}.")
        dni, dhi, ghi = apply_cloud_adjustment(
            clear_dni, clear_dhi, elevation_deg, fallback_cloud)
        return RadiationForcing(
            dni, dhi, ghi, np.full(len(model_times), fallback_cloud),
            np.full(len(model_times), np.nan),
            "clear_sky_cloud_fallback_insufficient_mobile_envelope",
            {
                "mode": "route_upper_envelope_cloud",
                "status": "insufficient_samples_fallback",
                "source_file": str(source),
                "time_column": time_column,
                "shortwave_column": sw_column,
                "model_date": str(model_date),
                "n_source_rows": int(len(frame)),
                "n_usable_daytime_samples": int(valid.sum()),
                "minimum_samples": minimum_samples,
                "fallback_cloud_fraction": fallback_cloud,
                "interpretation": (
                    "Mobile data were insufficient for an upper-envelope "
                    "estimate; no pointwise mobile solar forcing was used."),
                "warnings": [warning],
            },
        )
    ratios = shortwave[valid] / clear_at_obs[valid]
    envelope_ratio_raw = float(np.quantile(ratios, quantile))
    envelope_ratio = float(np.clip(envelope_ratio_raw, 0.0, 1.0))
    if envelope_ratio_raw > 1.0:
        warnings.append(
            "mobile upper-envelope shortwave exceeded pvlib clear-sky GHI; "
            "cloud attenuation was bounded at zero")

    # Select the scalar cloud fraction whose established TREC-Route adjustment
    # best reproduces the bounded high-ratio envelope at the observation times.
    grids = np.linspace(0.0, 1.0, 1001)
    target = envelope_ratio * clear_at_obs[valid]
    errors = np.empty_like(grids)
    for index, cloud in enumerate(grids):
        _, _, candidate_ghi = apply_cloud_adjustment(
            _periodic_interp(model_hours_local, clear_dni, obs_hours[valid]),
            _periodic_interp(model_hours_local, clear_dhi, obs_hours[valid]),
            elevation_at_obs[valid], cloud)
        errors[index] = np.sqrt(np.mean((candidate_ghi - target) ** 2))
    cloud_fraction = float(grids[int(np.argmin(errors))])
    dni, dhi, ghi = apply_cloud_adjustment(
        clear_dni, clear_dhi, elevation_deg, cloud_fraction)
    metadata = {
        "mode": "route_upper_envelope_cloud",
        "source_file": str(source),
        "time_column": time_column,
        "shortwave_column": sw_column,
        "model_date": str(model_date),
        "n_source_rows": int(len(frame)),
        "n_usable_daytime_samples": int(valid.sum()),
        "minimum_solar_elevation_deg": minimum_elevation,
        "upper_quantile": quantile,
        "observed_shortwave_max_Wm2": float(np.max(shortwave[valid])),
        "observed_to_clear_ghi_upper_ratio_raw": envelope_ratio_raw,
        "observed_to_clear_ghi_upper_ratio_used": envelope_ratio,
        "inferred_case_cloud_fraction": cloud_fraction,
        "fit_rmse_Wm2": float(np.min(errors)),
        "interpretation": (
            "One session-wide cloud attenuation inferred from high mobile SWin/clear-"
            "sky-GHI ratios; mobile pointwise SWin is not used as atmospheric forcing."
        ),
        "warnings": warnings,
    }
    return RadiationForcing(
        dni, dhi, ghi, np.full(len(model_times), cloud_fraction),
        np.full(len(model_times), np.nan),
        "mobile_shortwave_upper_envelope_cloud_adjustment", metadata,
    )


def resolve_radiation_forcing(
    config_path: str | Path | None,
    model_times: pd.DatetimeIndex,
    clear_dni: np.ndarray,
    clear_dhi: np.ndarray,
    clear_ghi: np.ndarray,
    elevation_deg: np.ndarray,
    default_cloud_fraction: float,
) -> RadiationForcing:
    """Resolve optional case forcing without embedding case names in the workflow."""
    n = len(model_times)
    if config_path is None:
        dni, dhi, ghi = apply_cloud_adjustment(
            clear_dni, clear_dhi, elevation_deg, default_cloud_fraction)
        return RadiationForcing(
            dni, dhi, ghi, np.full(n, default_cloud_fraction), np.full(n, np.nan),
            "pvlib_ineichen_clear_sky_plus_cloud_adjustment",
            {"mode": "clear_sky", "configured_cloud_fraction": default_cloud_fraction},
        )
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"radiation forcing config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    mode = str(config.get("mode", "clear_sky")).strip().lower()
    if mode == "clear_sky":
        cloud = float(config.get("cloud_fraction", default_cloud_fraction))
        if not 0 <= cloud <= 1:
            raise ValueError("radiation forcing cloud_fraction must lie in [0, 1]")
        dni, dhi, ghi = apply_cloud_adjustment(clear_dni, clear_dhi, elevation_deg, cloud)
        result = RadiationForcing(
            dni, dhi, ghi, np.full(n, cloud), np.full(n, np.nan),
            "pvlib_ineichen_clear_sky_plus_case_cloud_adjustment",
            {"mode": mode, "configured_cloud_fraction": cloud},
        )
    elif mode == "components_csv":
        source = Path(config.get("source_file", ""))
        if not source.is_absolute():
            source = (path.parent / source).resolve()
        result = _read_component_csv(source, model_times)
        cloud = np.where(np.isfinite(result.cloud_fraction),
                         result.cloud_fraction, default_cloud_fraction)
        result = RadiationForcing(
            result.dni_wm2, result.dhi_wm2, result.ghi_wm2, cloud,
            result.lwin_wm2, result.source, result.metadata)
        result.metadata["fallback_cloud_fraction_for_sky_longwave"] = (
            default_cloud_fraction)
    elif mode == "route_upper_envelope_cloud":
        result = _load_mobile_envelope(
            config, path, model_times, clear_ghi, elevation_deg,
            clear_dni, clear_dhi)
    else:
        raise ValueError(
            f"unsupported radiation forcing mode {mode!r}; use clear_sky, "
            "components_csv, or route_upper_envelope_cloud")
    result.metadata.update({
        "config_file": str(path),
        "resolved_utc": datetime.now(timezone.utc).isoformat(),
    })
    return result


def write_forcing_metadata(result: RadiationForcing, output_path: str | Path) -> None:
    """Write a JSON-safe forcing provenance sidecar."""
    path = Path(output_path)
    path.write_text(json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8")
