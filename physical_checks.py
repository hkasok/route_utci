"""
physical_checks.py -- fail-fast validation of physical quantities and their
UNITS at every point where they enter a model equation.

WHY THIS EXISTS
---------------
Every humidity consumer in this pipeline expects RELATIVE HUMIDITY IN PERCENT
(0-100), not a 0-1 fraction and not a vapour pressure:

  * pythermalcomfort  utci(rh=...)      -- percent (verified: UTCI rises
                                           monotonically from rh=10 to rh=90)
  * pythermalcomfort  JOS3().rh         -- percent (library default is 50)
  * thermal_common.clear_sky_emissivity -- takes PERCENT and converts it
                                           internally to vapour pressure:
                                             e0 [hPa] = (RH/100) * e_sat(Ta)
                                             w  [cm]  = 46.5 * e0 / Ta_K
                                           (Prata 1996 wants hPa and K)

The dangerous failure mode is a UNIT slip, because it is silent and plausible:
passing RH as a fraction (0.70 instead of 70) makes the Prata clear-sky
emissivity read 0.672 instead of 0.894 -- a desert sky in humid Miami, ~55 W/m2
of missing longwave and several degrees of Tmrt -- with no error and no
obviously wrong number. UTCI behaves the same way: rh=0.5 is silently treated
as 0.5% (bone dry), not 50%.

These helpers turn that silent corruption into a loud, specific exception.
"""

import numpy as np

# Plausible-value envelopes. Outside these the input is treated as an error,
# not as an extreme-but-real value.
RH_MIN_PCT, RH_MAX_PCT = 0.0, 100.0
AIR_TEMP_MIN_C, AIR_TEMP_MAX_C = -90.0, 60.0     # Vostok .. Death Valley
WIND_MIN_MS, WIND_MAX_MS = 0.0, 113.0            # 113 = surface record
TMRT_MIN_C, TMRT_MAX_C = -100.0, 150.0
# UTCI's operational applicability limits (Brode et al. 2012), used for
# advisory warnings rather than hard errors.
UTCI_WIND_MIN_MS, UTCI_WIND_MAX_MS = 0.5, 17.0


def _finite(values, name, context):
    a = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(a)):
        n = int((~np.isfinite(a)).sum())
        raise ValueError(
            f"[{context}] {name}: {n} non-finite value(s) (NaN/inf). "
            f"Fix the source data -- these propagate silently into the results.")
    return a


def check_rh_pct(rh, context, allow_zero=True):
    """Validate relative humidity is a PERCENT in [0, 100].

    Also catches the classic fraction-instead-of-percent slip: if every value
    is <= 1.0 (and not all exactly zero) the input is almost certainly a 0-1
    fraction, which every consumer here would silently read as ~0-1 % (bone
    dry). That is rejected explicitly with a fix-it message.
    """
    a = _finite(rh, "relative humidity", context)
    if a.size == 0:
        return a
    lo, hi = float(np.min(a)), float(np.max(a))
    if lo < RH_MIN_PCT or hi > RH_MAX_PCT:
        raise ValueError(
            f"[{context}] relative humidity out of physical range: "
            f"min {lo:.3g}, max {hi:.3g} -- must be a PERCENT in "
            f"[{RH_MIN_PCT:g}, {RH_MAX_PCT:g}]. "
            f"(If these look like a 0-1 fraction, multiply by 100; if they "
            f"look like a vapour pressure in hPa, convert to RH first -- "
            f"every model here takes RH in percent.)")
    nonzero = a[a > 0.0]
    if nonzero.size and float(np.max(nonzero)) <= 1.0:
        raise ValueError(
            f"[{context}] relative humidity looks like a 0-1 FRACTION, not a "
            f"percent: max value is {float(np.max(nonzero)):.4g}. Every "
            f"consumer (UTCI, JOS-3, Prata clear-sky emissivity) expects "
            f"PERCENT, and would silently treat this as ~{float(np.max(nonzero)):.2g}% "
            f"(bone dry). Multiply by 100. If this really is sub-1% humidity, "
            f"pass it as e.g. 1.0 rather than 0.01.")
    if not allow_zero and lo <= 0.0:
        raise ValueError(f"[{context}] relative humidity of {lo:.3g}% is not "
                         f"usable here (zero/negative humidity).")
    return a


def check_air_temp_c(ta, context):
    """Validate air temperature is in DEGREES CELSIUS (not kelvin)."""
    a = _finite(ta, "air temperature", context)
    if a.size == 0:
        return a
    lo, hi = float(np.min(a)), float(np.max(a))
    if lo < AIR_TEMP_MIN_C or hi > AIR_TEMP_MAX_C:
        hint = ""
        if lo > 150.0:
            hint = (" These look like KELVIN -- subtract 273.15; this pipeline "
                    "uses degrees Celsius for air temperature.")
        raise ValueError(
            f"[{context}] air temperature out of physical range: min {lo:.3g}, "
            f"max {hi:.3g} degC (allowed [{AIR_TEMP_MIN_C:g}, "
            f"{AIR_TEMP_MAX_C:g}]).{hint}")
    return a


def check_wind_ms(v, context, warn_utci_limits=False):
    """Validate wind speed in m/s. Optionally warn about UTCI's valid band."""
    a = _finite(v, "wind speed", context)
    if a.size == 0:
        return a
    lo, hi = float(np.min(a)), float(np.max(a))
    if lo < WIND_MIN_MS or hi > WIND_MAX_MS:
        raise ValueError(
            f"[{context}] wind speed out of physical range: min {lo:.3g}, "
            f"max {hi:.3g} m/s (allowed [{WIND_MIN_MS:g}, {WIND_MAX_MS:g}]). "
            f"(If these are km/h, divide by 3.6; if mph, divide by 2.237.)")
    if warn_utci_limits and (lo < UTCI_WIND_MIN_MS or hi > UTCI_WIND_MAX_MS):
        print(f"  NOTE [{context}]: wind {lo:.2g}-{hi:.2g} m/s extends outside "
              f"UTCI's operational band [{UTCI_WIND_MIN_MS}, {UTCI_WIND_MAX_MS}] "
              f"m/s; values there are extrapolated. UTCI expects 10 m wind.")
    return a


def check_tmrt_c(tr, context):
    """Validate mean radiant temperature in degrees Celsius."""
    a = _finite(tr, "Tmrt", context)
    if a.size == 0:
        return a
    lo, hi = float(np.min(a)), float(np.max(a))
    if lo < TMRT_MIN_C or hi > TMRT_MAX_C:
        hint = (" These look like KELVIN -- subtract 273.15." if lo > 200.0 else "")
        raise ValueError(
            f"[{context}] Tmrt out of physical range: min {lo:.3g}, max "
            f"{hi:.3g} degC (allowed [{TMRT_MIN_C:g}, {TMRT_MAX_C:g}]).{hint}")
    return a


def check_utci_inputs(tdb, tr, v, rh, context):
    """Validate the full argument set going into pythermalcomfort's utci().

    All four are validated together so a unit slip in ANY of them is caught at
    the call site rather than showing up as a plausible-looking UTCI value.
    """
    check_air_temp_c(tdb, context)
    check_tmrt_c(tr, context)
    check_wind_ms(v, context, warn_utci_limits=True)
    check_rh_pct(rh, context)


def check_jos3_inputs(tdb, tr, v, rh, context):
    """Validate the boundary conditions assigned to a JOS-3 model instance."""
    check_air_temp_c(tdb, context)
    check_tmrt_c(tr, context)
    check_wind_ms(v, context)
    check_rh_pct(rh, context)
