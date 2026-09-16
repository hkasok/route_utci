"""black_globe.py -- emulate a black-globe thermometer inside TREC-Route.

WHY THIS EXISTS
---------------
The Lisbon validation campaigns report a black-globe temperature (``BGTemp_C``)
and a globe-derived mean radiant temperature (``Tmrt``). TREC-Route's own MRT is
a STANDING-CYLINDER quantity: a walking body intercepts the direct beam through
the altitude-dependent Fanger projected-area factor, so at high sun it absorbs
far less beam per unit area than a sphere does. Comparing that number against a
globe reading is a convention mismatch, which is why the MRT comparison has been
switched off by default (see ``README_sensor_equivalents.md``).

This module closes the gap the honest way -- by simulating the instrument. Given
the sphere-weighted radiation load that stage 05 already knows how to trace, it
solves the globe's own energy balance and returns the temperature that a real
globe would report. The comparison then puts a modelled globe temperature beside
a measured globe temperature: one quantity against itself.

THE THREE THINGS THAT MAKE A GLOBE READ WHAT IT READS
-----------------------------------------------------
1. ANGULAR WEIGHTING. A sphere presents the same projected area in every
   direction, so its projected-area factor for the direct beam is exactly 0.25
   at all solar altitudes. The standing body's is ~0.31 near the horizon but
   ~0.08 at the zenith. This is a large, systematic, midday-peaking difference,
   and it is the whole reason globe MRT and cylinder MRT are different numbers.

2. CONVECTION. The globe is not a radiometer; it is a thermometer that happens
   to be radiatively coupled. Wind pulls it back toward air temperature, and the
   ISO 7726 inversion exists precisely to undo that. A globe in 3 m/s wind reads
   dramatically cooler than the same globe in still air under identical
   radiation.

3. THERMAL INERTIA -- and for a WALKING campaign this dominates. A 150 mm copper
   globe takes 20-30 minutes to equilibrate (ISO 7726). A pedestrian crosses a
   sun/shade boundary in seconds. The globe therefore never comes close to
   equilibrium during a walk: it low-pass filters the radiation field it moves
   through. In the Lisbon route-1 day walks the measured globe varies by well
   under 1 K standard deviation while incident shortwave swings across the full
   0-1000 W/m2 range. A steady-state globe emulation would swing by ~15 K and
   correlate poorly with the measurement for a reason that has nothing to do
   with the radiation model being wrong. So the transient integration here is
   not a refinement; without it the comparison is meaningless.

ENERGY BALANCE
--------------
Per unit globe surface area, with ``C`` the areal heat capacity [J m-2 K-1]::

    C dTg/dt = R_abs - eps_g * sigma * Tg^4 - h_c * (Tg - Ta)

``R_abs`` is the sphere-weighted absorbed radiation (shortwave times globe
absorptivity plus longwave times globe emissivity) supplied by stage 05.

Convection follows ISO 7726: forced ``h = 1.1e8 * sigma * V^0.6 / D^0.4`` and
natural ``h = 1.4 * (|Tg - Ta| / D)^0.25``, taking whichever is larger, which is
the usual treatment for the transition regime a walking cart lives in.

WHAT THIS IS NOT
----------------
This is an instrument emulator, exactly like the four-component radiometer
channels in ``radiant_flux_contributions.SENSOR_COLUMNS``. It is a diagnostic
for field comparison. The standing-cylinder MRT remains the simulation product,
because a pedestrian is not a sphere. Never feed a globe temperature into UTCI
or JOS-3 in place of the body MRT.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import numpy as np

SIGMA = 5.670374419e-8

# ISO 7726 forced-convection constant, converted from the standard's Tmrt form
# to a heat-transfer coefficient:
#
#   ISO:  Tmrt^4 = Tg^4 + 1.1e8 * V^0.6 / (eps * D^0.4) * (Tg - Ta)
#   balance: eps*sigma*Tmrt^4 = eps*sigma*Tg^4 + h*(Tg - Ta)
#   => h = 1.1e8 * sigma * V^0.6 / D^0.4
#
# Note the emissivity cancels, as it must: convection off a sphere cannot depend
# on how black the sphere is painted. (The standard's Tmrt formula keeps eps in
# the denominator only because it is dividing through by eps*sigma.)
ISO_FORCED_COEFFICIENT = 1.1e8 * SIGMA          # ~6.2374 W m-2 K-1 per (m/s)^0.6
ISO_FORCED_WIND_EXPONENT = 0.6
ISO_FORCED_DIAMETER_EXPONENT = 0.4
NATURAL_CONVECTION_COEFFICIENT = 1.4            # h = 1.4 * (dT / D)^0.25

COPPER_DENSITY_KG_M3 = 8960.0
COPPER_SPECIFIC_HEAT_J_KGK = 385.0


@dataclass(frozen=True)
class GlobeSpec:
    """Physical description of one black-globe thermometer.

    ``areal_heat_capacity_J_m2K`` is the globe's heat capacity per unit of its
    OWN surface area, i.e. ``m * c_p / (pi * D^2)``. Parameterising it this way
    keeps the energy balance in flux units and makes the time constant fall out
    as ``tau = C / (h_convective + h_radiative)``.
    """

    name: str
    diameter_m: float
    emissivity: float
    sw_absorptivity: float
    areal_heat_capacity_J_m2K: float
    reference: str = ""

    def __post_init__(self) -> None:
        if not (0.0 < self.diameter_m < 2.0):
            raise ValueError(f"globe diameter out of range: {self.diameter_m}")
        for field in ("emissivity", "sw_absorptivity"):
            value = getattr(self, field)
            if not (0.0 < value <= 1.0):
                raise ValueError(f"globe {field} must be in (0, 1]: {value}")
        if self.areal_heat_capacity_J_m2K <= 0.0:
            raise ValueError("globe areal heat capacity must be positive")

    def surface_area_m2(self) -> float:
        return float(np.pi * self.diameter_m ** 2)

    def heat_capacity_J_K(self) -> float:
        return self.areal_heat_capacity_J_m2K * self.surface_area_m2()

    def time_constant_s(self, wind_ms: float = 1.5,
                        temperature_K: float = 310.0) -> float:
        """Single-node time constant at reference conditions.

        Both loss paths matter: at typical urban wind the radiative coefficient
        ``4*eps*sigma*T^3`` is roughly a third of the convective one, so leaving
        it out would overstate the lag by ~30%.
        """
        h_conv = forced_convection_coefficient(wind_ms, self.diameter_m)
        h_rad = 4.0 * self.emissivity * SIGMA * float(temperature_K) ** 3
        return float(self.areal_heat_capacity_J_m2K / (h_conv + h_rad))

    def as_metadata(self) -> dict[str, Any]:
        record = asdict(self)
        record["surface_area_m2"] = self.surface_area_m2()
        record["heat_capacity_J_K"] = self.heat_capacity_J_K()
        record["time_constant_s_at_1p5ms"] = self.time_constant_s()
        return record


def copper_shell_areal_heat_capacity(thickness_m: float) -> float:
    """Areal heat capacity of a thin copper shell -- the globe's own metal.

    This is a LOWER BOUND on the effective value for a globe whose thermistor
    sits at the centre rather than bonded to the shell: the enclosed air and the
    sensor's own mount add a second lag stage that a single-node model has to
    absorb into C. See ``CAMPBELL_BLACKGLOBE_L`` for how the default handles it.
    """
    if thickness_m <= 0.0:
        raise ValueError("shell thickness must be positive")
    return COPPER_DENSITY_KG_M3 * COPPER_SPECIFIC_HEAT_J_KGK * float(thickness_m)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
# The Lisbon campaigns (six routes, day and night, seasonal "climate walking"
# missions run by the University of Lisbon IGOT group) used a mobile cart
# carrying a Gill MaxiMet GMX500 compact weather station with GPS, a Campbell
# Scientific CR350 datalogger, an up/down pyranometer + pyrgeometer pair, and a
# Campbell Scientific BLACKGLOBE-L. The BLACKGLOBE-L is a thermistor at the
# centre of a 15.2 cm (6 in) hollow copper sphere painted black, with a stated
# near-normal emittance of 0.957.
#
# Heat capacity: a bare 0.5 mm copper shell gives only ~1725 J m-2 K-1, i.e. a
# ~70 s time constant, but ISO 7726 states a 150 mm globe needs 20-30 minutes to
# equilibrate -- around a 300 s single-node time constant. The difference is the
# internal air gap and the centre-mounted thermistor, which a one-node model can
# only represent by inflating C. The default below is therefore set so that the
# time constant is ~300 s at a 1.5 m/s reference wind, which reproduces the
# instrument's documented response rather than the bare metal's. Override with
# --globe-areal-heat-capacity if a bench calibration is available.
_CAMPBELL_REFERENCE_TAU_S = 300.0
_CAMPBELL_DIAMETER_M = 0.152
_CAMPBELL_EMISSIVITY = 0.957


def _areal_heat_capacity_for_time_constant(tau_s: float, diameter_m: float,
                                           emissivity: float,
                                           wind_ms: float = 1.5,
                                           temperature_K: float = 310.0) -> float:
    h_conv = forced_convection_coefficient(wind_ms, diameter_m)
    h_rad = 4.0 * emissivity * SIGMA * temperature_K ** 3
    return float(tau_s * (h_conv + h_rad))


def forced_convection_coefficient(wind_ms: Any, diameter_m: float) -> Any:
    """ISO 7726 forced convection for a sphere [W m-2 K-1]."""
    wind = np.maximum(np.asarray(wind_ms, dtype=float), 0.0)
    return (ISO_FORCED_COEFFICIENT * wind ** ISO_FORCED_WIND_EXPONENT
            / float(diameter_m) ** ISO_FORCED_DIAMETER_EXPONENT)


def natural_convection_coefficient(globe_temp_C: Any, air_temp_C: Any,
                                   diameter_m: float) -> Any:
    """Free-convection coefficient for a sphere [W m-2 K-1]."""
    delta = np.abs(np.asarray(globe_temp_C, dtype=float)
                   - np.asarray(air_temp_C, dtype=float))
    return NATURAL_CONVECTION_COEFFICIENT * (delta / float(diameter_m)) ** 0.25


def convection_coefficient(globe_temp_C: Any, air_temp_C: Any, wind_ms: Any,
                           diameter_m: float) -> Any:
    """Mixed convection: the larger of the forced and natural coefficients.

    A walking cart sits squarely in the transition regime -- 0.2 m/s in a
    sheltered courtyard, 3 m/s on an exposed esplanade -- so neither limit alone
    is defensible across a route. Taking the maximum is the standard engineering
    treatment and is continuous enough for the implicit solve below.
    """
    forced = forced_convection_coefficient(wind_ms, diameter_m)
    natural = natural_convection_coefficient(globe_temp_C, air_temp_C, diameter_m)
    return np.maximum(forced, natural)


CAMPBELL_BLACKGLOBE_L = GlobeSpec(
    name="campbell_blackglobe_l",
    diameter_m=_CAMPBELL_DIAMETER_M,
    emissivity=_CAMPBELL_EMISSIVITY,
    # Matte black paint: shortwave absorptivity is close to, and conventionally
    # taken equal to, the longwave emittance for these globes.
    sw_absorptivity=0.95,
    areal_heat_capacity_J_m2K=_areal_heat_capacity_for_time_constant(
        _CAMPBELL_REFERENCE_TAU_S, _CAMPBELL_DIAMETER_M, _CAMPBELL_EMISSIVITY),
    reference=("Campbell Scientific BLACKGLOBE-L: 15.2 cm hollow copper sphere, "
               "black painted, near-normal emittance 0.957; response lumped to a "
               "~300 s single-node time constant per ISO 7726 equilibration time"),
)

ISO_7726_STANDARD_GLOBE = GlobeSpec(
    name="iso_7726_standard",
    diameter_m=0.15,
    emissivity=0.95,
    sw_absorptivity=0.95,
    areal_heat_capacity_J_m2K=_areal_heat_capacity_for_time_constant(
        300.0, 0.15, 0.95),
    reference="ISO 7726 reference globe: 150 mm, emissivity 0.95",
)

# A 40 mm grey/ping-pong globe is common in mobile work precisely BECAUSE it is
# fast (tens of seconds), at the cost of a much larger convective correction.
SMALL_GLOBE_40MM = GlobeSpec(
    name="small_globe_40mm",
    diameter_m=0.040,
    emissivity=0.95,
    sw_absorptivity=0.95,
    areal_heat_capacity_J_m2K=copper_shell_areal_heat_capacity(0.0004),
    reference="40 mm thin-shell globe, fast-response mobile variant",
)

GLOBE_PRESETS: dict[str, GlobeSpec] = {
    spec.name: spec for spec in
    (CAMPBELL_BLACKGLOBE_L, ISO_7726_STANDARD_GLOBE, SMALL_GLOBE_40MM)
}

DEFAULT_GLOBE = CAMPBELL_BLACKGLOBE_L


def resolve_globe_spec(preset: str | None = None, *,
                       diameter_m: float | None = None,
                       emissivity: float | None = None,
                       sw_absorptivity: float | None = None,
                       areal_heat_capacity_J_m2K: float | None = None
                       ) -> GlobeSpec:
    """Look up a preset and apply any explicit per-field overrides."""
    base = DEFAULT_GLOBE if not preset else GLOBE_PRESETS.get(preset)
    if base is None:
        raise ValueError(f"unknown globe preset {preset!r}; "
                         f"available: {sorted(GLOBE_PRESETS)}")
    overrides = {
        "diameter_m": diameter_m,
        "emissivity": emissivity,
        "sw_absorptivity": sw_absorptivity,
        "areal_heat_capacity_J_m2K": areal_heat_capacity_J_m2K,
    }
    applied = {key: value for key, value in overrides.items() if value is not None}
    if not applied:
        return base
    record = asdict(base)
    record.update(applied)
    # Note on resizing: the heat capacity is carried PER UNIT AREA, and for a
    # shell of given material and wall thickness that quantity (rho * c * t) is
    # independent of diameter. So changing only the diameter correctly leaves C
    # alone -- "the same construction, a different size". The time constant then
    # falls out of the physics and shortens, because a smaller sphere has a
    # larger convective coefficient. Do NOT be tempted to hold the time constant
    # fixed instead: that would make a 40 mm globe heavier per unit area than a
    # 150 mm one, which no real instrument is.
    record["name"] = base.name + "_modified"
    record["reference"] = base.reference + " (modified)"
    return GlobeSpec(**record)


# ---------------------------------------------------------------------------
# Radiation load and temperature solves
# ---------------------------------------------------------------------------
def sphere_projected_area_factor() -> float:
    """A sphere intercepts the beam with the same projected area from every
    direction, so ``f_p`` is exactly 0.25 -- the ratio of the disc it presents
    (pi D^2 / 4) to its own surface area (pi D^2). Unlike the standing body's
    factor this has no solar-altitude dependence at all."""
    return 0.25


def radiative_equilibrium_temperature_C(absorbed_flux_Wm2: Any,
                                        emissivity: float) -> Any:
    """Temperature a body reaches with radiation alone -- i.e. the sphere-weighted
    mean radiant temperature when the absorbed flux is sphere-weighted."""
    flux = np.asarray(absorbed_flux_Wm2, dtype=float)
    if np.any(flux <= 0.0):
        raise ValueError("absorbed radiant flux must be positive")
    return (flux / (float(emissivity) * SIGMA)) ** 0.25 - 273.15


def steady_globe_temperature_C(absorbed_flux_Wm2: Any, air_temp_C: Any,
                               wind_ms: Any, spec: GlobeSpec, *,
                               iterations: int = 60,
                               tolerance_K: float = 1e-9) -> np.ndarray:
    """Solve ``R = eps*sigma*Tg^4 + h(Tg)*(Tg - Ta)`` for Tg by damped Newton.

    The unknown appears in the natural-convection coefficient as well as the
    quartic, so h is refreshed each iteration rather than frozen.
    """
    flux = np.asarray(absorbed_flux_Wm2, dtype=float)
    air = np.asarray(air_temp_C, dtype=float)
    wind = np.asarray(wind_ms, dtype=float)
    flux, air, wind = np.broadcast_arrays(flux, air, wind)
    if np.any(flux <= 0.0):
        raise ValueError("absorbed radiant flux must be positive")
    eps_sigma = spec.emissivity * SIGMA
    # Start from the radiative equilibrium, which brackets the answer from the
    # hot side whenever the globe is warmer than the air (the daytime case).
    globe = (flux / eps_sigma) ** 0.25 - 273.15
    for _ in range(int(iterations)):
        h = convection_coefficient(globe, air, wind, spec.diameter_m)
        globe_K = globe + 273.15
        residual = flux - eps_sigma * globe_K ** 4 - h * (globe - air)
        derivative = -4.0 * eps_sigma * globe_K ** 3 - h
        step = residual / derivative
        globe = globe - step
        if np.max(np.abs(step)) < tolerance_K:
            break
    else:
        raise RuntimeError("steady globe temperature solve did not converge")
    return np.asarray(globe, dtype=float)


def integrate_globe_temperature_C(elapsed_s: Any, absorbed_flux_Wm2: Any,
                                  air_temp_C: Any, wind_ms: Any,
                                  spec: GlobeSpec, *,
                                  initial_temp_C: float | None = None,
                                  iterations: int = 40,
                                  tolerance_K: float = 1e-9
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the globe along a walk and flag the spin-up region.

    ``elapsed_s`` are the per-sample times of the walk (need not be uniform;
    a mobile logger's clock rarely is). The scheme is backward Euler with a
    Newton solve per step, which stays stable no matter how coarse the sampling
    is -- an explicit step would blow up whenever dt approached the ~300 s time
    constant, which is exactly the regime a 1 Hz logger straddles when it drops
    samples.

    Returns ``(globe_temp_C, spinup_affected)``. The globe has no memory of
    conditions before the first sample, so the run starts from the steady state
    of that first sample and the first ``3*tau`` of the walk is flagged: within
    that window the trace is following the assumed initial condition as much as
    the radiation field, and comparing it against a measurement would mostly
    test the guess.
    """
    time_s = np.asarray(elapsed_s, dtype=float)
    flux = np.asarray(absorbed_flux_Wm2, dtype=float)
    air = np.asarray(air_temp_C, dtype=float)
    wind = np.asarray(wind_ms, dtype=float)
    n = time_s.size
    if not (flux.size == air.size == wind.size == n):
        raise ValueError("globe integration inputs must have equal length")
    if n == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=bool)
    if np.any(np.diff(time_s) < 0.0):
        raise ValueError("globe integration requires non-decreasing sample times")
    if not np.isfinite(time_s).all():
        raise ValueError("globe integration requires finite sample times")

    eps_sigma = spec.emissivity * SIGMA
    capacity = spec.areal_heat_capacity_J_m2K

    if initial_temp_C is None:
        start = float(steady_globe_temperature_C(
            flux[:1], air[:1], wind[:1], spec)[0])
    else:
        start = float(initial_temp_C)

    globe = np.empty(n, dtype=float)
    globe[0] = start
    for index in range(1, n):
        dt = time_s[index] - time_s[index - 1]
        if dt <= 0.0:
            # Duplicate timestamps happen when a logger repeats a record; the
            # globe cannot change in zero time.
            globe[index] = globe[index - 1]
            continue
        previous = globe[index - 1]
        current = previous
        for _ in range(int(iterations)):
            h = convection_coefficient(current, air[index], wind[index],
                                       spec.diameter_m)
            current_K = current + 273.15
            residual = (capacity * (current - previous) / dt
                        - flux[index] + eps_sigma * current_K ** 4
                        + h * (current - air[index]))
            derivative = (capacity / dt + 4.0 * eps_sigma * current_K ** 3 + h)
            step = residual / derivative
            current = current - step
            if abs(step) < tolerance_K:
                break
        else:
            raise RuntimeError(
                f"globe transient solve did not converge at sample {index}")
        globe[index] = current

    reference_wind = float(np.median(wind)) if np.isfinite(wind).any() else 1.5
    tau = spec.time_constant_s(max(reference_wind, 0.1),
                               float(np.median(air)) + 273.15)
    spinup = (time_s - time_s[0]) < (3.0 * tau)
    return globe, spinup


def relative_air_speed(air_u_ms: Any, air_v_ms: Any,
                       receptor_u_ms: Any, receptor_v_ms: Any) -> np.ndarray:
    """Air speed felt by a receptor that is itself moving [m/s].

    Convection off the globe is driven by the air speed RELATIVE to the globe,
    and in a mobile campaign the globe is being pushed along at roughly walking
    pace. Feeding it the ambient wind alone under-ventilates it and leaves the
    modelled globe sitting too far above air temperature -- on Lisbon route 1
    that single omission accounts for most of a +4.9 K daytime offset, and
    including it also brings the modelled spread into line with the measured one.

    This is a vector difference, not a sum: walking into a headwind ventilates
    the globe far better than walking downwind at the same speed, and on a route
    that turns through the wind both happen within one walk.
    """
    du = np.asarray(air_u_ms, dtype=float) - np.asarray(receptor_u_ms, dtype=float)
    dv = np.asarray(air_v_ms, dtype=float) - np.asarray(receptor_v_ms, dtype=float)
    return np.hypot(du, dv)


def air_direction_is_resolved(air_u_ms: Any, air_v_ms: Any, *,
                              tolerance_deg: float = 1.0) -> bool:
    """Does the air-velocity field actually carry a per-point DIRECTION?

    This distinguishes a solved flow field, whose heading turns around corners
    and through canyons, from the uniform fallback, where every route point is
    handed one globally assumed wind direction. The test is simply whether the
    heading varies along the route at all.

    It matters because of how the relative velocity is then formed. Subtracting
    the walker's motion from a genuinely resolved local wind is correct. Doing
    the same against a single assumed direction is not: the answer then depends
    entirely on whether that guess happens to point along the route or across
    it, and it can just as easily reduce the ventilation as increase it.
    """
    u = np.asarray(air_u_ms, dtype=float)
    v = np.asarray(air_v_ms, dtype=float)
    if u.size == 0 or not (np.isfinite(u).all() and np.isfinite(v).all()):
        return False
    speed = np.hypot(u, v)
    moving = speed > 1e-9
    if moving.sum() < 2:
        return False
    heading = np.arctan2(v[moving], u[moving])
    concentration = np.hypot(np.mean(np.sin(heading)), np.mean(np.cos(heading)))
    return bool(concentration < np.cos(np.deg2rad(tolerance_deg)))


def ventilation_speed(air_speed_ms: Any, receptor_speed_ms: Any, *,
                      air_u_ms: Any = None, air_v_ms: Any = None,
                      receptor_u_ms: Any = None, receptor_v_ms: Any = None
                      ) -> np.ndarray:
    """Air speed ventilating a moving receptor, using the best available basis.

    With a resolved local wind direction this is the exact vector difference.
    Without one it falls back to quadrature, which is not a fudge: averaged over
    a uniformly distributed relative heading, the ROOT-MEAN-SQUARE relative speed
    is exactly ``sqrt(V_air^2 + V_receptor^2)``, because the cross term
    ``-2*V_air*V_receptor*cos(theta)`` integrates to zero. So quadrature is the
    direction-agnostic expectation, and it is the right answer when the only
    direction available is a single global assumption.

    (It very slightly overstates the convective coefficient, since h scales as
    V^0.6 and that is concave, so the mean of V^0.6 sits below the RMS. The
    effect is a fraction of a percent at these speeds.)
    """
    air_speed = np.asarray(air_speed_ms, dtype=float)
    receptor_speed = np.asarray(receptor_speed_ms, dtype=float)
    vectors = (air_u_ms, air_v_ms, receptor_u_ms, receptor_v_ms)
    if all(component is not None for component in vectors) \
            and air_direction_is_resolved(air_u_ms, air_v_ms):
        return relative_air_speed(air_u_ms, air_v_ms, receptor_u_ms, receptor_v_ms)
    return np.hypot(air_speed, receptor_speed)


def receptor_velocity(x_m: Any, y_m: Any, elapsed_s: Any, *,
                      smoothing_samples: int = 15,
                      maximum_speed_ms: float = 4.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Velocity of a walker along a route, from its geometry and its clock.

    Densified route geometry paired with recorded device timestamps gives a
    per-sample speed that is noisy at the individual-sample level (GPS jitter,
    a logger clock quantised to whole seconds). A short centred median filter
    removes that without touching the real pace changes, and the speed is capped
    because a spurious coordinate jump would otherwise inject an impossible
    ventilation velocity.
    """
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    time_s = np.asarray(elapsed_s, dtype=float)
    if not (x.size == y.size == time_s.size):
        raise ValueError("receptor velocity inputs must have equal length")
    if x.size < 2:
        return np.zeros_like(x), np.zeros_like(y)
    step = np.gradient(time_s)
    step = np.where(np.abs(step) < 1e-9, 1e-9, step)
    u = np.gradient(x) / step
    v = np.gradient(y) / step
    if smoothing_samples > 1:
        import pandas as pd  # local: keeps the physics module import-light
        window = int(smoothing_samples)
        u = pd.Series(u).rolling(window, center=True, min_periods=1).median().to_numpy()
        v = pd.Series(v).rolling(window, center=True, min_periods=1).median().to_numpy()
    speed = np.hypot(u, v)
    excessive = speed > maximum_speed_ms
    if np.any(excessive):
        scale = np.where(excessive, maximum_speed_ms / np.maximum(speed, 1e-9), 1.0)
        u, v = u * scale, v * scale
    return u, v


def globe_mrt_iso7726_C(globe_temp_C: Any, air_temp_C: Any, wind_ms: Any,
                        spec: GlobeSpec) -> np.ndarray:
    """The ISO 7726 inversion: globe temperature -> mean radiant temperature.

    This is the CAMPAIGN's convention, reproduced so a measured ``Tmrt`` column
    can be met on its own terms. Note what it does and does not give you: it
    removes the convective pull toward air temperature, but the result is still
    a SPHERE-weighted mean radiant temperature. It is not, and cannot be turned
    into, the standing-cylinder MRT that TREC-Route reports for the pedestrian.
    """
    globe = np.asarray(globe_temp_C, dtype=float)
    air = np.asarray(air_temp_C, dtype=float)
    wind = np.maximum(np.asarray(wind_ms, dtype=float), 0.0)
    h = convection_coefficient(globe, air, wind, spec.diameter_m)
    quartic = ((globe + 273.15) ** 4
               + h * (globe - air) / (spec.emissivity * SIGMA))
    quartic = np.maximum(quartic, 1.0)
    return quartic ** 0.25 - 273.15


def fit_campaign_wind_convention(globe_temp_C: Any, air_temp_C: Any,
                                 reported_mrt_C: Any, spec: GlobeSpec
                                 ) -> dict[str, float]:
    """Recover the wind speed a campaign actually used in its ISO inversion.

    Reported Tmrt columns are frequently NOT reproducible from the logged
    instantaneous wind: crews often substitute a campaign-mean or fixed-station
    value, because a cart-mounted anemometer in a street canyon is noisy and the
    inversion is violently sensitive to it near zero. This inverts the ISO
    relation for the single constant wind speed that reproduces the reported
    series, and reports how constant the implied coefficient actually was --
    which is the evidence for or against the constant-wind hypothesis.
    """
    globe = np.asarray(globe_temp_C, dtype=float)
    air = np.asarray(air_temp_C, dtype=float)
    reported = np.asarray(reported_mrt_C, dtype=float)
    delta = globe - air
    usable = (np.isfinite(globe) & np.isfinite(air) & np.isfinite(reported)
              & (np.abs(delta) > 0.5))
    if usable.sum() < 10:
        raise ValueError("not enough usable samples to identify the wind convention")
    implied = (((reported[usable] + 273.15) ** 4 - (globe[usable] + 273.15) ** 4)
               / delta[usable])
    positive = implied[implied > 0.0]
    if positive.size < 10:
        raise ValueError("implied convection term is not positive; "
                         "the reported MRT is not an ISO globe inversion")
    median = float(np.median(positive))
    # median = 1.1e8 * V^0.6 / (eps * D^0.4)  ->  solve for V
    wind = float((median * spec.emissivity * spec.diameter_m ** 0.4 / 1.1e8)
                 ** (1.0 / ISO_FORCED_WIND_EXPONENT))
    spread = float((np.percentile(positive, 75) - np.percentile(positive, 25))
                   / median)
    return {
        "implied_coefficient_median": median,
        "implied_wind_ms": wind,
        "implied_coefficient_iqr_fraction": spread,
        "usable_samples": int(positive.size),
    }


def describe(spec: GlobeSpec) -> str:
    return (f"{spec.name}: D={spec.diameter_m * 1000:.0f} mm, "
            f"eps={spec.emissivity:.3f}, alpha={spec.sw_absorptivity:.3f}, "
            f"C={spec.areal_heat_capacity_J_m2K:.0f} J/m2/K, "
            f"tau~{spec.time_constant_s():.0f} s at 1.5 m/s")


GLOBE_COLUMNS = [
    "globe_absorbed_flux_Wm2",
    "globe_steady_temperature_C",
    "globe_radiative_equilibrium_C",
]


if __name__ == "__main__":  # pragma: no cover - convenience listing
    for spec in GLOBE_PRESETS.values():
        print(describe(spec))
