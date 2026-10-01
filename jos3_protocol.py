"""jos3_protocol.py -- how every JOS-3 walk starts and what it reports.

Shared by stage 09, sensitivity.py and compute_effect_uniform.py so that the
route results, the uniform-meteorology studies and the paper figures all use
one definition.

CORE METRIC. JOS-3 was validated against RECTAL temperature (Takahashi et al.
2021), for which the pelvis core node is the model's counterpart. A
surface-area-weighted mean over all 17 segment cores is not a core
temperature: it includes the arm, leg, hand and foot cores, which cool with
vasoconstriction, and at rest it drifts downward for hours while the pelvis
core settles. ``pelvis`` is therefore the default; ``bsa_mean`` reproduces the
earlier metric.

START STATE. The earlier protocol equilibrated for 10 min at walking activity
in the start point's own outdoor conditions, from JOS-3's default state. That
state is not converged -- much of the reported "rise" was the model still
settling -- and it depends on the start point's sun or shade, which leaks into
any comparison between runs with different radiant fields. ``indoor`` instead
brings the walker to steady state standing in a conditioned building, in the
clothing of the walk, and only then starts walking. Every route, departure
time and counterfactual therefore starts from the same, physically defined
state.
"""
from __future__ import annotations

import numpy as np

CORE_METRICS = ("pelvis", "bsa_mean")
PRECONDITIONING = ("indoor", "outdoor_walk")

# Conditioned indoor space the walker leaves: typical cooled building in a hot
# climate, still air, standing (JOS-3 physical activity ratio 1.2).
INDOOR_AIR_TEMPERATURE_C = 26.0
INDOOR_RH_PCT = 50.0
INDOOR_AIR_SPEED_MS = 0.1
INDOOR_ACTIVITY_PAR = 1.2
# Steady state: pelvis core changing by less than this over one block.
STEADY_TOLERANCE_C = 0.01
STEADY_BLOCK_MIN = 10
MINIMUM_PRECONDITION_MIN = 60
MAXIMUM_PRECONDITION_MIN = 360


def core_temperature(model, metric: str = "pelvis") -> float:
    """Core temperature on the chosen metric, degC."""
    t_core = np.asarray(model.t_core, dtype=float)
    if metric == "pelvis":
        value = float(t_core[list(model.body_names).index("pelvis")])
    elif metric == "bsa_mean":
        weights = np.asarray(model.bsa, dtype=float)
        value = float(np.sum(t_core * weights / weights.sum()))
    else:
        raise ValueError(f"unknown core metric {metric!r}; use {CORE_METRICS}")
    if not np.isfinite(value):
        raise ValueError("non-finite JOS-3 core temperature")
    return value


def precondition_indoor(model, *, air_temperature_c=INDOOR_AIR_TEMPERATURE_C,
                        rh_pct=INDOOR_RH_PCT, air_speed_ms=INDOOR_AIR_SPEED_MS,
                        activity_par=INDOOR_ACTIVITY_PAR,
                        tolerance_c=STEADY_TOLERANCE_C,
                        minimum_min=MINIMUM_PRECONDITION_MIN,
                        maximum_min=MAXIMUM_PRECONDITION_MIN) -> dict:
    """Bring ``model`` (already clothed) to indoor steady state, standing.

    Runs in blocks of STEADY_BLOCK_MIN one-minute steps until the pelvis core
    changes by less than ``tolerance_c`` over a block, after at least
    ``minimum_min``. Raises if that does not happen by ``maximum_min``: a start
    state that is still drifting is exactly what this replaces. The caller sets
    the walking activity afterwards.
    """
    model.par = activity_par
    model.tdb = model.tr = float(air_temperature_c)
    model.rh, model.v = float(rh_pct), float(air_speed_ms)
    elapsed, previous, drift = 0, core_temperature(model, "pelvis"), np.inf
    while elapsed < maximum_min:
        model.simulate(times=STEADY_BLOCK_MIN, dtime=60, output=False)
        elapsed += STEADY_BLOCK_MIN
        current = core_temperature(model, "pelvis")
        drift, previous = abs(current - previous), current
        if elapsed >= minimum_min and drift < tolerance_c:
            return {"precondition_min": elapsed,
                    "final_block_drift_c": float(drift),
                    "pelvis_core_c": float(current)}
    raise RuntimeError(
        f"indoor preconditioning did not reach steady state in {maximum_min} "
        f"min (last {STEADY_BLOCK_MIN}-min drift {drift:.4f} C)")


def add_protocol_arguments(parser) -> None:
    parser.add_argument(
        "--core-metric", choices=CORE_METRICS, default="pelvis",
        help="Reported core temperature: 'pelvis' (rectal-equivalent, the "
             "quantity JOS-3 was validated against; default) or 'bsa_mean' "
             "(surface-area mean of all 17 segment cores; earlier behaviour).")
    parser.add_argument(
        "--precondition", choices=PRECONDITIONING, default="indoor",
        help="Start state: 'indoor' (default) = steady state standing in a "
             f"{INDOOR_AIR_TEMPERATURE_C:g} C conditioned building in the walk's "
             "clothing, then walk; 'outdoor_walk' = the earlier "
             "--equilibration-min minutes at walking activity in the start "
             "point's outdoor conditions.")
