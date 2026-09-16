"""wind_profile.py -- urban canopy wind profile from measured cart wind.

WHY THIS EXISTS
---------------
The Lisbon cart measures wind at roughly 1 m. Two different quantities are then
wanted from that one number, and they are NOT the same:

  * the APPROACH wind at pedestrian height, which is the boundary condition the
    2-D potential-flow solve needs at its domain edge;
  * the FREE-STREAM reference velocity, which is what an ``a + b*U`` convection
    correlation was calibrated against.

Both must come from the experiment, not from the flow model they go on to
drive. Deriving the inlet by inverting a measurement through the model's own
amplification makes the boundary condition depend on the solution it produces.

WHY A PLAIN LOG LAW IS WRONG HERE
---------------------------------
For lisbon1 the buildings have a median height H ~ 19 m and a plan area fraction
of 0.14, which by Macdonald puts the displacement height at d ~ 5.7 m. A sensor
at 1 m is therefore FAR BELOW d -- deep inside the canopy, where the
logarithmic surface-layer profile does not exist. Lifting 1 m to 10 m with
``ln((z-d)/z0)`` would take the logarithm of a negative number, and clamping it
would silently invent a value.

Worse, the conventional "10 m meteorological wind" is itself BELOW roof level at
this site. There is no height at which the familiar shortcut is both valid and
conventional.

THE PROFILE ACTUALLY USED
-------------------------
A two-layer canopy profile, matched at roof level:

    z <  H : u(z) = u(H) * exp[a (z/H - 1)]        (Cionco exponential)
    z >= H : u(z) = (u* / kappa) * ln((z - d)/z0)  (log, above the canopy)

with u* fixed by continuity at z = H. The morphology terms come from the actual
building geometry of the case rather than from a table:

    d/H  = 1 + A^(-lambda_p) (lambda_p - 1)                  Macdonald (1998)
    z0/H = (1 - d/H) exp[-(0.5 beta Cd/kappa^2 (1-d/H) lambda_f)^(-1/2)]
    a    ~ 9.6 lambda_f                                      Macdonald (2000)

WHAT THIS IS AND IS NOT
-----------------------
It is a spatially averaged NEIGHBOURHOOD profile: the mean vertical structure
over the urban array, not the wind at any particular street. Local sheltering by
individual buildings is the potential-flow solve's job, and applying both to the
same measurement would count the obstruction twice. That separation of duties is
the whole point -- profile for the vertical, potential flow for the horizontal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

VON_KARMAN = 0.4
MACDONALD_A = 4.43
MACDONALD_BETA = 1.0
MACDONALD_DRAG = 1.2
# Cionco attenuation from frontal area, Macdonald (2000). Clamped to the range
# the urban literature actually reports; outside it the exponential either fails
# to attenuate at all or collapses to zero at street level.
ATTENUATION_FROM_FRONTAL = 9.6
ATTENUATION_RANGE = (0.5, 4.0)

# MINIMUM street-to-roof wind ratio a MEASUREMENT may be assumed to sit at.
#
# The Cionco exponential is a canopy-AVERAGE parameterisation: it describes the
# mean wind through the whole built volume, including the sheltered interiors of
# blocks. A measurement cart walks the OPEN streets, which are the ventilated
# part of that volume. Applied literally to a dense canopy the average profile
# puts street level at 3-9% of roof level, and INVERTING it then multiplies the
# measurement by 11-26x -- lisbon2 came out at 51 m/s free stream and an
# implied h of 200 W/m2K.
#
# The inverse is the ill-conditioned direction: going DOWN from a known
# above-canopy wind damps errors, going UP from a street reading amplifies them
# by 1/ratio. Canyon observations put street-level wind at roughly 0.3-0.5 of
# roof level, so the ratio used for a street measurement is floored here. This
# is a documented bound on an ill-posed inversion, not a tuned constant, and
# whenever it binds the caller is told.
MINIMUM_STREET_TO_ROOF_RATIO = 0.30


class WindProfileError(ValueError):
    """Raised when a profile cannot be evaluated on physical grounds."""


@dataclass(frozen=True)
class CanopyMorphology:
    """Neighbourhood-average urban canopy geometry."""

    building_height_m: float
    plan_area_fraction: float
    frontal_area_index: float
    displacement_height_m: float
    roughness_length_m: float
    attenuation: float
    source: str = ""

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        return (f"H={self.building_height_m:.1f} m, lambda_p="
                f"{self.plan_area_fraction:.3f}, lambda_f="
                f"{self.frontal_area_index:.3f}, d="
                f"{self.displacement_height_m:.1f} m, z0="
                f"{self.roughness_length_m:.2f} m, a={self.attenuation:.2f}")


def macdonald_displacement(height_m: float, plan_area_fraction: float) -> float:
    """Displacement height, Macdonald et al. (1998)."""
    lam = float(np.clip(plan_area_fraction, 1e-4, 0.95))
    return float(height_m * (1.0 + MACDONALD_A ** (-lam) * (lam - 1.0)))


def macdonald_roughness(height_m: float, displacement_m: float,
                        frontal_area_index: float) -> float:
    """Aerodynamic roughness length, Macdonald et al. (1998)."""
    height_m = float(height_m)
    ratio = 1.0 - float(displacement_m) / max(height_m, 1e-6)
    ratio = float(np.clip(ratio, 1e-3, 1.0))
    lam_f = max(float(frontal_area_index), 1e-4)
    inner = (0.5 * MACDONALD_BETA * MACDONALD_DRAG / VON_KARMAN ** 2
             * ratio * lam_f)
    return float(height_m * ratio * math.exp(-(inner ** -0.5)))


def canopy_morphology(building_height_m: float, plan_area_fraction: float,
                      frontal_area_index: float,
                      source: str = "") -> CanopyMorphology:
    """Assemble the morphology terms the profile needs."""
    if building_height_m <= 0.0:
        raise WindProfileError("canopy height must be positive")
    displacement = macdonald_displacement(building_height_m, plan_area_fraction)
    roughness = macdonald_roughness(building_height_m, displacement,
                                    frontal_area_index)
    attenuation = float(np.clip(ATTENUATION_FROM_FRONTAL * frontal_area_index,
                                *ATTENUATION_RANGE))
    return CanopyMorphology(
        building_height_m=float(building_height_m),
        plan_area_fraction=float(plan_area_fraction),
        frontal_area_index=float(frontal_area_index),
        displacement_height_m=displacement,
        roughness_length_m=roughness,
        attenuation=attenuation,
        source=source)


def morphology_from_geometry(buildings_stl: str, cell_fluid_fraction=None,
                             wind_direction_deg: float = 270.0,
                             domain_area_m2: float | None = None
                             ) -> CanopyMorphology:
    """Derive the morphology from the case's own building mesh.

    ``lambda_p`` is taken from the potential-flow cut-cell fluid fraction when
    available -- that array already encodes exactly which plan area the
    buildings occupy at pedestrian level, so reusing it avoids a second,
    possibly disagreeing, definition of the same quantity.

    ``lambda_f`` is the frontal area facing the wind divided by the plan area,
    computed from the mesh faces rather than assumed.
    """
    import trimesh

    mesh = trimesh.load(buildings_stl, process=False, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    if len(faces) == 0:
        raise WindProfileError(f"{buildings_stl} contains no geometry")

    import surface_groups as sg
    labels = sg.connected_object_labels(faces, len(faces), vertices=vertices)
    heights = []
    for index in range(int(labels.max()) + 1):
        member = np.unique(faces[labels == index])
        if member.size:
            z = vertices[member, 2]
            # HEIGHT ABOVE ITS OWN BASE, not absolute elevation. A building mesh
            # spans from local ground to roof, so max - min is its height; the
            # bare maximum is terrain elevation plus height and is meaningless
            # on a hill. Lisbon is hilly: using absolute tops gave a "canopy
            # height" of 92 m for lisbon3 and a 57x profile lift for lisbon2,
            # against true median building heights of 21 m and 17 m.
            heights.append(float(z.max() - z.min()))
    if not heights:
        raise WindProfileError("no building objects found")
    # Median rather than mean: a handful of towers should not set the
    # neighbourhood canopy height for a low-rise district.
    height = float(np.median(heights))
    if not 2.0 <= height <= 120.0:
        raise WindProfileError(
            f"derived canopy height {height:.1f} m is outside the plausible "
            "urban range 2-120 m; check that the building mesh is a set of "
            "separate buildings and not a merged shell")

    if cell_fluid_fraction is not None:
        plan_fraction = float(1.0 - np.asarray(cell_fluid_fraction,
                                               dtype=float).mean())
    elif domain_area_m2:
        footprint = float(mesh.area_faces[
            np.asarray(mesh.face_normals)[:, 2] > 0.7].sum())
        plan_fraction = footprint / float(domain_area_m2)
    else:
        raise WindProfileError(
            "plan area fraction needs either the cut-cell fluid fraction or a "
            "domain area")
    plan_fraction = float(np.clip(plan_fraction, 1e-4, 0.9))

    # Frontal area: wall area projected onto the plane normal to the wind.
    bearing = math.radians(float(wind_direction_deg))
    flow = np.array([-math.sin(bearing), -math.cos(bearing)])
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    walls = np.abs(normals[:, 2]) <= 0.7
    frontal = float(np.sum(areas[walls]
                           * np.abs(normals[walls, :2] @ flow)))
    plan_area = (float(np.asarray(cell_fluid_fraction).size)
                 if cell_fluid_fraction is None else None)
    if domain_area_m2:
        total_area = float(domain_area_m2)
    else:
        span = vertices[:, :2].max(axis=0) - vertices[:, :2].min(axis=0)
        total_area = float(span[0] * span[1])
    # Half, because summing |n.d| over a closed body counts windward and leeward.
    frontal_index = float(np.clip(0.5 * frontal / max(total_area, 1e-6),
                                  1e-4, 0.9))
    return canopy_morphology(
        height, plan_fraction, frontal_index,
        source=f"derived from {buildings_stl} at wind {wind_direction_deg:g} deg")


def friction_velocity(reference_speed_ms, reference_height_m: float,
                      morphology: CanopyMorphology):
    """u* implied by a wind speed given ABOVE the canopy."""
    z = float(reference_height_m)
    if z <= morphology.displacement_height_m + morphology.roughness_length_m:
        raise WindProfileError(
            f"reference height {z:.1f} m is not above d + z0 "
            f"({morphology.displacement_height_m:.1f} + "
            f"{morphology.roughness_length_m:.2f} m); the log layer does not "
            "exist there")
    denominator = math.log((z - morphology.displacement_height_m)
                           / morphology.roughness_length_m)
    return VON_KARMAN * np.asarray(reference_speed_ms, dtype=float) / denominator


def speed_at_height(speed_ms, from_height_m: float, to_height_m: float,
                    morphology: CanopyMorphology):
    """Move a wind speed between two heights through the canopy profile.

    Handles all four combinations of in-canopy and above-canopy endpoints by
    routing through the roof-level speed ``u(H)``, which is where the
    exponential and logarithmic branches are matched.
    """
    speed = np.asarray(speed_ms, dtype=float)
    if np.any(speed < 0.0) or not np.isfinite(speed).all():
        raise WindProfileError("wind speed must be finite and non-negative")
    height = morphology.building_height_m

    def to_roof(value, z):
        if z >= height:
            return value / _log_factor(z, morphology)
        return value / in_canopy_ratio(z, morphology)

    def from_roof(value, z):
        if z >= height:
            return value * _log_factor(z, morphology)
        return value * in_canopy_ratio(z, morphology)

    roof = to_roof(speed, float(from_height_m))
    return from_roof(roof, float(to_height_m))


def in_canopy_ratio(z: float, morphology: CanopyMorphology) -> float:
    """u(z)/u(H) inside the canopy, floored for invertibility.

    The raw Cionco exponential is kept wherever it is above the floor. The floor
    only binds in dense canopies, where the canopy-average profile would put a
    street measurement at a few percent of roof level and inverting it would
    amplify the reading by an order of magnitude.
    """
    height = morphology.building_height_m
    raw = math.exp(morphology.attenuation * (float(z) / height - 1.0))
    return max(raw, MINIMUM_STREET_TO_ROOF_RATIO)


def street_ratio_is_floored(z: float, morphology: CanopyMorphology) -> bool:
    """Whether the invertibility floor binds at this height."""
    height = morphology.building_height_m
    raw = math.exp(morphology.attenuation * (float(z) / height - 1.0))
    return bool(raw < MINIMUM_STREET_TO_ROOF_RATIO)


def _log_factor(z: float, morphology: CanopyMorphology) -> float:
    """u(z)/u(H) on the logarithmic branch, z >= H."""
    height = morphology.building_height_m
    displacement = morphology.displacement_height_m
    roughness = morphology.roughness_length_m
    numerator = math.log(max(z - displacement, roughness * 1.01) / roughness)
    denominator = math.log(max(height - displacement, roughness * 1.01)
                           / roughness)
    return numerator / max(denominator, 1e-9)


def free_stream_speed(measured_speed_ms, measured_height_m: float,
                      morphology: CanopyMorphology,
                      reference_height_m: float | None = None):
    """Free-stream reference velocity for an ``a + b*U`` convection correlation.

    The correlation was calibrated against the UNDISTURBED approach velocity of
    a wind tunnel, so the urban analogue is the wind above the roughness
    sublayer rather than the sheltered value inside the street. Default
    reference height is 2H, the usual blending height at which the flow has
    lost memory of individual buildings.
    """
    if reference_height_m is None:
        reference_height_m = 2.0 * morphology.building_height_m
    return speed_at_height(measured_speed_ms, measured_height_m,
                           float(reference_height_m), morphology)


def approach_speed_at_pedestrian_height(measured_speed_ms,
                                        measured_height_m: float,
                                        morphology: CanopyMorphology,
                                        pedestrian_height_m: float = 1.1):
    """Neighbourhood approach wind at pedestrian height, for the flow BC.

    This is the inlet the 2-D potential-flow solve wants: the horizontally
    averaged wind entering the domain at its sampling height, WITHOUT any
    local sheltering removed -- the solve itself produces the sheltering, and
    correcting for it here as well would count the buildings twice.

    When the sensor and pedestrian heights coincide, this is the identity.
    """
    return speed_at_height(measured_speed_ms, measured_height_m,
                           float(pedestrian_height_m), morphology)


def profile_report(morphology: CanopyMorphology, measured_height_m: float,
                   heights_m=(1.0, 1.1, 2.0, 10.0, None, None)
                   ) -> list[tuple[float, float]]:
    """Profile shape as (height, speed ratio to the measured height)."""
    levels = [morphology.building_height_m if h is None else float(h)
              for h in heights_m if h is not None]
    levels += [morphology.building_height_m, 2.0 * morphology.building_height_m]
    out = []
    for z in sorted(set(levels)):
        ratio = float(speed_at_height(1.0, measured_height_m, z, morphology))
        out.append((z, ratio))
    return out
