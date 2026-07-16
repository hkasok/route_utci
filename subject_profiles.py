"""
subject_profiles.py -- literature-backed virtual-subject presets for the
JOS-3 route thermal-strain stage (09).

PURPOSE
-------
The default JOS-3 subject is a healthy young adult. Heat-vulnerability,
however, differs by age and health status. This module supplies presets
that adjust ONLY the JOS-3 parameters that are (a) actually exposed by the
model and (b) supported by published physiology. It deliberately does NOT
invent knobs the model does not have.

WHAT JOS-3 EXPOSES (and this module uses)
-----------------------------------------
  height, weight, fat, age, sex   -> body geometry & mass; drive the
                                     surface-area-to-mass ratio that
                                     governs radiative/convective heat
                                     exchange and thermal inertia.
  ci  (cardiac index, L/min/m^2)  -> whole-body perfusion capacity; lower
                                     ci reduces the circulatory support for
                                     skin blood flow / heat convection to
                                     the surface.
  cr_set_point (core setpoint)    -> shifted DOWN a few tenths of a degree
                                     to represent heat ACCLIMATIZATION
                                     (earlier sweating onset / lower
                                     exercising core temperature).

WHAT JOS-3 DOES *NOT* EXPOSE (so we do NOT claim to model it)
------------------------------------------------------------
There is no public scalar in this JOS-3 build for sweat-gland output,
sweating sensitivity/slope, skin-blood-flow gain, hydration/plasma
volume, or medication effects. These are central to REAL elderly and
clinical heat vulnerability (blunted & delayed sweating, reduced skin
vasodilation, diuretics, impaired thirst). Therefore:

  * The elderly/ill presets capture only the BODY-GEOMETRY and PERFUSION
    (ci) part of vulnerability, which UNDER-STATES their true risk.
  * The correct way to express elderly/ill risk is as a SHIFTED SAFETY
    MARGIN and SLOWER RECOVERY, not a larger per-walk core number. The
    epidemiology (heat mortality concentrated in the old and chronically
    ill) is the citation for that; a JOS-3 core delta alone will not show
    the full effect. Each preset carries a `caveat` string stating this.

CITATIONS (values below are chosen to sit inside these published ranges)
------------------------------------------------------------------------
  Cardiac index, healthy adults ~2.6-4.2 L/min/m^2; healthy >60 yr lower,
    ~2.1-3.2 L/min/m^2 (Cioccari et al. 2019, Crit Care Resusc, scoping
    review; consistent with Wikipedia/【CMR normative】summaries). JOS-3
    default ci = 2.59.
  Heat acclimatization lowers the core-temperature threshold for sweating
    onset and reduces exercising core temperature by ~0.3-0.4 C (Shido
    et al. 1999 AJP-Regul; Periard et al. 2016 review; a 4-wk HA study
    measured a 0.4 C reduction, PMC12311370; CDC NIOSH acclimatization).
  Child anthropometry (~8 yr): stature ~1.28 m, mass ~26 kg (WHO/CDC
    growth references, 50th percentile order of magnitude). Children have
    a higher body-surface-area-to-mass ratio -> faster heat exchange with
    a hot radiant environment (Falk & Dotan 2008; standard pediatric
    thermoregulation physiology). JOS-3 represents THIS effect through the
    body geometry.
  Elderly anthropometry values are representative healthy-older values;
    the associated blunted-thermoregulation caveat follows the reviews
    above and the general heat-mortality epidemiology.

USAGE
-----
    from subject_profiles import PROFILES, get_profile
    prof = get_profile("elderly_female")
    model = JOS3(height=prof.height, weight=prof.weight, fat=prof.fat,
                 age=prof.age, sex=prof.sex, ci=prof.ci)
    # apply acclimatization setpoint shift AFTER construction:
    if prof.setpoint_shift_c:
        model.cr_set_point = model.cr_set_point + prof.setpoint_shift_c
"""

from dataclasses import dataclass, field


@dataclass
class SubjectProfile:
    key: str
    label: str
    age: int
    sex: str
    height: float          # m
    weight: float          # kg
    fat: float             # body fat %
    ci: float              # cardiac index, L/min/m^2 (JOS-3 default 2.59)
    setpoint_shift_c: float = 0.0   # core setpoint shift (- = acclimatized)
    rationale: str = ""
    caveat: str = ""


# JOS-3 default ci = 2.59 (used as the healthy-adult anchor).
PROFILES = {
    "healthy_adult": SubjectProfile(
        key="healthy_adult", label="Healthy adult (35 yr, male)",
        age=35, sex="male", height=1.72, weight=70.0, fat=15.0, ci=3.0,
        rationale="Reference subject; ci ~3.0 mid healthy-adult range "
                  "(2.6-4.2 L/min/m^2).",
        caveat=""),

    "healthy_adult_female": SubjectProfile(
        key="healthy_adult_female", label="Healthy adult (35 yr, female)",
        age=35, sex="female", height=1.62, weight=62.0, fat=27.0, ci=3.0,
        rationale="Female reference; ci indexed to BSA so ~same as male.",
        caveat=""),

    "child": SubjectProfile(
        key="child", label="Child (~8 yr)",
        age=8, sex="male", height=1.28, weight=26.0, fat=18.0, ci=3.4,
        rationale="~8-yr anthropometry (WHO/CDC 50th-pct order). Higher "
                  "surface-area-to-mass ratio -> faster heat gain from a "
                  "hot radiant field; children's ci runs higher than "
                  "adults. Captured via body geometry.",
        caveat="Children also sweat less efficiently (higher core "
               "threshold for sweating, lower sweat rate per gland) and "
               "self-regulate behaviour poorly -- effects JOS-3 does not "
               "model, so real risk is somewhat higher than the geometry "
               "alone implies."),

    "elderly_male": SubjectProfile(
        key="elderly_male", label="Elderly (75 yr, male)",
        age=75, sex="male", height=1.70, weight=72.0, fat=25.0, ci=2.4,
        rationale="Healthy >60-yr ci lowered to ~2.4 (published 2.1-3.2 "
                  "L/min/m^2 range); higher body-fat fraction typical of "
                  "aging body composition.",
        caveat="Captures only reduced perfusion + body composition. Real "
               "elderly heat risk is dominated by BLUNTED/DELAYED SWEATING, "
               "reduced skin blood flow, impaired thirst, and medications "
               "-- NOT exposed by JOS-3. Interpret as a shifted safety "
               "margin and slower recovery, not a larger per-walk core "
               "rise; heat mortality epidemiology is the basis for the "
               "elevated risk."),

    "elderly_female": SubjectProfile(
        key="elderly_female", label="Elderly (78 yr, female)",
        age=78, sex="female", height=1.58, weight=62.0, fat=35.0, ci=2.3,
        rationale="Healthy >60-yr ci ~2.3 (2.1-3.2 range); female body "
                  "composition.",
        caveat="Same limitation as elderly_male: JOS-3 does not model the "
               "impaired sweating / skin-blood-flow / hydration that drive "
               "real elderly heat risk. Treat as margin-and-recovery, not "
               "a bigger core number."),

    "obese_adult": SubjectProfile(
        key="obese_adult", label="Obese adult (45 yr, male, BMI ~34)",
        age=45, sex="male", height=1.75, weight=104.0, fat=35.0, ci=2.7,
        rationale="Low surface-area-to-mass ratio + high thermal mass "
                  "buffer a SHORT walk (smaller transient rise) but impair "
                  "heat loss over longer exposure; ci modestly reduced.",
        caveat="A short-walk core delta UNDER-STATES obese heat risk: the "
               "low SA:mass ratio that damps the transient also limits "
               "sustained heat dissipation and raises cardiovascular load. "
               "Compare longer exposures, not just the 25-min snapshot."),

    "acclimatized_adult": SubjectProfile(
        key="acclimatized_adult",
        label="Heat-acclimatized adult (Miami resident)",
        age=35, sex="male", height=1.72, weight=70.0, fat=15.0, ci=3.2,
        setpoint_shift_c=-0.3,
        rationale="Represents a habituated local: core setpoint shifted "
                  "-0.3 C (published acclimatization effect ~0.3-0.4 C: "
                  "earlier sweating onset, lower exercising core temp; "
                  "Shido 1999, Periard 2016, CDC) and slightly higher ci "
                  "from plasma-volume expansion.",
        caveat="Acclimatization is modeled here only as a setpoint shift; "
               "JOS-3 does not expose sweat-rate/plasma-volume gains, so "
               "this is a conservative (partial) representation of the "
               "real benefit."),
}


def get_profile(key):
    if key not in PROFILES:
        raise KeyError(f"unknown subject profile '{key}'. "
                       f"Choices: {', '.join(PROFILES)}")
    return PROFILES[key]


def apply_profile_to_model(model, prof):
    """Apply the acclimatization setpoint shift (anthropometry + ci are set
    in the JOS3 constructor). Returns the model for chaining."""
    if prof.setpoint_shift_c:
        model.cr_set_point = model.cr_set_point + prof.setpoint_shift_c
    return model
