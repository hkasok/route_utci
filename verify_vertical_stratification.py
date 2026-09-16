#!/usr/bin/env python3
"""
verify_vertical_stratification.py -- checks for vertical_stratification.py
and its wiring into 02_vegetation_to_stl.py.

Exits nonzero on failure, like every other verify_* suite in this repo.

The property that matters is not "cuts happen" but "cuts happen for the
right reason and are refused for the right reason". A detector that splits
enthusiastically would sever every trunk and turn one tree into two floating
blobs, which is a worse geometry error than the merge it set out to fix. So
the trunk guard gets as much coverage here as the detection does.
"""

import subprocess
import sys

import numpy as np

import vertical_stratification as vs

FAILURES = []
N_CHECKS = 0


def check(name, condition, detail=""):
    global N_CHECKS
    N_CHECKS += 1
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def rng():
    return np.random.default_rng(20260829)


def blob(n, centre, radius, z_lo, z_hi, generator):
    """A cylinder-ish cloud of n points."""
    theta = generator.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(generator.uniform(0, 1, n))
    return np.column_stack([
        centre[0] + r * np.cos(theta),
        centre[1] + r * np.sin(theta),
        generator.uniform(z_lo, z_hi, n),
    ])


# ---------------------------------------------------------------- A: width
print("\n[A] lateral_width")
g = rng()
trunk = blob(200, (0, 0), 0.2, 0, 10, g)
check("A1 trunk reads under the guard width",
      vs.lateral_width(trunk) < 1.5, f"got {vs.lateral_width(trunk):.2f}")
hedge = blob(500, (0, 0), 4.0, 0, 2, g)
check("A2 hedge reads well above the guard width",
      vs.lateral_width(hedge) > 3.0, f"got {vs.lateral_width(hedge):.2f}")
check("A3 empty input is 0 not a crash", vs.lateral_width(np.zeros((0, 3))) == 0.0)
# The quantile (not the max) is what keeps a few strays from disabling the guard.
strayed = np.vstack([trunk, np.array([[30.0, 0.0, 5.0]])])
check("A4 a stray return does not inflate a trunk past the guard",
      vs.lateral_width(strayed) < 1.5, f"got {vs.lateral_width(strayed):.2f}")


# ----------------------------------------------------------------- B: gaps
print("\n[B] find_vertical_gaps")
g = rng()
two = np.vstack([blob(400, (0, 0), 4.0, 0, 2, g), blob(400, (0, 0), 4.0, 10, 16, g)])
gaps = vs.find_vertical_gaps(two[:, 2])
check("B1 finds exactly one void between two separated layers", len(gaps) == 1, f"got {len(gaps)}")
if gaps:
    lo, hi = gaps[0]
    check("B2 void is located between the layers", 1.5 < lo and hi < 10.5, f"got {lo:.1f}-{hi:.1f}")
    check("B3 void height is about 8 m", 6.5 < (hi - lo) < 9.0, f"got {hi - lo:.1f}")

solid = blob(800, (0, 0), 4.0, 0, 16, g)
check("B4 a continuous column has no void", len(vs.find_vertical_gaps(solid[:, 2])) == 0)

# A sparse run at an end is a density tail, not a separation.
tail = np.vstack([blob(600, (0, 0), 4.0, 10, 16, g), np.array([[0.0, 0.0, 0.0]])])
check("B5 a sparse tail at the bottom is not reported as a void",
      len(vs.find_vertical_gaps(tail[:, 2])) == 0,
      f"got {vs.find_vertical_gaps(tail[:, 2])}")

narrow = np.vstack([blob(400, (0, 0), 4.0, 0, 2, g), blob(400, (0, 0), 4.0, 2.8, 8, g)])
check("B6 a void shorter than min_gap_m is ignored",
      len(vs.find_vertical_gaps(narrow[:, 2])) == 0,
      f"got {vs.find_vertical_gaps(narrow[:, 2])}")
check("B7 the same void is found once min_gap_m is lowered",
      len(vs.find_vertical_gaps(narrow[:, 2], vs.StratificationSettings(min_gap_m=0.3))) == 1)
check("B8 empty input returns no voids", vs.find_vertical_gaps(np.zeros(0)) == [])

# Bins are physical, so detection must not depend on the total span.
far = np.vstack([blob(400, (0, 0), 4.0, 0, 2, g), blob(400, (0, 0), 4.0, 10, 60, g)])
check("B9 detection is unaffected by a much taller upper layer",
      len(vs.find_vertical_gaps(far[:, 2])) == 1, f"got {len(vs.find_vertical_gaps(far[:, 2]))}")


# ------------------------------------------------------------- C: splitting
print("\n[C] split_vertical_strata -- the merge case")
g = rng()
hedge = blob(500, (0, 0), 4.0, 0.0, 2.0, g)
canopy = blob(900, (0, 0), 5.0, 10.0, 18.0, g)
merged = np.vstack([hedge, canopy])
strata, rep = vs.split_vertical_strata(merged)
check("C1 the merged pair splits into two strata", len(strata) == 2, f"got {len(strata)}")
check("C2 one cut recorded", rep.n_cuts == 1, f"got {rep.n_cuts}")
if len(strata) == 2:
    low, high = strata
    check("C3 strata are ordered bottom-up", low.z_low < high.z_low)
    check("C4 lower stratum keeps the hedge's depth",
          low.thickness_m < 3.0, f"got {low.thickness_m:.1f}")
    check("C5 upper stratum keeps the canopy's depth",
          7.0 < high.thickness_m < 9.0, f"got {high.thickness_m:.1f}")
    check("C6 no stratum spans the void",
          max(s.thickness_m for s in strata) < 0.9 * merged[:, 2].ptp())
    check("C7 point counts are preserved up to the discarded gap points",
          sum(s.n_points for s in strata) + rep.n_points_dropped == len(merged))

# Three layers -> three strata.
g = rng()
three = np.vstack([blob(400, (0, 0), 4.0, 0, 2, g),
                   blob(400, (0, 0), 4.0, 8, 11, g),
                   blob(400, (0, 0), 4.0, 18, 24, g)])
strata3, rep3 = vs.split_vertical_strata(three)
check("C8 three separated layers give three strata", len(strata3) == 3, f"got {len(strata3)}")
check("C9 two cuts recorded", rep3.n_cuts == 2, f"got {rep3.n_cuts}")


# ------------------------------------------------------------ D: trunk guard
print("\n[D] the trunk guard")
g = rng()
# A tree: narrow trunk under a wide crown, with the sparse return pattern a
# real airborne scan gives. This MUST survive as one object.
tree = np.vstack([blob(25, (0, 0), 0.2, 0.0, 9.0, g), blob(900, (0, 0), 5.0, 10.0, 18.0, g)])
strata_t, rep_t = vs.split_vertical_strata(tree)
check("D1 a trunk-and-crown tree is NOT split", len(strata_t) == 1, f"got {len(strata_t)}")
# A sparse trunk running down to the bottom of the cluster is one sparse run
# TOUCHING THE BOTTOM, so the interior-only rule dismisses it before any guard
# is consulted. The tree is safe; it is simply never a candidate.
check("D2 such a tree never becomes a split candidate at all",
      rep_t.n_gaps_found == 0, f"got {rep_t.n_gaps_found} gaps")
check("D3 a refused cut discards no points",
      rep_t.n_points_dropped == 0 and strata_t[0].n_points == len(tree))
check("D4 the refused tree keeps its full height",
      abs(strata_t[0].thickness_m - tree[:, 2].ptp()) < 1e-9)

# Same geometry but the lower part is wide -> it is a plant, and it splits.
check("D5 widening the lower part alone turns the refusal into a cut",
      len(vs.split_vertical_strata(
          np.vstack([blob(300, (0, 0), 4.0, 0.0, 2.0, g),
                     blob(900, (0, 0), 5.0, 10.0, 18.0, g)]))[0]) == 2)

# Where the trunk guard actually bears: a DENSE, NARROW lower object -- a
# stem base or pole -- is dense enough to anchor the bottom (so the void
# above it is interior and a real candidate) yet is not a separate plant.
# Only the width measurement can tell it apart from a hedge.
g = rng()
pole = np.vstack([blob(200, (0, 0), 0.2, 0.0, 2.0, g), blob(900, (0, 0), 5.0, 10.0, 18.0, g)])
strata_p, rep_p = vs.split_vertical_strata(pole)
check("D6 a dense narrow stem base DOES produce a candidate void",
      rep_p.n_gaps_found == 1, f"got {rep_p.n_gaps_found}")
check("D7 the trunk guard refuses that cut", rep_p.n_refused_trunk == 1
      and len(strata_p) == 1, f"trunk={rep_p.n_refused_trunk} strata={len(strata_p)}")
check("D8 the guard is a real threshold, not a constant refusal",
      len(vs.split_vertical_strata(
          pole, vs.StratificationSettings(trunk_width_m=0.1))[0]) == 2)
check("D9 widening only that lower object flips the same case to a cut",
      len(vs.split_vertical_strata(
          np.vstack([blob(200, (0, 0), 4.0, 0.0, 2.0, g),
                     blob(900, (0, 0), 5.0, 10.0, 18.0, g)]))[0]) == 2)

# Population guard. The lower group must be dense enough to read as a real
# bin (otherwise the interior-only rule dismisses it as a tail before any
# guard runs) and wide enough to clear the trunk guard, but still too small
# to reconstruct -- so the POPULATION guard is what refuses the cut.
g = rng()
strays = np.vstack([blob(10, (0, 0), 4.0, 0.0, 0.4, g), blob(600, (0, 0), 5.0, 10, 18, g)])
check("D10 the sparse lower group does produce a candidate void",
      len(vs.find_vertical_gaps(strays[:, 2])) == 1,
      f"got {vs.find_vertical_gaps(strays[:, 2])}")
s_str, r_str = vs.split_vertical_strata(strays)
check("D11 too few points below refuses the cut", len(s_str) == 1, f"got {len(s_str)}")
check("D12 the refusal is attributed to the population guard",
      r_str.n_refused_population >= 1 and r_str.n_refused_trunk == 0,
      f"pop={r_str.n_refused_population} trunk={r_str.n_refused_trunk}")


# ------------------------------------------------------- E: totality / flags
print("\n[E] total function, pass-through, and low vegetation")
g = rng()
single = blob(500, (0, 0), 4.0, 10, 18, g)
check("E1 an unsplittable cluster returns exactly one stratum",
      len(vs.split_vertical_strata(single)[0]) == 1)
check("E2 empty input returns no strata and does not raise",
      vs.split_vertical_strata(np.zeros((0, 3)))[0] == [])
check("E3 a single point returns one stratum",
      len(vs.split_vertical_strata(np.array([[0.0, 0.0, 5.0]]))[0]) == 1)

off = vs.StratificationSettings(enabled=False)
s_off, r_off = vs.split_vertical_strata(merged, off)
check("E4 enabled=False is an exact pass-through",
      len(s_off) == 1 and s_off[0].n_points == len(merged) and r_off.n_cuts == 0)

# Low-vegetation tagging needs height above ground; without it, no tag.
hag = merged[:, 2] - 0.0
s_tag, _ = vs.split_vertical_strata(merged, height_above_ground=hag)
check("E5 the hedge stratum is tagged low vegetation",
      s_tag[0].is_low_vegetation is True)
check("E6 the canopy stratum is not tagged low vegetation",
      s_tag[1].is_low_vegetation is False)
s_no, _ = vs.split_vertical_strata(merged)
check("E7 without height-above-ground nothing is tagged",
      not any(s.is_low_vegetation for s in s_no))
check("E8 raising the threshold above the canopy tags both",
      all(s.is_low_vegetation for s in vs.split_vertical_strata(
          merged, vs.StratificationSettings(low_vegetation_max_height_m=100.0),
          height_above_ground=hag)[0]))

clusters = [merged, single, tree]
flat, total = vs.stratify_clusters(clusters)
check("E9 stratify_clusters flattens every cluster",
      len(flat) == 2 + 1 + 1, f"got {len(flat)}")
check("E10 the accumulated report counts every cut",
      total.n_cuts == 1 and total.n_strata == len(flat))


# ------------------------------------------------------------ F: integration
print("\n[F] 02_vegetation_to_stl.py wiring")
proc = subprocess.run([sys.executable, "02_vegetation_to_stl.py", "--help"],
                      capture_output=True, text=True)
check("F1 --help still runs", proc.returncode == 0, proc.stderr[-300:])
helptext = proc.stdout
for flag in ("--vertical-stratification", "--strat-min-gap", "--strat-trunk-width",
             "--low-vegetation-max-height", "--drop-low-vegetation"):
    check(f"F2 {flag} is exposed", flag in helptext)
check("F3 stratification defaults to on", "default: on" in helptext)

src = open("02_vegetation_to_stl.py").read()
check("F4 stratification runs AFTER the watershed, not before",
      src.index("segment_tree_crowns(") < src.index("split_vertical_strata("))
check("F5 low vegetation is meshed, not dropped, unless asked",
      "args.drop_low_vegetation" in src and "action=\"store_true\"" in src)
check("F6 the ground raster is shared with the height filter",
      "build_ground_height_grid" in src and "height_above_ground(pts" in src)


print(f"\n{'=' * 62}")
if FAILURES:
    print(f"FAILED {len(FAILURES)}/{N_CHECKS}: " + ", ".join(FAILURES))
    sys.exit(1)
print(f"All {N_CHECKS} vertical-stratification checks passed.")
