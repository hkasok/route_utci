"""
vertical_stratification.py -- split vertically merged vegetation clusters
into physically separate strata.

WHY THIS EXISTS
---------------
Every stage that decides "what is one tree" in this pipeline works in plan
view. `grid_cluster.grid_cluster_2d` takes XY only -- the z coordinate is
never passed in. `tree_crown_segmentation.rasterize_chm` builds a canopy
height model as max-Z per cell, so only the topmost return in a cell
participates in the watershed. Between them the pipeline can split
neighbouring trees standing SIDE BY SIDE, but it has no operator at all
that splits a vertical column into strata.

The consequence is structural, not a tuning problem: anything sharing an
XY footprint with a canopy -- a hedge, a shrub bed, understory -- is
absorbed into that canopy by construction. `ground_height_filter` mitigates
only the ground-hugging part of this (its docstring describes the same
"solid pillar from ground to canopy top" artifact); its default 1.5 m cut
removes grass but leaves any shrub or hedge taller than that merged into
the tree above it.

Measured on the real lisbon cloud (250 m region, 628,760 points, 58 macro
clusters): 44% of clusters span more than 15 m vertically, and 23 clusters
show an interior void of 4-9 m between a low stratum and the canopy. The
benign explanation -- that the low stratum is the TRUNK -- was tested and
rejected: measuring the lateral width of the low stratum in every gap
cluster gave 0 narrower than 1.5 m and 23 wider than 3.0 m, with widths
running 3.6 m to 105 m. They are separate plants, not trunks.

The downstream damage is to shade, which is the reason the vegetation mesh
exists at all. A watertight crown spanning 9-31 m is opaque through the
8 m void inside it, so the Beer-Lambert entry/exit pairing in the MRT ray
trace sees a canopy path length roughly 2-3x too long. The route is
over-shaded and MRT reads low. The error enters BEFORE crown reconstruction,
so no amount of reconstruction quality removes it.

METHOD
------
Histogram z in fixed-size physical bins, find interior runs of sparse bins
long enough to be a real void, and cut there -- subject to guards that stop
the cut from doing something worse than the merge it fixes:

  * TRUNK GUARD. A trunk produces exactly the signature we are cutting on:
    a few sparse returns spanning the space between ground and crown. The
    discriminator is lateral width -- a trunk is under ~1.5 m across, a
    shrub bed is metres. If the lower side of a candidate cut is narrower
    than `trunk_width_m` it is a trunk, and the cut is refused so the tree
    stays whole.
  * POPULATION GUARD. Both sides must carry enough points to reconstruct;
    otherwise the cut just sheds a sliver of noise.

Points lying INSIDE an accepted gap are dropped rather than assigned to
either side. They are the sparse trunk returns, and giving them to the
upper stratum would re-create the exact pillar being removed, while giving
them to the lower one would grow a spike out of the top of a hedge. The
physical cost is small and bounded -- a trunk's ~0.3 m2 silhouette against
a crown's tens of m2 -- and the count is reported rather than hidden.

Bins are fixed in METRES, not quantiles: a decile-based histogram rescales
itself to whatever span the merge produced, so the very artifact being
detected would resize the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StratificationSettings:
    """Tunables for vertical stratification.

    Defaults are set from the measured lisbon gap statistics: observed
    interior voids run 4-9 m, so a 2.0 m minimum gap is comfortably
    conservative, and observed low-stratum widths are all >3.6 m while a
    trunk is <1.5 m, so the two populations are cleanly separated.
    """

    bin_size_m: float = 0.5
    """Height of one z histogram bin, in metres. Fixed physical size."""

    min_gap_m: float = 2.0
    """A run of sparse bins must be at least this tall to count as a void."""

    sparse_fraction: float = 0.10
    """A bin is sparse below this fraction of a DENSE bin's count.

    "Dense" is the 90th percentile of occupied bins, not the median. The
    median is unusable here because it moves with the RATIO of sparse bins
    to dense ones: a tall trunk contributes many nearly-empty bins, dragging
    the median down until the trunk's own bins no longer read as sparse and
    the void vanishes from the detector -- so the tree would pass the trunk
    guard by never reaching it. The 90th percentile tracks the canopy
    regardless of how much trunk sits below it.
    """

    sparse_floor_points: int = 1
    """A bin holding at most this many points is always sparse."""

    min_stratum_points: int = 20
    """Both sides of a cut must carry at least this many points."""

    trunk_width_m: float = 1.5
    """Lower side narrower than this is a trunk -- refuse the cut."""

    width_quantile: float = 0.90
    """Quantile of radial distance used for the lateral width measure."""

    low_vegetation_max_height_m: float = 3.0
    """A stratum whose top is below this height above ground is low vegetation."""

    enabled: bool = True
    """False makes `split_vertical_strata` a pass-through."""


@dataclass
class Stratum:
    """One vertically separated piece of a cluster."""

    points: np.ndarray
    z_low: float
    z_high: float
    is_low_vegetation: bool = False
    top_height_above_ground_m: Optional[float] = None

    @property
    def n_points(self) -> int:
        return int(len(self.points))

    @property
    def thickness_m(self) -> float:
        return float(self.z_high - self.z_low)


@dataclass
class StratificationReport:
    """What stratification did to one cluster."""

    n_strata: int = 1
    n_cuts: int = 0
    n_gaps_found: int = 0
    n_refused_trunk: int = 0
    n_refused_population: int = 0
    n_points_dropped: int = 0
    gap_heights_m: List[float] = field(default_factory=list)

    def merge(self, other: "StratificationReport") -> None:
        """Accumulate another cluster's report into this one."""
        self.n_strata += other.n_strata
        self.n_cuts += other.n_cuts
        self.n_gaps_found += other.n_gaps_found
        self.n_refused_trunk += other.n_refused_trunk
        self.n_refused_population += other.n_refused_population
        self.n_points_dropped += other.n_points_dropped
        self.gap_heights_m.extend(other.gap_heights_m)


def lateral_width(points: np.ndarray, quantile: float = 0.90) -> float:
    """
    Lateral extent of a point set as a diameter, in metres.

    Twice the given quantile of horizontal distance from the centroid. The
    quantile (rather than the maximum) keeps a handful of stray returns from
    inflating a genuinely narrow trunk into an apparent shrub, which would
    silently disable the trunk guard.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return 0.0
    centre = points[:, :2].mean(axis=0)
    radius = np.linalg.norm(points[:, :2] - centre, axis=1)
    return float(2.0 * np.quantile(radius, quantile))


def find_vertical_gaps(
    z: np.ndarray,
    settings: Optional[StratificationSettings] = None,
) -> List[Tuple[float, float]]:
    """
    Locate interior voids in a vertical point distribution.

    Returns a list of (z_bottom, z_top) intervals, ordered bottom-up, each
    a maximal run of sparse histogram bins at least `min_gap_m` tall.

    Only INTERIOR runs are returned. A sparse run touching the top or bottom
    of the range is a density tail, not a gap between two strata; cutting
    there would shave off noise rather than separate two plants.
    """
    settings = settings or StratificationSettings()
    z = np.asarray(z, dtype=float)
    if len(z) == 0:
        return []

    z_min, z_max = float(z.min()), float(z.max())
    span = z_max - z_min
    if span < settings.min_gap_m:
        return []

    n_bins = max(1, int(np.ceil(span / settings.bin_size_m)))
    counts, edges = np.histogram(z, bins=n_bins, range=(z_min, z_max))

    occupied = counts[counts > 0]
    if occupied.size == 0:
        return []
    threshold = max(
        float(settings.sparse_floor_points),
        settings.sparse_fraction * float(np.percentile(occupied, 90)),
    )
    sparse = counts <= threshold

    gaps: List[Tuple[float, float]] = []
    i = 0
    while i < len(sparse):
        if not sparse[i]:
            i += 1
            continue
        j = i
        while j < len(sparse) and sparse[j]:
            j += 1
        # Interior only: a run touching either end is a tail.
        if i > 0 and j < len(sparse):
            gap_bottom, gap_top = float(edges[i]), float(edges[j])
            if gap_top - gap_bottom >= settings.min_gap_m:
                gaps.append((gap_bottom, gap_top))
        i = j
    return gaps


def split_vertical_strata(
    points: np.ndarray,
    settings: Optional[StratificationSettings] = None,
    height_above_ground: Optional[np.ndarray] = None,
) -> Tuple[List[Stratum], StratificationReport]:
    """
    Split one cluster into vertically separate strata.

    Parameters
    ----------
    points : (N, 3) array
        The cluster's points.
    settings : StratificationSettings, optional
    height_above_ground : (N,) array, optional
        Per-point height above local ground, aligned with `points`. Used
        only to tag a stratum as low vegetation; stratification itself does
        not need it. Supply it from
        `ground_height_filter.height_above_ground` so the tag uses the same
        ground raster as the rest of the pipeline.

    Returns
    -------
    (strata, report)
        `strata` is ordered bottom-up and always holds at least one entry;
        an unsplittable cluster comes back as a single stratum, so callers
        can treat this as a total function.

    Cuts are evaluated bottom-up. A refused cut merges the two candidate
    segments back together -- including the gap's own points, which are only
    dropped when a cut is actually accepted.
    """
    settings = settings or StratificationSettings()
    points = np.asarray(points, dtype=float)
    report = StratificationReport()

    if len(points) == 0:
        report.n_strata = 0
        return [], report

    def finish(strata: List[Stratum]) -> Tuple[List[Stratum], StratificationReport]:
        if height_above_ground is not None:
            for stratum in strata:
                if stratum.top_height_above_ground_m is not None:
                    stratum.is_low_vegetation = (
                        stratum.top_height_above_ground_m
                        < settings.low_vegetation_max_height_m
                    )
        report.n_strata = len(strata)
        return strata, report

    def make(mask: np.ndarray) -> Stratum:
        chunk = points[mask]
        top_hag = None
        if height_above_ground is not None:
            hag = np.asarray(height_above_ground, dtype=float)[mask]
            if hag.size:
                top_hag = float(np.quantile(hag, 0.98))
        return Stratum(
            points=chunk,
            z_low=float(chunk[:, 2].min()),
            z_high=float(chunk[:, 2].max()),
            top_height_above_ground_m=top_hag,
        )

    all_mask = np.ones(len(points), dtype=bool)
    if not settings.enabled:
        return finish([make(all_mask)])

    z = points[:, 2]
    gaps = find_vertical_gaps(z, settings)
    report.n_gaps_found = len(gaps)
    if not gaps:
        return finish([make(all_mask)])

    strata: List[Stratum] = []
    # `active` accumulates the current stratum; it grows when a cut is refused.
    active = z <= gaps[0][0]

    for index, (gap_bottom, gap_top) in enumerate(gaps):
        next_bottom = gaps[index + 1][0] if index + 1 < len(gaps) else np.inf
        upper = (z >= gap_top) & (z <= next_bottom)
        in_gap = (z > gap_bottom) & (z < gap_top)

        n_lower, n_upper = int(active.sum()), int(upper.sum())
        cut = True
        if n_lower < settings.min_stratum_points or n_upper < settings.min_stratum_points:
            report.n_refused_population += 1
            cut = False
        elif lateral_width(points[active], settings.width_quantile) < settings.trunk_width_m:
            # The lower side is a trunk, not a separate plant. Keep the tree whole.
            report.n_refused_trunk += 1
            cut = False

        if cut:
            strata.append(make(active))
            report.n_cuts += 1
            report.n_points_dropped += int(in_gap.sum())
            report.gap_heights_m.append(float(gap_top - gap_bottom))
            active = upper
        else:
            active = active | in_gap | upper

    strata.append(make(active))
    return finish(strata)


def stratify_clusters(
    clusters: Sequence[np.ndarray],
    settings: Optional[StratificationSettings] = None,
    height_above_ground: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[List[Stratum], StratificationReport]:
    """
    Apply `split_vertical_strata` to a sequence of clusters.

    Returns the flattened strata plus one accumulated report. `height_above_ground`,
    if given, must be a matching sequence of per-cluster arrays.
    """
    settings = settings or StratificationSettings()
    out: List[Stratum] = []
    total = StratificationReport(n_strata=0)
    for index, cluster in enumerate(clusters):
        hag = height_above_ground[index] if height_above_ground is not None else None
        strata, report = split_vertical_strata(cluster, settings, hag)
        out.extend(strata)
        total.merge(report)
    return out, total
