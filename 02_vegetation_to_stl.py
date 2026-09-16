"""
02_vegetation_to_stl.py -- cluster vegetation points into individual trees and
fit each crown a closed, data-driven shape.

CROWN MODEL (--crown-model radial, default)
-------------------------------------------
Each crown becomes a stack of HEIGHT BANDS, every band carrying its own radius
per azimuth sector, lofted into a closed mesh. Both crown height and crown
width therefore come from the points; neither is implied by the other.

This replaces a flat-based hemisphere, whose defining problem was that a dome's
height EQUALS its radius. On the lisbon1 mesh that produced a height/width
ratio of exactly 1.00 for all 2441 crowns -- a property of the assumption, not
of the trees. Real crowns run from columnar to broadly spreading, and the flat
disc underneath shadowed a full circle at base level where a real crown tapers.

It matters because shading is Beer-Lambert on the path length THROUGH the crown
(``vegetation_transmission_from_intersections``), so both the silhouette and the
interior chord are physical. Forcing an axis ratio of 1 biased every chord.

WHY BANDS BY HEIGHT AND NOT DIRECTIONS FROM A CENTRE
----------------------------------------------------
Airborne LiDAR samples a crown from ABOVE: coverage in height is good, coverage
in solid angle about the crown centre is not. A star-shape parameterisation
therefore leaves most of the lower hemisphere unmeasured and lets the fill
prior set the vertical size -- measured directly, that approach recovered a
height/width of 0.66 for a true sphere and 1.10 for a true 3:1 column. Banding
by height makes both dimensions data-driven and independent by construction.

WHAT IS MEASURED AND WHAT IS MODELLED
-------------------------------------
The upper crown is data. The underside is not: LiDAR barely sees it, so the
bottom band is tapered toward the trunk by ``--crown-lower-taper``. That is an
explicit model of a rounded underside -- better than the hemisphere's flat disc,
but still a model. The run reports mean measured-cell coverage (about 60% on the
Lisbon cloud) so the split is visible rather than implied.

ROBUSTNESS
----------
Each cell takes a high QUANTILE of its points' horizontal distance, so stray
branch tips fall outside the crown instead of inflating a sector; cells with too
few points inherit their band's median; bands with no points are interpolated
between bands that have some, so a gap cannot pinch the mesh shut. Crowns too
sparse for the grid fall back to the legacy dome rather than being dropped --
a crudely shaped crown still casts roughly the right shade, a missing one casts
none.

Closure is a correctness requirement, not cosmetics: the transmission integral
pairs ray entries against exits, so an open crown reports a nonsense chord
rather than an obvious defect. The loft is closed for any positive radius field,
and the builder additionally rejects inward winding, which is watertight but
encloses negative volume and reads as valid to a casual check.

Cost is unchanged: the default 8x16 grid is 256 faces per crown, exactly what
the hemisphere used. ``--crown-model hemisphere`` restores the old behaviour.

Pipeline:
  0. Height-above-ground filter (ground_height_filter.py): drops points too
     close to the local ground surface BEFORE clustering. Necessary because
     our vegetation classification merges low/medium/high vegetation into
     one bucket, so ground-hugging grass/shrub points sharing a tree's XY
     footprint would otherwise get pulled into the same cluster as the
     canopy above.
  1. Grid-based connected components (grid_cluster.py) -- fast, memory-safe
     macro clustering.
  2. Canopy-height-model watershed segmentation (tree_crown_segmentation.py)
     -- splits macro-clusters large enough to plausibly be multiple trees
     into individual crowns.
  3. Vertical stratification (vertical_stratification.py) -- splits a column
     into separate plants. Steps 1 and 2 are both plan-view: grid_cluster_2d
     takes XY only, and the watershed runs on a max-Z canopy height model,
     so between them they can separate trees standing SIDE BY SIDE but never
     a hedge from the tree above it. Step 0's height filter removes only the
     ground-hugging part of that problem; anything taller than its cut stays
     merged. Cuts at interior voids, guarded so a trunk is not mistaken for
     a separate plant.

Run:
    python3 02_vegetation_to_stl.py \
        --input split/vegetation_points.npy \
        --output vegetation.stl \
        --ground-npy split/ground_and_water_points.npy \
        --min-height-above-ground 1.5 \
        --crown-model radial --crown-lat-bands 8 --crown-lon-sectors 16
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from grid_cluster import grid_cluster_2d
from tree_crown_segmentation import segment_tree_crowns, should_segment
import vertical_stratification as vstrat
import crown_field_reconstruction as cfr
from ground_height_filter import build_ground_height_grid, height_above_ground
import crown_reconstruction as cr


def fit_hemisphere(points, horiz_quantile=0.9, base_z_quantile=0.15,
                   min_radius=1.0, max_radius=10.0):
    """Fit a flat-side-down hemisphere (dome) to one tree crown's points.

    A crown is modeled as a dome: a horizontal circular flat base with a
    half-sphere bulging upward from it. The fit is deliberately robust to
    stray LiDAR returns (single leaves, thin protruding branches, low
    hanging shoots) -- it represents the CORE crown mass rather than the
    convex extent of every point:

      * Center (cx, cy): per-axis median of the horizontal positions --
        resistant to a few outlying points pulling the crown sideways.
      * Radius R: the `horiz_quantile` (default 90th percentile) of each
        point's horizontal distance from the center. The outermost ~10% of
        points -- stray branch tips -- fall OUTSIDE the dome instead of
        inflating it. R is clamped to [min_radius, max_radius], where
        max_radius is a realistic upper bound on crown radius so a noisy or
        merged cluster can't produce an absurdly large dome.
      * Base elevation (base_z): the `base_z_quantile` (default 15th
        percentile) of the points' z. Using a low percentile rather than
        the minimum ignores a few low hanging points/branches, and because
        near-ground vegetation was already dropped by the height-above-
        ground filter upstream, this base sits well above the pedestrian
        zone -- the dome rarely occupies the lowest ~2 m and never clips
        below ground.

    The dome's flat side is at z = base_z and it rises to base_z + R, so its
    height equals its horizontal radius. Returns (cx, cy, base_z, R).
    """
    cx, cy = np.median(points[:, 0]), np.median(points[:, 1])
    horiz_dist = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    radius = float(np.quantile(horiz_dist, horiz_quantile))
    radius = min(max(radius, min_radius), max_radius)
    base_z = float(np.quantile(points[:, 2], base_z_quantile))
    return float(cx), float(cy), base_z, radius


def build_hemisphere_mesh(cx, cy, base_z, radius, subdivisions=2):
    """Watertight flat-side-down hemisphere mesh at (cx, cy, base_z), dome up.

    Built directly as a UV-parametrized dome (top pole -> equator) plus a flat
    circular base cap, so there's no dependency on shapely/slice_plane. The
    flat base lies at z = base_z and the dome rises to z = base_z + radius.
    `subdivisions` controls tessellation density (n_lon = 4 * 2**subdivisions).
    """
    n_lon = max(4, 4 * (2 ** int(subdivisions)))     # sectors around the axis
    n_lat = max(2, n_lon // 2)                        # rings pole -> equator

    verts = [[0.0, 0.0, radius]]                      # 0: top pole
    for i in range(1, n_lat + 1):                     # rings; i=n_lat is equator
        lat = (np.pi / 2.0) * (i / n_lat)
        z = radius * np.cos(lat)
        rr = radius * np.sin(lat)
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            verts.append([rr * np.cos(phi), rr * np.sin(phi), z])
    base_center = len(verts)
    verts.append([0.0, 0.0, 0.0])                     # base center (flat side)

    def ring(i, j):                                   # vertex index in ring i (1..n_lat)
        return 1 + (i - 1) * n_lon + (j % n_lon)

    faces = []
    for j in range(n_lon):                            # top cap fan (pole -> ring 1)
        faces.append([0, ring(1, j), ring(1, j + 1)])
    for i in range(1, n_lat):                         # dome bands
        for j in range(n_lon):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            faces.append([a, c, b])
            faces.append([b, c, d])
    for j in range(n_lon):                            # flat base cap (downward)
        faces.append([base_center, ring(n_lat, j + 1), ring(n_lat, j)])

    dome = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64),
                           process=True)
    dome.apply_translation([cx, cy, base_z])
    return dome


def crown_radial_profile(points, n_lat=8, n_lon=16, quantile=0.80,
                         min_bin_points=3, smoothing_passes=2,
                         min_radius=1.0, max_radius=10.0,
                         lower_taper=0.65, crown_base_fraction=0.45,
                         max_depth_ratio=2.0):
    """Fit a crown as a stack of height bands, each with its own radii.

    WHY NOT A HEMISPHERE
    --------------------
    A flat-based dome forces crown HEIGHT to equal crown RADIUS -- on the
    lisbon1 mesh every one of 2441 crowns had a height/radius ratio of exactly
    1.00, because the shape says so, not because the trees do. Real crowns run
    from columnar to broadly spreading, and the flat disc underneath shadows a
    full circle at base level where a real crown tapers.

    Shading here is Beer-Lambert on the path length through the crown, so BOTH
    the silhouette and the interior chord matter. Forcing an axis ratio of 1
    biases every chord.

    WHY HEIGHT BANDS RATHER THAN A STAR SHAPE ABOUT A POINT
    -------------------------------------------------------
    Airborne LiDAR samples a crown from ABOVE: coverage in height is good,
    coverage in solid angle about the crown centre is not. Parameterising by
    direction therefore leaves most of the lower hemisphere unmeasured and lets
    a modelled prior dominate the fitted shape -- tested directly, a star-shape
    fit recovered a height/width of 0.66 for a true sphere and 1.10 for a true
    3:1 column, because the prior, not the data, was setting the vertical size.

    Banding by HEIGHT instead makes both dimensions data-driven: the vertical
    extent comes from the observed z range, and each band's radii from the
    points actually in it. The two are then independent by construction.

    THE FIT
    -------
    ``n_lat`` height bands between a low and a high z quantile; ``n_lon``
    azimuth sectors per band; each cell takes a high quantile of its points'
    horizontal distance from the crown axis -- high enough to follow the crown,
    below 1.0 so one stray branch tip cannot inflate a whole sector. Cells with
    too few points inherit their band's median, and bands with no data at all
    are interpolated between the bands that have some.

    The result is lofted into a closed mesh, watertight BY CONSTRUCTION. That
    matters more than it sounds: the transmission integral pairs ray entries
    against exits, so a crown with a hole reports a nonsense path length rather
    than merely looking wrong.

    ``lower_taper`` closes the bottom band toward the trunk rather than leaving
    the flat disc the hemisphere had. Returns
    ``(axis_xy, z_levels, radius_grid, coverage_fraction)``.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 4:
        raise ValueError("crown needs at least 4 points for a radial fit")

    cx = float(np.median(points[:, 0]))
    cy = float(np.median(points[:, 1]))
    z_bottom = float(np.quantile(points[:, 2], 0.02))
    z_top = float(np.quantile(points[:, 2], 0.98))
    if not z_top > z_bottom:
        raise ValueError("crown has no vertical extent")

    horizontal = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    azimuth = np.mod(np.arctan2(points[:, 1] - cy, points[:, 0] - cx),
                     2.0 * np.pi)
    height_fraction = (points[:, 2] - z_bottom) / (z_top - z_bottom)
    band = np.clip((height_fraction * n_lat).astype(int), 0, n_lat - 1)
    sector = np.clip((azimuth / (2.0 * np.pi) * n_lon).astype(int),
                     0, n_lon - 1)

    radius = np.full((n_lat, n_lon), np.nan)
    flat = band * n_lon + sector
    order = np.argsort(flat)
    flat_sorted, horizontal_sorted = flat[order], horizontal[order]
    boundaries = np.flatnonzero(np.diff(flat_sorted)) + 1
    for start, stop in zip(np.concatenate([[0], boundaries]),
                           np.concatenate([boundaries, [len(flat_sorted)]])):
        if stop - start < min_bin_points:
            continue
        cell = int(flat_sorted[start])
        radius[cell // n_lon, cell % n_lon] = np.quantile(
            horizontal_sorted[start:stop], quantile)

    covered = np.isfinite(radius)
    coverage = float(covered.mean())
    if not covered.any():
        raise ValueError("no crown cell held enough points to fit a radius")

    # ------------------------------------------------------------------
    # SEPARATE CROWN FROM TRUNK -- BEFORE ANY FILLING
    #
    # A vegetation CLUSTER is not a crown. LiDAR returns include trunk hits and
    # understory, so a cluster spans from the height filter to the treetop; on
    # the Lisbon cloud that gave 18.6 m median "crowns", i.e. canopy-width
    # geometry wrapped around a trunk.
    #
    # The data separates them: the crown is where horizontal spread is large,
    # the trunk where it collapses. But the test must run on the MEASURED band
    # widths -- filling sparse bands first destroys exactly the signal it needs,
    # because an empty trunk band then inherits a canopy radius by
    # interpolation and passes the test. That ordering bug left the trim inert.
    #
    # The hemisphere never met this because it set height = radius and ignored
    # the cluster's true vertical extent.
    # ------------------------------------------------------------------
    measured_band = np.array([np.nanmedian(radius[i]) if covered[i].any() else np.nan
                              for i in range(n_lat)])
    if not np.isfinite(measured_band).any():
        raise ValueError("no height band held enough points to fit a radius")
    widest = float(np.nanmax(measured_band))
    is_crown = np.isfinite(measured_band) & (measured_band
                                             >= crown_base_fraction * widest)
    first = int(np.flatnonzero(is_crown)[0])
    first = min(first, n_lat - 2)          # keep at least two bands
    radius = radius[first:]
    covered = covered[first:]
    z_full = z_bottom + (z_top - z_bottom) * (np.arange(n_lat) + 0.5) / n_lat
    z_levels = z_full[first:]
    n_bands = radius.shape[0]

    # Now fill: sparse cells take their own band's median, and bands with no
    # data at all are interpolated between bands that have some, so a gap
    # cannot pinch the mesh shut.
    band_radius = np.array([np.nanmedian(radius[i]) if covered[i].any() else np.nan
                            for i in range(n_bands)])
    known = np.isfinite(band_radius)
    band_radius = np.interp(np.arange(n_bands), np.flatnonzero(known),
                            band_radius[known])
    for i in range(n_bands):
        row = radius[i]
        row[~np.isfinite(row)] = band_radius[i]

    for _ in range(max(0, int(smoothing_passes))):
        padded = np.pad(radius, ((1, 1), (0, 0)), mode="edge")
        padded = np.concatenate([padded[:, -1:], padded, padded[:, :1]], axis=1)
        radius = (padded[1:-1, 1:-1] * 4.0
                  + padded[:-2, 1:-1] + padded[2:, 1:-1]
                  + padded[1:-1, :-2] + padded[1:-1, 2:]) / 8.0

    radius = np.clip(radius, min_radius, max_radius).copy()

    # ------------------------------------------------------------------
    # BOTANICAL DEPTH BOUND
    #
    # On the real cloud this fit produced a median crown DEPTH of ~18 m at ~7 m
    # width -- slender columns, not crowns. It is not terrain: ground relief
    # under those footprints is 0.9 m median, and depth correlates with relief
    # at only r = 0.29. The clusters really do span that height over flat
    # ground, which points upstream at the clustering/segmentation rather than
    # at this fit.
    #
    # The hemisphere never showed this because it set depth = radius by
    # construction, so a vertically over-extended cluster silently became a
    # normal-looking dome. Exposing the problem is progress; shipping slender
    # columns is not.
    #
    # Until segmentation is settled, crown depth is capped at
    # ``max_depth_ratio`` times the crown width. This is a far weaker assumption
    # than the hemisphere's depth = radius, and it is applied by trimming the
    # LOWEST bands -- keeping the measured top of the canopy, which is the part
    # airborne LiDAR actually resolves. Set it to a large value to disable.
    # ------------------------------------------------------------------
    if max_depth_ratio and max_depth_ratio > 0 and len(radius) > 2:
        spacing = (z_top - z_bottom) / n_lat
        width = 2.0 * float(np.median(radius.mean(axis=1)))
        allowed_bands = int(np.ceil(max_depth_ratio * width / max(spacing, 1e-6)))
        if 2 <= allowed_bands < len(radius):
            radius = radius[len(radius) - allowed_bands:]
            z_levels = z_levels[len(z_levels) - allowed_bands:]

    # Close the underside toward the trunk instead of a flat disc.
    radius = radius.copy()
    radius[0] = radius[0] * float(lower_taper)
    return np.array([cx, cy]), z_levels, radius, coverage


def build_crown_mesh(axis_xy, z_levels, radius, z_bottom=None, z_top=None,
                     ground_clearance_z=None):
    """Watertight lofted crown from stacked height bands.

    Topology is a closed loft: a nadir vertex, ``n_lat`` rings, an apex vertex.
    Closed by construction for ANY positive radius field, which the Beer-Lambert
    path length requires -- it pairs ray entries with exits, so an open crown
    yields a garbage chord rather than an obvious defect.
    """
    radius = np.asarray(radius, dtype=float)
    z_levels = np.asarray(z_levels, dtype=float)
    n_lat, n_lon = radius.shape
    spacing = (z_levels[-1] - z_levels[0]) / max(n_lat - 1, 1)
    if z_bottom is None:
        z_bottom = z_levels[0] - 0.5 * spacing
    if z_top is None:
        z_top = z_levels[-1] + 0.5 * spacing

    vertices = [[0.0, 0.0, float(z_bottom)]]
    for i in range(n_lat):
        for j in range(n_lon):
            azimuth = 2.0 * np.pi * j / n_lon
            r = float(radius[i, j])
            vertices.append([r * np.cos(azimuth), r * np.sin(azimuth),
                             float(z_levels[i])])
    vertices.append([0.0, 0.0, float(z_top)])
    apex = len(vertices) - 1

    def ring(i, j):
        return 1 + i * n_lon + (j % n_lon)

    # Winding: side bands are ordered so the normal is phi_hat x z_hat = +r_hat,
    # i.e. outward. Getting this backwards produces a mesh that is watertight
    # but encloses a NEGATIVE volume, which reads as valid to a casual check.
    faces = []
    for j in range(n_lon):                                  # bottom fan (-z out)
        faces.append([0, ring(0, j + 1), ring(0, j)])
    for i in range(n_lat - 1):                              # side bands
        for j in range(n_lon):
            a, b = ring(i, j), ring(i, j + 1)
            c, d = ring(i + 1, j), ring(i + 1, j + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])
    for j in range(n_lon):                                  # top fan (+z out)
        faces.append([apex, ring(n_lat - 1, j), ring(n_lat - 1, j + 1)])

    crown = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64),
                            faces=np.asarray(faces, dtype=np.int64),
                            process=True)
    # Guard rather than trust the reasoning above: a closed mesh wound inward
    # still reports is_watertight, so the sign of the enclosed volume is the
    # check that actually catches it.
    if crown.is_watertight and crown.volume < 0:
        crown.invert()
    crown.apply_translation([float(axis_xy[0]), float(axis_xy[1]), 0.0])
    if ground_clearance_z is not None:
        lowest = float(crown.vertices[:, 2].min())
        if lowest < ground_clearance_z:
            crown.apply_translation([0.0, 0.0, ground_clearance_z - lowest])
    return crown


def parse_args():
    p = argparse.ArgumentParser(description="Vegetation points -> per-crown hemisphere (dome) STL")
    p.add_argument("--input", required=True, help="Path to vegetation_points.npy (Nx3 array)")
    p.add_argument("--output", required=True, help="Output STL path")

    p.add_argument("--ground-npy", default=None,
                    help="Path to ground_and_water_points.npy. If given, points too close "
                         "to the local ground surface are dropped before clustering (fixes "
                         "canopy-to-ground pillar artifacts from merged low/high vegetation "
                         "classes). Strongly recommended -- omit only if you don't have a "
                         "ground reference available.")
    p.add_argument("--min-height-above-ground", type=float, default=1.5,
                    help="Drop vegetation points within this height (meters) of the local "
                         "ground surface (default: 1.5)")
    p.add_argument("--ground-cell-size", type=float, default=2.0,
                    help="Ground height lookup grid resolution, meters -- coarser than the "
                         "full ground DEM is fine here (default: 2.0)")

    p.add_argument("--cell-size", type=float, default=0.5,
                    help="Macro clustering grid cell size, meters (default: 0.5)")
    p.add_argument("--connect-radius", type=int, default=1,
                    help="Macro clustering dilation radius, grid cells (default: 1)")
    p.add_argument("--min-hull-points", type=int, default=10,
                    help="Skip final tree clusters with fewer than this many points "
                         "(default: 10)")
    p.add_argument("--crown-horiz-quantile", type=float, default=0.9,
                    help="Crown dome radius = this quantile of points' horizontal distance "
                         "from the crown center. The outermost (1 - q) fraction of points -- "
                         "stray branch tips/leaves -- fall OUTSIDE the dome instead of "
                         "inflating it. Lower = tighter, more outlier-resistant (default: 0.9)")
    p.add_argument("--crown-base-z-quantile", type=float, default=0.15,
                    help="Crown dome flat base sits at this quantile of the crown points' "
                         "height. A low (but non-zero) percentile ignores a few low hanging "
                         "points while keeping the base above the pedestrian zone and above "
                         "ground (default: 0.15)")
    p.add_argument("--max-crown-radius", type=float, default=10.0,
                    help="Realistic upper bound on crown dome radius, meters (~20 m diameter). "
                         "Caps domes from noisy or under-segmented clusters. Also bounds dome "
                         "HEIGHT, since a hemisphere's height equals its radius (default: 10.0)")
    p.add_argument("--min-crown-radius", type=float, default=1.0,
                    help="Floor on crown dome radius, meters -- avoids degenerate slivers for "
                         "very tight clusters (default: 1.0)")
    p.add_argument("--crown-model",
                   choices=["field", "reconstruct", "radial", "hemisphere"],
                   default="field",
                   help="Crown shape. 'field' (default) is the only model with "
                        "NO shape prior: an implicit union-of-balls field around "
                        "the measured points, surfaced by marching cubes, so the "
                        "crown may be forked, leaning or re-entrant in plan. "
                        "'radial' fits a star-shaped R(theta,phi) about a vertical "
                        "axis -- height and width are independent but every plan "
                        "outline is a convex polygon, which reads as a field of "
                        "circles from above. 'hemisphere' forces crown height to "
                        "EQUAL crown radius. 'reconstruct' is the alpha-shape "
                        "attempt; it does not close on real LiDAR (see "
                        "crown_field_reconstruction.py) and is kept for reference.")
    p.add_argument("--crown-bridge-radius", type=float,
                   default=cfr.FieldSettings.bridge_radius_m,
                   help="field model: union-of-balls radius, metres -- the canopy "
                        "gap scale being bridged. Smaller hugs the points but "
                        f"fragments the crown (default: {cfr.FieldSettings.bridge_radius_m})")
    p.add_argument("--crown-face-budget", type=int,
                   default=cfr.FieldSettings.face_budget,
                   help="field model: target faces per crown after decimation "
                        f"(default: {cfr.FieldSettings.face_budget})")
    p.add_argument("--crown-smooth-iterations", type=int,
                   default=cfr.FieldSettings.smooth_iterations,
                   help="field model: Taubin smoothing passes removing voxel "
                        f"stair-stepping (default: {cfr.FieldSettings.smooth_iterations})")
    p.add_argument("--crown-lat-bands", type=int, default=8,
                   help="Colatitude bands of the crown mesh (default 8). With "
                        "16 sectors this gives 256 faces per crown -- the same "
                        "budget the hemisphere used, spent on shape instead of "
                        "smoothness.")
    p.add_argument("--crown-lon-sectors", type=int, default=16,
                   help="Azimuth sectors of the crown mesh (default 16)")
    p.add_argument("--crown-radial-quantile", type=float, default=0.80,
                   help="Per-bin quantile of point distance used as that "
                        "direction's radius (default 0.80). Below 1.0 so one "
                        "stray branch tip cannot inflate a whole sector.")
    p.add_argument("--crown-min-bin-points", type=int, default=3,
                   help="A bin needs this many points to be treated as "
                        "measured; otherwise it is filled from the prior and "
                        "counted as modelled (default 3)")
    p.add_argument("--crown-smoothing-passes", type=int, default=2,
                   help="Sphere-wrapped smoothing passes over the radial grid "
                        "(default 2). Removes bin-to-bin noise that would "
                        "otherwise show as facet spikes.")
    p.add_argument("--crown-base-fraction", type=float, default=0.45,
                   help="A height band belongs to the CROWN once its mean "
                        "radius reaches this fraction of the widest band "
                        "(default 0.45). Bands below that are trunk or "
                        "understory and are dropped -- a vegetation cluster's "
                        "vertical extent is not the crown's, and without this "
                        "the fit wraps canopy-width geometry around the trunk.")
    p.add_argument("--crown-max-depth-ratio", type=float, default=2.0,
                   help="Cap crown DEPTH at this multiple of crown width "
                        "(default 2.0), by trimming the lowest bands and "
                        "keeping the measured canopy top. A botanical bound, "
                        "far weaker than the hemisphere's implicit depth = "
                        "radius, and needed because the clustering currently "
                        "yields clusters ~18 m deep at ~7 m wide over flat "
                        "ground. Set large to disable.")
    p.add_argument("--crown-lower-taper", type=float, default=0.65,
                   help="Radius at the crown base as a fraction of the "
                        "equatorial radius (default 0.65). Airborne LiDAR sees "
                        "almost nothing of the underside, so this is an "
                        "explicit MODEL of it -- a rounded taper rather than "
                        "the flat disc the hemisphere used.")
    p.add_argument("--crown-subdivisions", type=int, default=2,
                    help="Icosphere subdivision level for each crown dome -- higher is "
                         "smoother but more faces (default: 2)")

    p.add_argument("--crown-chm-res", type=float, default=0.25,
                    help="Canopy height model resolution for crown segmentation, meters "
                         "(default: 0.25)")
    p.add_argument("--crown-smooth-sigma", type=float, default=2.0,
                    help="CHM smoothing sigma, grid cells -- higher suppresses more of the "
                         "natural bumpiness within a single canopy/hedge row that would "
                         "otherwise be mistaken for separate treetops (default: 2.0)")
    p.add_argument("--crown-min-tree-distance", type=float, default=3.0,
                    help="Minimum distance between detected treetops, meters -- should "
                         "roughly match your site's minimum realistic trunk spacing (most "
                         "campus/urban trees: 4m+). Too low causes 'field of pebbles' "
                         "over-segmentation of natural canopy texture into many tiny "
                         "spurious fragments; too high risks merging genuinely distinct "
                         "nearby trees. (default: 3.0)")
    p.add_argument("--crown-min-fragment-points", type=int, default=50,
                    help="Structural backstop: any detected crown with fewer than this many "
                         "points gets merged into its nearest larger neighbor rather than "
                         "kept as an isolated sliver hull. Set to 0 to disable (default: 50)")
    p.add_argument("--crown-min-points", type=int, default=300,
                    help="Only attempt crown segmentation on macro-clusters with at least "
                         "this many points (default: 300)")
    p.add_argument("--crown-min-extent", type=float, default=8.0,
                    help="Only attempt crown segmentation on macro-clusters whose XY extent "
                         "exceeds this, meters -- a real single tree crown rarely exceeds "
                         "this diameter, so bigger almost certainly means multiple trees "
                         "(default: 8.0)")

    # -- Alpha-shape reconstruction (--crown-model reconstruct) -------------
    # Defaults mirror crown_reconstruction.ReconstructionSettings.
    p.add_argument("--crown-voxel-size", type=float,
                    default=cr.ReconstructionSettings.voxel_size,
                    help="Target reconstruction resolution, meters. Finer than this "
                         "over-refines without adding real canopy detail "
                         f"(default: {cr.ReconstructionSettings.voxel_size})")
    p.add_argument("--crown-outlier-std-ratio", type=float,
                    default=cr.ReconstructionSettings.outlier_std_ratio,
                    help="Statistical outlier rejection: drop points whose mean "
                         "k-NN distance exceeds this many standard deviations "
                         f"(default: {cr.ReconstructionSettings.outlier_std_ratio})")
    p.add_argument("--crown-coverage-target", type=float,
                    default=cr.ReconstructionSettings.coverage_target,
                    help="Alpha is grown until the solid contains at least this "
                         f"fraction of the crown's points (default: {cr.ReconstructionSettings.coverage_target})")
    p.add_argument("--crown-max-faces", type=int,
                    default=cr.ReconstructionSettings.max_faces,
                    help="Decimate each reconstructed crown to at most this many "
                         f"faces (default: {cr.ReconstructionSettings.max_faces})")

    # -- Vertical stratification -------------------------------------------
    # Macro-clustering is XY-only and the crown watershed runs on a max-Z
    # canopy height model, so neither can separate a hedge from the tree
    # standing over it. See vertical_stratification.py for the measurements.
    p.add_argument("--vertical-stratification", choices=["on", "off"], default="on",
                    help="Split vertically merged vegetation (a shrub or hedge sharing "
                         "a tree's footprint) into separate objects before crown "
                         "fitting. 'off' restores the pre-stratification geometry "
                         "exactly (default: on)")
    p.add_argument("--strat-bin-size", type=float, default=0.5,
                    help="Height of one z-histogram bin used for void detection, "
                         "meters (default: 0.5)")
    p.add_argument("--strat-min-gap", type=float, default=2.0,
                    help="Minimum height of an interior void before it is treated as "
                         "a separation between two plants, meters. Measured voids on "
                         "the lisbon cloud run 4-9 m (default: 2.0)")
    p.add_argument("--strat-trunk-width", type=float, default=1.5,
                    help="TRUNK GUARD: if the lower side of a candidate cut is "
                         "narrower than this diameter it is a trunk, not a separate "
                         "plant, and the cut is refused so the tree stays whole. "
                         "Measured low strata are all wider than 3.6 m (default: 1.5)")
    p.add_argument("--strat-min-points", type=int, default=20,
                    help="Both sides of a cut must carry at least this many points "
                         "(default: 20)")
    p.add_argument("--low-vegetation-max-height", type=float, default=3.0,
                    help="A stratum whose top is below this height above ground is "
                         "tagged low vegetation (default: 3.0)")
    p.add_argument("--drop-low-vegetation", action="store_true",
                    help="Discard strata tagged as low vegetation instead of meshing "
                         "them. Off by default: a hedge is real geometry that really "
                         "shades a pedestrian, and dropping it loses that shade.")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.load(in_path)
    print(f"[veg] Loaded {len(pts):,} vegetation points")

    if len(pts) == 0:
        print("[veg] No vegetation points -- writing an empty placeholder STL.")
        trimesh.Trimesh().export(str(out_path))
        print(f"[veg_result] n_trees=0 total_faces=0 output={out_path}")
        return

    # Retained so vertical stratification can tag low vegetation against the
    # SAME ground raster the height filter used, rather than a second estimate.
    hag = None

    if args.ground_npy is not None:
        ground_pts = np.load(args.ground_npy)
        print(f"[veg] Applying height-above-ground filter "
              f"(min_height={args.min_height_above_ground} m) ...")
        ground_grid = build_ground_height_grid(ground_pts,
                                               cell_size=args.ground_cell_size)
        hag = height_above_ground(pts, ground_grid)
        keep = hag >= args.min_height_above_ground
        n_dropped = int((~keep).sum())
        pts, hag = pts[keep], hag[keep]
        print(f"[veg] Dropped {n_dropped:,} near-ground points, {len(pts):,} remain")
        if len(pts) == 0:
            print("[veg] No points remain after ground filtering -- writing empty STL.")
            trimesh.Trimesh().export(str(out_path))
            print(f"[veg_result] n_trees=0 total_faces=0 output={out_path}")
            return
    else:
        print("[veg] WARNING: no --ground-npy given -- skipping height-above-ground "
              "filtering. Canopy-to-ground pillar artifacts are possible if low "
              "vegetation points share a tree's footprint.")

    xy = pts[:, :2]
    print(f"[veg] Macro-clustering with grid-based connected components "
          f"(cell_size={args.cell_size}, connect_radius={args.connect_radius}) ...")
    macro_labels, n_macro = grid_cluster_2d(xy, cell_size=args.cell_size,
                                             connect_radius_cells=args.connect_radius)
    unique_macro = np.unique(macro_labels)
    print(f"[veg] Found {len(unique_macro)} macro-clusters")

    tree_meshes = []
    n_trees_kept = 0
    coverage_total = 0.0
    coverage_count = 0
    n_fallback = 0
    alpha_total = 0.0
    face_total = 0
    n_trees_skipped = 0
    n_macro_segmented = 0
    n_macro_passthrough = 0
    n_low_vegetation = 0
    n_low_dropped = 0
    n_not_watertight = 0
    n_repaired = 0

    strat_settings = vstrat.StratificationSettings(
        bin_size_m=args.strat_bin_size,
        min_gap_m=args.strat_min_gap,
        min_stratum_points=args.strat_min_points,
        trunk_width_m=args.strat_trunk_width,
        low_vegetation_max_height_m=args.low_vegetation_max_height,
        enabled=(args.vertical_stratification == "on"),
    )
    strat_report = vstrat.StratificationReport(n_strata=0)
    if strat_settings.enabled:
        print(f"[veg] Vertical stratification ON (min gap {strat_settings.min_gap_m} m, "
              f"trunk guard {strat_settings.trunk_width_m} m)")
    else:
        print("[veg] Vertical stratification OFF -- vertically merged plants "
              "will be meshed as one solid column.")

    for macro_id in unique_macro:
        macro_mask = macro_labels == macro_id
        cluster_pts = pts[macro_mask]
        cluster_hag = hag[macro_mask] if hag is not None else None

        if should_segment(cluster_pts, args.cell_size,
                           args.crown_min_points, args.crown_min_extent):
            crown_labels, n_crowns = segment_tree_crowns(
                cluster_pts,
                chm_res=args.crown_chm_res,
                smooth_sigma=args.crown_smooth_sigma,
                min_tree_distance_m=args.crown_min_tree_distance,
                min_fragment_points=args.crown_min_fragment_points if args.crown_min_fragment_points > 0 else None,
            )
            n_macro_segmented += 1
            sub_masks = [crown_labels == c for c in np.unique(crown_labels)]
        else:
            n_macro_passthrough += 1
            sub_masks = [np.ones(len(cluster_pts), dtype=bool)]

        # The watershed above splits trees standing SIDE BY SIDE. It cannot
        # split a column into strata -- its canopy height model is max-Z per
        # cell, so anything under a canopy is invisible to it. Do that here.
        sub_clusters = []
        for sub_mask in sub_masks:
            sub_pts = cluster_pts[sub_mask]
            sub_hag = cluster_hag[sub_mask] if cluster_hag is not None else None
            strata, report = vstrat.split_vertical_strata(
                sub_pts, strat_settings, sub_hag)
            strat_report.merge(report)
            for stratum in strata:
                if stratum.is_low_vegetation:
                    n_low_vegetation += 1
                    if args.drop_low_vegetation:
                        n_low_dropped += 1
                        continue
                sub_clusters.append(stratum.points)

        for tree_pts in sub_clusters:
            if len(tree_pts) < args.min_hull_points:
                n_trees_skipped += 1
                continue
            try:
                if args.crown_model == "field":
                    fsettings = cfr.FieldSettings(
                        target_resolution_m=args.crown_voxel_size,
                        bridge_radius_m=args.crown_bridge_radius,
                        face_budget=args.crown_face_budget,
                        smooth_iterations=args.crown_smooth_iterations)
                    try:
                        dome, freport = cfr.reconstruct_crown(tree_pts, fsettings)
                        coverage_total += freport.containment
                        coverage_count += 1
                        face_total += freport.n_faces
                        if not freport.watertight:
                            n_not_watertight += 1
                        if freport.repaired:
                            n_repaired += 1
                    except cfr.CrownFieldError:
                        # Too sparse or degenerate for a field; it still has to
                        # cast shade, so fall back rather than drop it.
                        cx, cy, base_z, radius = fit_hemisphere(
                            tree_pts,
                            horiz_quantile=args.crown_horiz_quantile,
                            base_z_quantile=args.crown_base_z_quantile,
                            min_radius=args.min_crown_radius,
                            max_radius=args.max_crown_radius)
                        dome = build_hemisphere_mesh(
                            cx, cy, base_z, radius,
                            subdivisions=args.crown_subdivisions)
                        n_fallback += 1
                elif args.crown_model == "reconstruct":
                    settings = cr.ReconstructionSettings(
                        voxel_size=args.crown_voxel_size,
                        outlier_std_ratio=args.crown_outlier_std_ratio,
                        coverage_target=args.crown_coverage_target,
                        max_faces=args.crown_max_faces)
                    try:
                        dome, report = cr.reconstruct_crown(tree_pts, settings)
                        coverage_total += report.coverage
                        coverage_count += 1
                        alpha_total += report.alpha_m
                        face_total += report.n_faces
                    except cr.CrownReconstructionError:
                        # A cluster too sparse or degenerate for reconstruction
                        # still needs to cast shade; fall back rather than drop.
                        cx, cy, base_z, radius = fit_hemisphere(
                            tree_pts,
                            horiz_quantile=args.crown_horiz_quantile,
                            base_z_quantile=args.crown_base_z_quantile,
                            min_radius=args.min_crown_radius,
                            max_radius=args.max_crown_radius)
                        dome = build_hemisphere_mesh(
                            cx, cy, base_z, radius,
                            subdivisions=args.crown_subdivisions)
                        n_fallback += 1
                elif args.crown_model == "radial":
                    base_z = float(np.quantile(tree_pts[:, 2],
                                               args.crown_base_z_quantile))
                    try:
                        axis_xy, z_levels, radial, coverage = crown_radial_profile(
                            tree_pts,
                            n_lat=args.crown_lat_bands,
                            n_lon=args.crown_lon_sectors,
                            quantile=args.crown_radial_quantile,
                            min_bin_points=args.crown_min_bin_points,
                            smoothing_passes=args.crown_smoothing_passes,
                            min_radius=args.min_crown_radius,
                            max_radius=args.max_crown_radius,
                            lower_taper=args.crown_lower_taper,
                            crown_base_fraction=args.crown_base_fraction,
                            max_depth_ratio=args.crown_max_depth_ratio,
                        )
                        dome = build_crown_mesh(axis_xy, z_levels, radial,
                                                ground_clearance_z=base_z)
                        coverage_total += coverage
                        coverage_count += 1
                    except ValueError:
                        # Too few points to resolve any cell of the grid. Fall
                        # back to the dome rather than dropping the tree: a
                        # crudely shaped crown still casts roughly the right
                        # shade, whereas a missing one casts none at all.
                        cx, cy, base_z, radius = fit_hemisphere(
                            tree_pts,
                            horiz_quantile=args.crown_horiz_quantile,
                            base_z_quantile=args.crown_base_z_quantile,
                            min_radius=args.min_crown_radius,
                            max_radius=args.max_crown_radius,
                        )
                        dome = build_hemisphere_mesh(
                            cx, cy, base_z, radius,
                            subdivisions=args.crown_subdivisions)
                        n_fallback += 1
                else:
                    cx, cy, base_z, radius = fit_hemisphere(
                        tree_pts,
                        horiz_quantile=args.crown_horiz_quantile,
                        base_z_quantile=args.crown_base_z_quantile,
                        min_radius=args.min_crown_radius,
                        max_radius=args.max_crown_radius,
                    )
                    dome = build_hemisphere_mesh(
                        cx, cy, base_z, radius,
                        subdivisions=args.crown_subdivisions,
                    )
            except Exception as e:
                print(f"[veg] WARNING: {args.crown_model} crown fit failed for a "
                      f"cluster of {len(tree_pts)} points: {e}")
                n_trees_skipped += 1
                continue
            if dome is None or len(dome.faces) == 0:
                n_trees_skipped += 1
                continue
            tree_meshes.append(dome)
            n_trees_kept += 1

    if args.crown_model == "field" and n_fallback:
        print(f"[veg] WARNING: {n_fallback} cluster(s) could not form a field and fell "
              "back to a HEMISPHERE -- these are the only axisymmetric crowns in "
              "the output. Raise --min-hull-points above "
              f"{cfr.FieldSettings.minimum_points} to drop them instead.")
    if args.crown_model == "field" and coverage_count:
        print(f"[veg] Crown model: implicit union-of-balls field "
              f"(bridge radius {args.crown_bridge_radius:.2f} m) surfaced by "
              "marching cubes -- no shape prior, no axis of symmetry.")
        print(f"[veg]   mean point containment {coverage_total / coverage_count:.1%}, "
              f"mean {face_total / coverage_count:.0f} faces/crown; "
              f"{n_not_watertight} crown(s) not watertight, {n_repaired} hole-filled")
    if args.crown_model == "reconstruct" and coverage_count:
        print(f"[veg] Crown model: adaptive alpha-shape reconstruction at "
              f"{args.crown_voxel_size:.2f} m target resolution -- no geometric "
              "primitive assumed.")
        print(f"[veg]   mean alpha {alpha_total / coverage_count:.2f} m, "
              f"mean point coverage {coverage_total / coverage_count:.1%}, "
              f"mean {face_total / coverage_count:.0f} faces/crown")
    if args.crown_model == "radial" and coverage_count:
        print(f"[veg] Crown model: radial star-shaped hull "
              f"({args.crown_lat_bands}x{args.crown_lon_sectors} grid). "
              f"Mean measured-bin coverage {coverage_total / coverage_count:.1%}; "
              "the remainder -- chiefly the underside, which airborne LiDAR "
              "cannot see -- is the tapering prior.")
        if n_fallback:
            print(f"[veg] {n_fallback} crown(s) too sparse for the radial grid "
                  "fell back to the legacy dome rather than being dropped")
    print(f"[veg] Macro-clusters segmented into multiple trees: {n_macro_segmented}")
    print(f"[veg] Macro-clusters treated as a single tree: {n_macro_passthrough}")
    if strat_settings.enabled:
        gaps = strat_report.gap_heights_m
        print(f"[veg] Vertical stratification: {strat_report.n_cuts} cut(s) from "
              f"{strat_report.n_gaps_found} candidate void(s); refused "
              f"{strat_report.n_refused_trunk} as trunks, "
              f"{strat_report.n_refused_population} as too sparse")
        if gaps:
            print(f"[veg]   voids cut: median {np.median(gaps):.1f} m, "
                  f"max {max(gaps):.1f} m; {strat_report.n_points_dropped:,} in-gap "
                  "points discarded (sparse trunk returns -- keeping them would "
                  "rebuild the pillar being removed)")
        print(f"[veg]   low-vegetation strata: {n_low_vegetation} "
              f"({'dropped' if args.drop_low_vegetation else 'meshed as separate shade'})")
    print(f"[veg] Trees kept: {n_trees_kept}, skipped (too few points): {n_trees_skipped}")

    if not tree_meshes:
        print("[veg] No valid crown domes produced -- writing empty placeholder STL.")
        trimesh.Trimesh().export(str(out_path))
        print(f"[veg_result] n_trees=0 total_faces=0 output={out_path}")
        return

    combined = trimesh.util.concatenate(tree_meshes)
    combined.export(str(out_path))

    total_faces = len(combined.faces)
    avg_faces = total_faces / n_trees_kept
    print(f"[veg] Total faces: {total_faces} across {n_trees_kept} trees "
          f"({avg_faces:.1f} faces/tree average)")
    print(f"[veg] Wrote: {out_path}")
    print(f"[veg_result] n_trees={n_trees_kept} n_skipped={n_trees_skipped} "
          f"n_macro_segmented={n_macro_segmented} n_macro_passthrough={n_macro_passthrough} "
          f"n_vertical_cuts={strat_report.n_cuts} "
          f"n_low_vegetation={n_low_vegetation} n_low_dropped={n_low_dropped} "
          f"total_faces={total_faces} avg_faces_per_tree={avg_faces:.1f} output={out_path}")


if __name__ == "__main__":
    main()
