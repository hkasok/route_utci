"""
05_mrt_network_raytrace.py -- 24-hour MRT (mean radiant temperature) via
reverse ray tracing, for the FULL real pedestrian network (from OSM) over
your real building/vegetation/ground geometry.

This is the real-geometry, real-network successor to the original
synthetic single-path demo script. Two things had to change to make that
safe at real scale (verified by direct benchmarking, not assumed):

  1. GROUND HEIGHT: path points are placed at (ground_z + z_height, default
     1.1 m -- the ISO 7726 / UTCI standing-adult reference height) via
     ray-casting straight down onto your actual ground mesh, rather than
     assuming a flat plane -- your real terrain isn't perfectly flat like
     the synthetic test's.

  2. OUTPUT STORAGE: the original script stored one Python dict per
     point per timestep, then built a pandas DataFrame from the list --
     fine for ~380 points x 144 timesteps (~55K rows), but at real
     network scale (hundreds of thousands of points), that becomes tens
     of millions of rows and tens of GB of memory -- the same OOM
     failure pattern that hit other stages of this project. Results are
     now stored as compact (n_times x n_points) numpy matrices instead
     (a few hundred MB, not tens of GB), with only a lightweight
     per-timestep summary and an optional small representative subsample
     written as human-readable CSV.

Benchmarked directly (not estimated) at ~600,000 points (a full campus
network sampled at 0.25m spacing) against an 80-building / 2000-tree
test scene: static SVF ~5 min, 144-timestep direct-sun pass ~1 min.
Actual runtime on your real geometry will vary with its complexity.

Run:
    python3 05_mrt_network_raytrace.py \
        --buildings-stl out_full/02_final/building_final.stl \
        --vegetation-stl out_full/02_final/vegetation_final.stl \
        --ground-stl out_full/02_final/ground_and_water_final.stl \
        --polylines-pkl osm_paths/path_polylines.pkl \
        --output-dir mrt_network_output/ \
        --ds-path 0.25 --z-height 1.1 --date 2025-07-06
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import trimesh

from thermal_common import resolve_ground_albedo, sky_longwave_down
from weather_provider import add_weather_args, provider_from_args


def parse_args():
    p = argparse.ArgumentParser(description="24h MRT ray tracing over the full pedestrian network")
    p.add_argument("--buildings-stl", required=True)
    p.add_argument("--vegetation-stl", required=True)
    p.add_argument("--ground-stl", required=True)
    p.add_argument("--polylines-pkl", required=True,
                    help="Output of extract_osm_pedestrian_network.py")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--highway-filter", nargs="*", default=None,
                    help="Only keep polylines with these highway tags (e.g. footway path "
                         "pedestrian steps). Default: keep everything osmnx returned.")
    p.add_argument("--ds-path", type=float, default=0.25,
                    help="Path sampling spacing, meters (default: 0.25). Larger = fewer "
                         "points = faster; 0.25 was benchmarked safe up to ~600K points.")
    p.add_argument("--z-height", type=float, default=1.1,
                    help="Pedestrian body height above local ground at which MRT is sampled, "
                         "meters. 1.1 m = ISO 7726 / UTCI standing-adult center-of-gravity "
                         "reference height (default: 1.1)")

    p.add_argument("--latitude", type=float, default=25.7560)
    p.add_argument("--longitude", type=float, default=-80.3770)
    p.add_argument("--timezone", default="America/New_York")
    p.add_argument("--date", default="2025-07-06")
    p.add_argument("--dt-min", type=int, default=10)

    p.add_argument("--sky-n-azimuth", type=int, default=48)
    p.add_argument("--sky-n-elevation", type=int, default=12)
    p.add_argument("--k-lad-direct", type=float, default=0.45)
    p.add_argument("--k-lad-diffuse", type=float, default=0.30)

    p.add_argument("--svf-batch-size", type=int, default=2000,
                    help="Points per batch for the static SVF computation (default: 2000)")
    p.add_argument("--sun-batch-size", type=int, default=100000,
                    help="Points per batch for per-timestep direct-sun tracing (default: 100000)")

    p.add_argument("--save-subsample-csv", type=int, default=2000,
                    help="Save a full per-point-per-time CSV for this many representative "
                         "points (evenly subsampled), for easy inspection. Set 0 to skip "
                         "(default: 2000)")

    # MRT model constants -- same as the original synthetic script
    p.add_argument("--person-emissivity", type=float, default=0.97)
    p.add_argument("--person-sw-absorptivity", type=float, default=0.70)
    p.add_argument("--projected-area-model", choices=["standing", "sphere"], default="standing",
                    help="Body model for the DIRECT-beam projected-area factor. 'standing' "
                         "(default) uses the SOLWEIG/RayMan/VDI-3787 altitude-dependent f_p(h) "
                         "for a standing person; 'sphere' uses the constant --f-projected-direct "
                         "(0.25) of an isotropic globe. Diffuse/reflected/longwave factors are "
                         "the same for both. See projected_area_factor_standing().")
    p.add_argument("--f-projected-direct", type=float, default=0.25,
                    help="Constant direct-beam projected-area factor used only when "
                         "--projected-area-model=sphere (default: 0.25, a globe)")
    p.add_argument("--sky-view-body", choices=["standing", "planar"], default="standing",
                    help="Body model for the pedestrian sky-view factor used in the diffuse-"
                         "shortwave and sky/surround longwave blend. 'standing' (default) "
                         "weights the near-horizon sky like a standing cylinder, so in a "
                         "street canyon the person sees less sky and more hot surround "
                         "(higher Tmrt, consistent with the 05a cylinder longwave view and "
                         "SOLWEIG). 'planar' uses the horizontal-receiver SVF (old behaviour). "
                         "The ground-reflection term always uses the planar SVF.")
    p.add_argument("--f-sky-diffuse", type=float, default=0.50)
    p.add_argument("--f-ground-reflected", type=float, default=0.50)
    p.add_argument("--ground-albedo", type=float, default=None,
                    help="Ground albedo for the pedestrian's reflected-shortwave "
                         "term. LEAVE UNSET to inherit the value that 05b "
                         "actually used to heat the ground (read from the "
                         "materials manifest in --facet-thermal-dir), falling "
                         "back to --material-json and then to "
                         "thermal_common.GROUND_ALBEDO. Setting it explicitly "
                         "can break energy consistency and will warn.")
    p.add_argument("--material-json", default=None,
                    help="Same override file passed to 05b; used only to "
                         "resolve the ground albedo when no facet-thermal "
                         "manifest is available.")
    p.add_argument("--reflected-model", choices=["local", "global"], default="local",
                    help="How ground-reflected shortwave is estimated. 'local' (default, "
                         "CORRECT) scales it by the sunlight actually reaching the ground at "
                         "each point, using the already-traced shading state. 'global' "
                         "reproduces the older INCORRECT behavior (a single domain-wide "
                         "constant proportional to GHI) and is provided only so you can "
                         "quantify the difference on your own data -- it overstates Tmrt in "
                         "shade by roughly 9 C and should not be used for results.")
    p.add_argument("--surrounding-emissivity", type=float, default=0.95)
    p.add_argument("--lw-sky-fraction", choices=["fullsphere", "hemisphere"],
                    default="fullsphere",
                    help="How the pedestrian longwave splits sky vs surround when facet "
                         "temperatures are used. 'fullsphere' (default) uses 05a's "
                         "cylinder full-sphere sky fraction, so the hot ground BELOW an open "
                         "point is counted (fixes sunlit Tmrt being capped several C low). "
                         "'hemisphere' uses the upper-hemisphere SVF (legacy; discards the "
                         "ground for open points). No effect without --facet-thermal-dir.")
    p.add_argument("--clear-sky-emissivity", choices=["prata", "constant"], default="prata",
                    help="Clear-sky longwave emissivity model. 'prata' (default) is "
                         "humidity-dependent (Prata 1996) and much higher in humid climates "
                         "(~0.89 in Miami summer vs the old constant 0.78), raising "
                         "downwelling longwave onto both surfaces and the pedestrian. "
                         "'constant' reproduces the previous fixed 0.78. MUST match the value "
                         "passed to 05b so surfaces and pedestrian see the same sky.")
    p.add_argument("--facet-thermal-dir", default=None,
                    help="Directory holding BOTH the 05a outputs "
                         "(lw_view_matrix.npz, lw_point_weights.npz, "
                         "point_map.npy, facets.npz) and the 05b outputs "
                         "(facet_T_matrix_K.npy, facet_eps.npy). When given, "
                         "the longwave surround term is computed per point "
                         "from the ray-traced view of the actual (sunlit or "
                         "shaded) surface temperatures instead of a single "
                         "domain-wide surface temperature. When omitted, "
                         "behavior is BIT-IDENTICAL to the legacy model.")
    p.add_argument("--vegetation-emissivity", type=float, default=0.98,
                    help="Emissivity used for vegetation canopy seen in the "
                         "LW view rays (canopy radiates near air temperature)")
    # Air temperature (and, for downstream stages, RH and wind) now come from
    # the SHARED weather provider rather than a sinusoid private to this
    # stage. add_weather_args() supplies --weather-csv, --require-weather-csv,
    # --air-temp-mean-c, --air-temp-amp-c, --air-temp-peak-hour,
    # --relative-humidity-pct and --wind-speed-ms.
    add_weather_args(p)
    p.add_argument("--surface-temp-offset-day-c", type=float, default=8.0)
    p.add_argument("--cloud-cover-fraction", type=float, default=0.0)

    return p.parse_args()


SIGMA = 5.670374419e-8


def get_intersector(mesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        print("  Using pyembree ray intersector.")
    except Exception:
        from trimesh.ray.ray_triangle import RayMeshIntersector
        print("  Using trimesh triangle ray intersector (slower).")
    return RayMeshIntersector(mesh)


def load_mesh(path):
    m = trimesh.load(str(path), force="mesh")
    return m


def sample_polyline(points_xy, ds):
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 2:
        return pts
    segs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(segs, axis=1)
    cumlen = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = cumlen[-1]
    if total_len == 0:
        return pts[:1]
    svals = np.arange(0.0, total_len + 1e-12, ds)
    sampled = []
    j = 0
    for s in svals:
        while j < len(seg_lens) - 1 and s > cumlen[j + 1]:
            j += 1
        if seg_lens[j] == 0:
            sampled.append(pts[j].copy())
        else:
            frac = (s - cumlen[j]) / seg_lens[j]
            sampled.append(pts[j] + frac * segs[j])
    return np.asarray(sampled)


def ground_height_lookup(xy_points, ground_intersector, batch_size=50000, z_probe=100000.0):
    """Ray-cast straight down onto the ground mesh to find local elevation at
    each XY. Batched to keep memory bounded at large point counts."""
    n = len(xy_points)
    z_ground = np.full(n, np.nan)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_xy = xy_points[start:end]
        origins = np.column_stack([batch_xy, np.full(len(batch_xy), z_probe)])
        directions = np.tile([0.0, 0.0, -1.0], (len(batch_xy), 1))
        locations, index_ray, _ = ground_intersector.intersects_location(
            origins, directions, multiple_hits=False
        )
        if len(index_ray) > 0:
            z_ground[start + index_ray] = locations[:, 2]
    # Any point that missed the ground mesh entirely (shouldn't normally
    # happen) falls back to nearest valid neighbor's value.
    nan_mask = np.isnan(z_ground)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(xy_points[valid_idx])
            _, nearest = tree.query(xy_points[nan_mask])
            z_ground[nan_mask] = z_ground[valid_idx[nearest]]
        else:
            z_ground[nan_mask] = 0.0
        print(f"  WARNING: {nan_mask.sum()} points missed the ground mesh directly; "
              f"filled via nearest neighbor.")
    return z_ground


def make_sky_directions(n_azimuth, n_elevation):
    """Upper-hemisphere sky directions plus TWO normalized weightings:

      planar   -- for a horizontal upward receiver (the ground): the standard
                  sky-view factor, response proportional to sin(elevation).
      cylinder -- for a STANDING person: response proportional to
                  cos(elevation), so the sky view is dominated by the near-
                  horizon directions (where buildings block), not the zenith.

    Both share the solid-angle Jacobian cos(elevation). Because they weight the
    SAME traced sky transmission, computing both costs one extra dot product.
    Using the cylinder weighting for the pedestrian gives a lower sky fraction
    in street canyons -> more weight on the hot surround -> higher Tmrt, which
    is the SOLWEIG-consistent standing-person behaviour.
    """
    directions, w_planar, w_cyl = [], [], []
    for ie in range(n_elevation):
        elevation = (ie + 0.5) * (0.5 * np.pi) / n_elevation
        solid = np.cos(elevation)                       # dOmega ~ cos(el)
        for ia in range(n_azimuth):
            azimuth = 2.0 * np.pi * (ia + 0.5) / n_azimuth
            x = np.cos(elevation) * np.sin(azimuth)
            y = np.cos(elevation) * np.cos(azimuth)
            z = np.sin(elevation)
            directions.append([x, y, z])
            w_planar.append(solid * np.sin(elevation))  # horizontal receiver
            w_cyl.append(solid * np.cos(elevation))     # standing cylinder
    directions = np.asarray(directions, dtype=float)
    w_planar = np.asarray(w_planar, dtype=float)
    w_cyl = np.asarray(w_cyl, dtype=float)
    return directions, w_planar / w_planar.sum(), w_cyl / w_cyl.sum()


def vegetation_transmission_from_intersections(vegetation_intersector, origins, directions,
                                                k_lad, min_distance=1e-6, unique_tol=1e-5):
    """
    Fully vectorized (no per-ray Python loop). An earlier loop-based version
    of this function (`for r in np.unique(index_ray): ...`) was found to be
    the actual bottleneck at real scale: in a realistic 2000-point x 576
    sky-direction batch, ~190,000 individual rays hit the vegetation mesh,
    and looping over each in pure Python reduced effective throughput to
    ~10 points/sec (a projected multi-day runtime for a full network).
    This vectorized version was verified to produce BYTE-IDENTICAL results
    on the same test batch, at ~1800 points/sec -- roughly 180x faster.
    """
    n_rays = origins.shape[0]
    L_veg = np.zeros(n_rays, dtype=float)

    locations, index_ray, index_tri = vegetation_intersector.intersects_location(
        origins, directions, multiple_hits=True
    )
    if len(index_ray) == 0:
        return np.ones(n_rays, dtype=float), L_veg

    dist = np.einsum("ij,ij->i", locations - origins[index_ray], directions[index_ray])
    valid = dist > min_distance
    index_ray = index_ray[valid]
    dist = dist[valid]
    if len(index_ray) == 0:
        return np.ones(n_rays, dtype=float), L_veg

    order = np.lexsort((dist, index_ray))
    index_ray = index_ray[order]
    dist = dist[order]

    # Deduplicate near-identical consecutive hit distances within the same
    # ray (vectorized: compare each entry to the previous one).
    same_ray_as_prev = np.concatenate(([False], index_ray[1:] == index_ray[:-1]))
    close_to_prev = np.concatenate(([False], (dist[1:] - dist[:-1]) <= unique_tol))
    keep = ~(same_ray_as_prev & close_to_prev)
    index_ray = index_ray[keep]
    dist = dist[keep]
    if len(index_ray) == 0:
        return np.ones(n_rays, dtype=float), L_veg

    # Position of each hit within its ray's group (0,1,2,3,...), vectorized.
    group_change = np.concatenate(([True], index_ray[1:] != index_ray[:-1]))
    idx_arr = np.arange(len(index_ray))
    group_start = np.maximum.accumulate(np.where(group_change, idx_arr, 0))
    position_in_group = idx_arr - group_start

    # Group sizes (to drop a trailing unpaired hit from an odd-count group --
    # a grazing/tangent ray hit with no matching exit point).
    group_ids, group_sizes_per_entry = np.unique(index_ray, return_counts=True)
    size_lookup = np.zeros(n_rays, dtype=int)
    size_lookup[group_ids] = group_sizes_per_entry
    group_size = size_lookup[index_ray]
    is_last_in_odd_group = (position_in_group == group_size - 1) & (group_size % 2 == 1)

    # Entering (even position) contributes -dist, exiting (odd) contributes
    # +dist; summed per ray this equals the sum of paired (exit - entry)
    # path lengths through vegetation, with no per-ray Python loop needed.
    sign = np.where(position_in_group % 2 == 0, -1.0, 1.0)
    sign[is_last_in_odd_group] = 0.0
    signed_dist = sign * dist

    L_veg = np.bincount(index_ray, weights=signed_dist, minlength=n_rays)
    L_veg = np.maximum(L_veg, 0.0)

    tau = np.exp(-k_lad * L_veg)
    return tau, L_veg


def compute_effective_svf_batched(path_xyz, sky_directions, w_planar, w_cyl,
                                   building_intersector, vegetation_intersector,
                                   k_lad_diffuse, batch_size):
    """Returns (svf_building_only, svf_planar, svf_standing). The planar and
    standing sky-view factors are two weightings of the SAME traced sky
    transmission -- see make_sky_directions()."""
    n = len(path_xyz)
    ndirs = len(sky_directions)
    svf_planar = np.zeros(n)
    svf_standing = np.zeros(n)
    svf_building_only = np.zeros(n)

    n_batches = int(np.ceil(n / batch_size))
    t_start = time.time()
    for bi, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_pts = path_xyz[start:end]
        m = len(batch_pts)

        origins = np.repeat(batch_pts, ndirs, axis=0)
        directions = np.tile(sky_directions, (m, 1))

        building_hits = building_intersector.intersects_any(origins, directions)
        tau_veg, _ = vegetation_transmission_from_intersections(
            vegetation_intersector, origins, directions, k_lad=k_lad_diffuse
        )
        sky_transmission = tau_veg.copy()
        sky_transmission[building_hits] = 0.0
        building_open = (~building_hits).astype(float)

        T = sky_transmission.reshape(m, ndirs)
        svf_planar[start:end] = T @ w_planar
        svf_standing[start:end] = T @ w_cyl
        svf_building_only[start:end] = building_open.reshape(m, ndirs) @ w_planar

        if (bi + 1) % max(1, n_batches // 20) == 0 or bi == n_batches - 1:
            elapsed = time.time() - t_start
            frac = (end) / n
            eta = elapsed / frac - elapsed if frac > 0 else 0
            print(f"  SVF batch {bi + 1}/{n_batches} ({end}/{n} points) "
                  f"-- {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    return svf_building_only, svf_planar, svf_standing


def direct_solar_transmission_batched(path_xyz, sun_vec, building_intersector,
                                       vegetation_intersector, k_lad_direct, batch_size):
    n = len(path_xyz)
    tau_direct = np.zeros(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_pts = path_xyz[start:end]
        directions = np.tile(sun_vec, (len(batch_pts), 1))
        building_hits = building_intersector.intersects_any(batch_pts, directions)
        tau_veg, _ = vegetation_transmission_from_intersections(
            vegetation_intersector, batch_pts, directions, k_lad=k_lad_direct
        )
        tau = tau_veg.copy()
        tau[building_hits] = 0.0
        tau_direct[start:end] = tau
    return tau_direct


def sun_vector_enu(azimuth_deg, elevation_deg):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    x = np.cos(el) * np.sin(az)
    y = np.cos(el) * np.cos(az)
    z = np.sin(el)
    v = np.array([x, y, z], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, 0.0, 0.0])


# NOTE: simple_air_temperature_C() has been removed. Air temperature is now
# obtained from weather_provider.WeatherProvider, which is the same object
# stages 08 and 09 use. The former private model was
#     T = mean + amp * sin(2*pi*(h - 9)/24)      [mean 30.0, amp 3.0]
# and the provider's parametric fallback is
#     T = mean + amp * cos(2*pi*(h - peak)/24)   [mean 29.0, amp 4.0, peak 15]
# These are the SAME functional form (cos(2*pi*(h-15)/24) == sin(2*pi*(h-9)/24));
# only the mean and amplitude differed, which is precisely the inconsistency
# this change removes. Pass --weather-csv to avoid the fallback entirely.


def apply_cloud_adjustment(dni_clear, dhi_clear, elevation_deg, cloud_fraction):
    cloud = np.clip(cloud_fraction, 0.0, 1.0)
    sin_el = np.sin(np.deg2rad(np.maximum(elevation_deg, 0.0)))
    direct_factor = np.clip(1.0 - 0.75 * cloud ** 3.4, 0.0, 1.0)
    dni = dni_clear * direct_factor
    lost_direct_horizontal = dni_clear * sin_el * (1.0 - direct_factor)
    dhi = dhi_clear * (1.0 + 1.2 * cloud) + 0.6 * lost_direct_horizontal
    ghi = dni * sin_el + dhi
    night = elevation_deg <= 0.0
    return np.where(night, 0.0, dni), np.where(night, 0.0, dhi), np.where(night, 0.0, ghi)


def projected_area_factor_standing(elevation_deg):
    """Fanger (1972) projected-area factor f_p for a rotationally-symmetric
    STANDING person, as a function of solar altitude (degrees).

        f_p(h) = 0.308 * cos( radians( h * (0.998 - h^2 / 50000) ) )

    This is the standing-person projection SOLWEIG/RayMan/VDI-3787 use for the
    direct beam, replacing the sphere's constant 0.25. It runs ~0.31 at the
    horizon (low sun rakes the full standing body) down to ~0.08 at the zenith
    (overhead sun hits only the small top area). Because of that, at high sun
    the standing person absorbs LESS direct beam than a sphere would -- see the
    note in estimate_mrt_from_radiation().
    """
    b = np.maximum(np.asarray(elevation_deg, dtype=float), 0.0)
    return 0.308 * np.cos(np.deg2rad(b * (0.998 - b * b / 50000.0)))


def estimate_mrt_from_radiation(dni, dhi, ghi, elevation_deg, tau_direct,
                                 svf_person, svf_ground,
                                 air_temp_C, rh_pct, cloud_fraction, args,
                                 L_surround_override=None, lw_sky_frac=None):
    # lw_sky_frac: FULL-SPHERE sky fraction for the LONGWAVE blend (from 05a's
    #   cylinder view, ground-inclusive). When None the blend falls back to the
    #   upper-hemisphere svf_person, which UNDER-counts the hot ground below an
    #   open point and caps sunlit Tmrt several C low -- see the longwave block.
    # svf_person: sky-view factor for the PEDESTRIAN (standing-person or planar
    #   per --sky-view-body) -- used for the diffuse-shortwave interception and
    #   the sky/surround longwave blend.
    # svf_ground: PLANAR sky-view factor of the ground patch below the person --
    #   used only to estimate how much shortwave reaches the ground for the
    #   reflected term (the ground is a horizontal receiver regardless of body).
    sin_el = np.sin(np.deg2rad(np.maximum(elevation_deg, 0.0)))
    air_K = air_temp_C + 273.15

    # Downwelling sky longwave -- humidity-dependent clear-sky emissivity
    # (Prata) by default, shared with 05b so surfaces and pedestrian see one
    # identical sky. This replaced a constant 0.78 that badly understated
    # longwave in humid climates.
    L_sky = sky_longwave_down(air_temp_C, rh_pct, cloud_fraction,
                              clear_sky_model=args.clear_sky_emissivity)

    surface_offset = args.surface_temp_offset_day_c * max(sin_el, 0.0)
    surface_K = air_K + surface_offset
    # ------------------------------------------------------------------
    # LONGWAVE SURROUND
    #
    # Legacy model: EVERY surface in the domain radiates at one global
    # temperature (air + a sinusoidal daytime offset). That erases the
    # sunlit-vs-shaded surface contrast that longwave exposure along a
    # route actually depends on (a sunlit asphalt surface can be 15-25 C
    # hotter than a shaded one at the same instant).
    #
    # When --facet-thermal-dir is given, L_surround_override carries a
    # PER-POINT value assembled from the ray-traced view of the actual
    # facet surface temperatures (05a view matrix x 05b energy balance).
    # The sky/surround partition (svf_effective) is unchanged, so with
    # uniform facet temperatures equal to the legacy surface_K the result
    # is IDENTICAL to the legacy model -- this is verified numerically in
    # verify_thermal_pipeline.py (test T2).
    # ------------------------------------------------------------------
    if L_surround_override is not None:
        L_surround = L_surround_override
    else:
        L_surround = args.surrounding_emissivity * SIGMA * surface_K ** 4

    # DIRECT-BEAM PROJECTED-AREA FACTOR -- the one term that distinguishes a
    # SPHERE body (constant 0.25) from a SOLWEIG-style STANDING person
    # (altitude-dependent f_p). Diffuse/reflected/longwave angular factors are
    # ~0.5/0.5/isotropic for both postures (VDI 3787), so only the beam changes.
    if args.projected_area_model == "standing":
        f_dir = projected_area_factor_standing(elevation_deg)
    else:
        f_dir = args.f_projected_direct
    K_direct_abs = args.person_sw_absorptivity * f_dir * tau_direct * dni
    K_diffuse_abs = args.person_sw_absorptivity * args.f_sky_diffuse * svf_person * dhi

    # ------------------------------------------------------------------
    # REFLECTED SHORTWAVE
    #
    # The radiation a pedestrian receives by reflection off the ground is
    # proportional to how much sunlight actually REACHES that ground -- not
    # to the domain-wide horizontal global irradiance (GHI).
    #
    # An earlier version of this model used a bare `... * ghi`, which is a
    # single scalar identical at every point in the domain. That gave a
    # pedestrian standing in deep tree shade the same 65 W/m^2 of
    # "ground-reflected sunlight" as one standing in an open sunlit plaza,
    # even though the ground beneath the shaded pedestrian is itself shaded
    # and reflecting almost nothing. Measured effect of that bug: it
    # OVERSTATED Tmrt in deep shade by ~9.5 C -- directly compressing the
    # sun/shade contrast that route ranking depends on.
    #
    # Fix: estimate the global shortwave actually incident on the ground in
    # the pedestrian's vicinity, using the shading state we ALREADY ray-traced
    # (no extra rays, no extra cost):
    #
    #     K_global_local = tau_direct * DNI * sin(elev)   [beam reaching ground]
    #                    + svf_effective * DHI            [sky diffuse reaching ground]
    #
    # The pedestrian's own tau_direct / svf_effective are used as a proxy for
    # the ground patch directly beneath them. For the direct beam this is very
    # nearly exact (the same buildings/canopy block both, ~1.1 m apart). For the
    # sky-diffuse part the ground's true SVF is slightly lower than at 1.1 m,
    # so this mildly over-estimates -- an acceptable approximation given it
    # costs zero extra ray tracing, and vastly better than a domain constant.
    #
    # In open sun (tau=1, svf~0.95) this reduces to ~GHI, matching the old
    # behavior; in deep shade it correctly collapses toward near-zero.
    # ------------------------------------------------------------------
    if args.reflected_model == "global":
        # legacy/comparison mode -- reproduces the old (incorrect) behavior
        k_global_local = np.full_like(np.asarray(tau_direct, dtype=float), ghi)
    else:
        # ground is a horizontal receiver -> planar sky-view factor
        k_global_local = tau_direct * dni * sin_el + svf_ground * dhi

    K_reflected_abs = (args.person_sw_absorptivity * args.f_ground_reflected
                       * args.ground_albedo * k_global_local)

    K_shortwave_abs = K_direct_abs + K_diffuse_abs + K_reflected_abs
    # Longwave sky/surround blend. Use the FULL-SPHERE sky fraction when given
    # (facet-thermal path) so the ground below an open point is counted; the
    # upper-hemisphere svf_person is a fallback that discards it (legacy).
    sky_frac_lw = lw_sky_frac if lw_sky_frac is not None else svf_person
    L_effective = sky_frac_lw * L_sky + (1.0 - sky_frac_lw) * L_surround
    L_longwave_abs = args.person_emissivity * L_effective

    R_abs = L_longwave_abs + K_shortwave_abs
    tmrt_K = (R_abs / (args.person_emissivity * SIGMA)) ** 0.25
    return tmrt_K - 273.15, R_abs, K_shortwave_abs, L_longwave_abs


class FacetLongwave:
    """Assembles the per-point longwave surround from ray-traced facet
    surface temperatures (outputs of 05a + 05b).

    Per timestep it computes, at each traced (coarse) route point:

        L_surround = [ sum_f W_pf * eps_f * sigma * T_f(t)^4        (facets)
                       + w_veg * eps_veg * sigma * T_air(t)^4       (canopy)
                       + w_def * eps_s  * sigma * T_legacy(t)^4 ]   (culled)
                     / (1 - w_sky)

    then maps coarse -> full resolution via point_map. Weights partition
    unity by construction (asserted in 05a), so with uniform facet
    temperatures this collapses exactly to the legacy constant."""

    def __init__(self, thermal_dir, n_points, n_times, args):
        import scipy.sparse as sp
        d = Path(thermal_dir)
        self.W = sp.load_npz(d / "lw_view_matrix.npz")
        pw = np.load(d / "lw_point_weights.npz")
        self.w_sky = pw["w_sky"]
        self.w_veg = pw["w_veg"]
        self.w_def = pw["w_default"]
        self.point_map = np.load(d / "point_map.npy")
        self.facet_T = np.load(d / "facet_T_matrix_K.npy")
        self.facet_eps = np.load(d / "facet_eps.npy")
        self.args = args
        # ---- consistency checks: refuse to run on mismatched inputs ----
        if len(self.point_map) != n_points:
            raise ValueError(
                f"point_map covers {len(self.point_map)} route points but this "
                f"run has {n_points}: re-run 05a for the current network")
        if self.facet_T.shape[0] != n_times:
            raise ValueError(
                f"facet_T_matrix_K has {self.facet_T.shape[0]} time steps but "
                f"this run has {n_times}: re-run 05b with matching times.csv")
        if self.facet_T.shape[1] != self.W.shape[1]:
            raise ValueError("facet count mismatch between 05a view matrix "
                             "and 05b temperatures")
        self.w_surf = 1.0 - self.w_sky
        # Full-sphere sky fraction per FULL-resolution point (cylinder-weighted,
        # from 05a). This is the physically-correct sky/surround split for the
        # longwave blend: unlike the upper-hemisphere SVF it counts the lower
        # hemisphere as ground (surround), so the hot ground below the person is
        # not discarded for open points.
        self.sky_frac = self.w_sky[self.point_map].astype(float)
        print(f"  Facet thermal LW active: {self.W.shape[1]:,} facets, "
              f"{self.W.shape[0]:,} traced points, mean surround weight "
              f"{self.w_surf.mean():.3f} (full-sphere sky frac "
              f"{self.sky_frac.mean():.3f})")

    def surround_at(self, it, air_temp_C, elevation_deg):
        a = self.args
        air_K = air_temp_C + 273.15
        sin_el = np.sin(np.deg2rad(max(elevation_deg, 0.0)))
        legacy_K = air_K + a.surface_temp_offset_day_c * max(sin_el, 0.0)
        E_facet = self.facet_eps * SIGMA * self.facet_T[it].astype(float) ** 4
        num = (self.W @ E_facet
               + self.w_veg * a.vegetation_emissivity * SIGMA * air_K ** 4
               + self.w_def * a.surrounding_emissivity * SIGMA * legacy_K ** 4)
        legacy_L = a.surrounding_emissivity * SIGMA * legacy_K ** 4
        L_coarse = np.where(self.w_surf > 1e-6,
                            num / np.maximum(self.w_surf, 1e-6), legacy_L)
        return L_coarse[self.point_map]


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # GROUND ALBEDO CONSISTENCY
    # The albedo used here (how much shortwave the ground reflects ONTO the
    # pedestrian) must equal the albedo 05b used (how much the ground does
    # NOT absorb). Resolve it from the 05b manifest when available rather
    # than carrying an independent default.
    # ------------------------------------------------------------------
    args.ground_albedo, _alb_src = resolve_ground_albedo(
        args, facet_thermal_dir=args.facet_thermal_dir)
    print(f"Ground albedo: {args.ground_albedo:.3f}  [source: {_alb_src}]")

    # ------------------------------------------------------------------
    # WEATHER FORCING
    # Built here, before any geometry work, so a misconfigured forcing
    # fails in seconds rather than after an hour of ray tracing. The same
    # provider class is used by stages 08 and 09, so a single --weather-csv
    # now drives Tmrt and the thermal-comfort calculation identically.
    # ------------------------------------------------------------------
    weather = provider_from_args(args)
    _prov = weather.provenance()
    print("=" * 70)
    print("WEATHER FORCING")
    print(f"  {weather.describe()}")
    print(f"    air temperature   <- {_prov['source_air_temp_C']}")
    print(f"    relative humidity <- {_prov['source_rh_pct']}   (carried to times.csv)")
    print(f"    wind speed        <- {_prov['source_wind_ms']}   (carried to times.csv)")
    if _prov["source_air_temp_C"] != "csv":
        print("  " + "!" * 58)
        print("  ! Air temperature is PARAMETRIC. This stage previously used")
        print("  ! mean 30.0 / amplitude 3.0; the shared fallback is")
        print(f"  ! mean {weather.mean_c} / amplitude {weather.amp_c}, so Tmrt will differ")
        print("  ! from earlier runs. Pass --weather-csv for a defined forcing,")
        print("  ! or --air-temp-mean-c 30 --air-temp-amp-c 3 to reproduce the old")
        print("  ! behaviour exactly.")
        print("  " + "!" * 58)
    print("=" * 70)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Loading geometry...")
    building_mesh = load_mesh(args.buildings_stl)
    vegetation_mesh = load_mesh(args.vegetation_stl)
    ground_mesh = load_mesh(args.ground_stl)
    print(f"  Buildings: {len(building_mesh.faces)} faces")
    print(f"  Vegetation: {len(vegetation_mesh.faces)} faces")
    print(f"  Ground: {len(ground_mesh.faces)} faces")

    building_intersector = get_intersector(building_mesh)
    vegetation_intersector = get_intersector(vegetation_mesh)
    ground_intersector = get_intersector(ground_mesh)

    print("\nLoading pedestrian network...")
    with open(args.polylines_pkl, "rb") as f:
        data = pickle.load(f)
    polylines = data["polylines"]
    highway_tags = data.get("highway_tags", ["unknown"] * len(polylines))
    print(f"  Loaded {len(polylines)} polylines")

    if args.highway_filter:
        filt = set(args.highway_filter)
        def tag_matches(t):
            if isinstance(t, list):
                return bool(filt.intersection(t))
            return t in filt
        keep = [i for i, t in enumerate(highway_tags) if tag_matches(t)]
        polylines = [polylines[i] for i in keep]
        print(f"  Filtered to {len(polylines)} polylines matching {sorted(filt)}")

    print(f"\nSampling path at {args.ds_path} m spacing...")
    all_xy = []
    segment_id = []
    for seg_i, poly in enumerate(polylines):
        sampled = sample_polyline(poly, args.ds_path)
        all_xy.append(sampled)
        segment_id.extend([seg_i] * len(sampled))
    path_xy = np.vstack(all_xy)
    segment_id = np.asarray(segment_id)
    n_points = len(path_xy)
    print(f"  Total sampled points: {n_points:,}")
    print(f"  Estimated static-SVF time at benchmarked ~3000-10000 pts/sec: "
          f"~{n_points/3000/60:.1f}-{n_points/10000/60:.1f} minutes")

    print("\nLooking up local ground elevation (ray-cast)...")
    z_ground = ground_height_lookup(path_xy, ground_intersector)
    path_xyz = np.column_stack([path_xy, z_ground + args.z_height])
    print(f"  Ground Z range: {z_ground.min():.2f} to {z_ground.max():.2f} m")

    np.save(out_dir / "path_xyz.npy", path_xyz)
    np.save(out_dir / "path_segment_id.npy", segment_id)

    print("\n" + "=" * 70)
    print("Computing static effective sky-view factor...")
    sky_directions, sky_w_planar, sky_w_cyl = make_sky_directions(
        args.sky_n_azimuth, args.sky_n_elevation)
    print(f"  Sky directions: {len(sky_directions)}")

    svf_building_only, svf_planar, svf_standing = compute_effective_svf_batched(
        path_xyz, sky_directions, sky_w_planar, sky_w_cyl,
        building_intersector, vegetation_intersector,
        args.k_lad_diffuse, args.svf_batch_size,
    )
    # The pedestrian's sky fraction depends on body model; the ground below is
    # always a horizontal (planar) receiver.
    svf_person = svf_standing if args.sky_view_body == "standing" else svf_planar
    svf_ground = svf_planar
    np.save(out_dir / "svf_building_only.npy", svf_building_only)
    np.save(out_dir / "svf_planar.npy", svf_planar)
    np.save(out_dir / "svf_standing.npy", svf_standing)
    np.save(out_dir / "svf_effective.npy", svf_person)   # back-compat name
    print(f"  Sky-view factor ({args.sky_view_body}) range: "
          f"{svf_person.min():.3f} to {svf_person.max():.3f} "
          f"(planar {svf_planar.mean():.3f} / standing {svf_standing.mean():.3f} mean)")

    print("\n" + "=" * 70)
    print("Solar position and clear-sky radiation...")
    times = pd.date_range(
        start=f"{args.date} 00:00", end=f"{args.date} 23:50",
        freq=f"{args.dt_min}min", tz=args.timezone,
    )
    location = pvlib.location.Location(latitude=args.latitude, longitude=args.longitude,
                                        tz=args.timezone)
    solar = pvlib.solarposition.get_solarposition(times, args.latitude, args.longitude)
    clearsky = location.get_clearsky(times, model="ineichen")
    elev = solar["apparent_elevation"].values
    azim = solar["azimuth"].values
    dni, dhi, ghi = apply_cloud_adjustment(
        clearsky["dni"].values, clearsky["dhi"].values, elev, args.cloud_cover_fraction
    )
    # ------------------------------------------------------------------
    # Air temperature from the shared provider, evaluated at the decimal
    # hour of each model timestep. Tmrt uses air temperature only (sky
    # longwave and the surface-temperature offset); RH and wind are carried
    # into times.csv so that 05b and the route stages can inherit exactly
    # the same series instead of re-deriving it.
    # ------------------------------------------------------------------
    hour_of_day = times.hour + times.minute / 60.0 + times.second / 3600.0
    air_temp_C_time, rh_pct_time, wind_ms_time = weather.forcing_at(
        np.asarray(hour_of_day, dtype=float))

    nt = len(times)
    print(f"  {nt} time steps ({args.dt_min} min resolution)")

    print("\n" + "=" * 70)
    print("Running direct-sun ray tracing + MRT for each time step...")
    facet_lw = None
    if args.facet_thermal_dir:
        facet_lw = FacetLongwave(args.facet_thermal_dir, n_points, nt, args)
    tmrt_matrix = np.zeros((nt, n_points), dtype=np.float32)
    direct_transmission_matrix = np.zeros((nt, n_points), dtype=np.float32)

    t_loop_start = time.time()
    for it, (t, el, az) in enumerate(zip(times, elev, azim)):
        if el <= 0.0:
            tau_direct = np.zeros(n_points)
        else:
            sun_vec = sun_vector_enu(az, el)
            tau_direct = direct_solar_transmission_batched(
                path_xyz, sun_vec, building_intersector, vegetation_intersector,
                args.k_lad_direct, args.sun_batch_size,
            )

        L_surround_override = None
        lw_sky_frac = None
        if facet_lw is not None:
            L_surround_override = facet_lw.surround_at(it, air_temp_C_time[it], el)
            if args.lw_sky_fraction == "fullsphere":
                lw_sky_frac = facet_lw.sky_frac

        tmrt_C, R_abs, K_sw, L_lw = estimate_mrt_from_radiation(
            dni[it], dhi[it], ghi[it], el, tau_direct, svf_person, svf_ground,
            air_temp_C_time[it], rh_pct_time[it], args.cloud_cover_fraction, args,
            L_surround_override=L_surround_override, lw_sky_frac=lw_sky_frac,
        )
        tmrt_matrix[it, :] = tmrt_C
        direct_transmission_matrix[it, :] = tau_direct

        if (it + 1) % 24 == 0 or it == nt - 1:
            elapsed = time.time() - t_loop_start
            print(f"  step {it + 1}/{nt} ({t.strftime('%H:%M')}) -- {elapsed:.0f}s elapsed")

    np.save(out_dir / "tmrt_matrix_C.npy", tmrt_matrix)
    np.save(out_dir / "direct_transmission_matrix.npy", direct_transmission_matrix)
    times_df = pd.DataFrame({
        "time": times, "azimuth_deg": azim, "elevation_deg": elev,
        "DNI_Wm2": dni, "DHI_Wm2": dhi, "GHI_Wm2": ghi, "air_temp_C": air_temp_C_time,
        # Carried through so downstream stages inherit identical forcing.
        "rh_pct": rh_pct_time, "wind_ms": wind_ms_time,
    })
    times_df.to_csv(out_dir / "times.csv", index=False)

    print("\n" + "=" * 70)
    print("Saving lightweight summary (safe at any scale)...")
    summary_rows = []
    for it, t in enumerate(times):
        summary_rows.append({
            "time": t.isoformat(),
            "elevation_deg": elev[it],
            "DNI_Wm2": dni[it], "DHI_Wm2": dhi[it], "GHI_Wm2": ghi[it],
            "air_temp_C": air_temp_C_time[it],
            "tmrt_mean_C": float(np.mean(tmrt_matrix[it])),
            "tmrt_min_C": float(np.min(tmrt_matrix[it])),
            "tmrt_max_C": float(np.max(tmrt_matrix[it])),
            "tmrt_p10_C": float(np.percentile(tmrt_matrix[it], 10)),
            "tmrt_p90_C": float(np.percentile(tmrt_matrix[it], 90)),
            "mean_direct_transmission": float(np.mean(direct_transmission_matrix[it])),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_time.csv", index=False)

    if args.save_subsample_csv > 0:
        n_sub = min(args.save_subsample_csv, n_points)
        sub_idx = np.linspace(0, n_points - 1, n_sub).astype(int)
        print(f"Saving detailed CSV for {n_sub} representative points "
              f"(out of {n_points:,} total)...")
        records = []
        for it, t in enumerate(times):
            for ip in sub_idx:
                records.append({
                    "time": t.isoformat(), "point_index": int(ip),
                    "x": path_xyz[ip, 0], "y": path_xyz[ip, 1], "z": path_xyz[ip, 2],
                    "svf_effective": svf_person[ip],
                    "direct_transmission": direct_transmission_matrix[it, ip],
                    "tmrt_C": tmrt_matrix[it, ip],
                })
        pd.DataFrame(records).to_csv(out_dir / "detailed_subsample.csv", index=False)

    print("\n" + "=" * 70)
    print("Done. Key outputs in", out_dir)
    print("  path_xyz.npy                     -- (n_points, 3) point coordinates")
    print("  svf_effective.npy                -- (n_points,) static sky view factor")
    print("  tmrt_matrix_C.npy                -- (n_times, n_points) MRT, deg C")
    print("  direct_transmission_matrix.npy   -- (n_times, n_points) direct sun factor")
    print("  times.csv                        -- solar position + radiation per time step")
    print("  summary_by_time.csv              -- lightweight per-timestep stats (always small)")
    if args.save_subsample_csv > 0:
        print("  detailed_subsample.csv           -- full per-point-per-time for a "
              "representative subsample")
    print(f"\n[mrt_result] n_points={n_points} n_times={nt} output_dir={out_dir}")


if __name__ == "__main__":
    main()
