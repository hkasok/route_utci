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
import copy
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import trimesh

from thermal_common import (MATERIALS_MANIFEST, RADIOSITY_ENVIRONMENT,
                            RADIOSITY_MATRIX, resolve_ground_albedo,
                            sky_longwave_down)
from radiant_flux_contributions import (
    CONTRIBUTION_ARCHIVE, CONTRIBUTION_METADATA, GLOBE_COLUMNS,
    LW_SOURCE_COLUMNS, SENSOR_COLUMNS,
    PRIMARY_COLUMNS, SW_SOURCE_COLUMNS, TOTAL_COLUMNS, canonical_lw_source,
    canonical_sw_source, load_contribution_config,
    validate_contribution_arrays, write_contribution_metadata)
from black_globe import (GLOBE_PRESETS, describe as describe_globe,
                         radiative_equilibrium_temperature_C,
                         resolve_globe_spec, sphere_projected_area_factor,
                         steady_globe_temperature_C)
from weather_provider import add_weather_args, provider_from_args
from microclimate_field import EnvironmentField, add_microclimate_argument
from radiation_forcing import resolve_radiation_forcing


def parse_args():
    p = argparse.ArgumentParser(description="24h MRT ray tracing over the full pedestrian network")
    p.add_argument("--buildings-stl", required=True)
    p.add_argument("--vegetation-stl", required=True)
    p.add_argument("--ground-stl", required=True)
    p.add_argument("--polylines-pkl", required=True,
                    help="Output of extract_osm_pedestrian_network.py")
    p.add_argument("--output-dir", required=True)
    add_microclimate_argument(p)
    p.add_argument(
        "--prep-only", action="store_true",
        help=("Write route points, time forcing, and static SVF products, then "
              "exit before direct-sun ray tracing and MRT. This mode never "
              "writes tmrt_matrix_C.npy."),
    )
    p.add_argument(
        "--svf-cache", default=None, metavar="DIR",
        help=("Optional directory for exact-match reuse of geometry-only SVF "
              "arrays. Cache identity includes the STL inputs, sampled route "
              "points, sky sampling, and diffuse vegetation attenuation."),
    )
    p.add_argument(
        "--force-svf", action="store_true",
        help="Recompute SVF even when --svf-cache contains a valid exact match.",
    )
    p.add_argument(
        "--radiant-flux-config", default=None,
        help=("Optional radiant_flux_contribution_results.json. When omitted, "
              "contribution recording is disabled and legacy memory/output "
              "behavior is preserved."),
    )

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
    p.add_argument("--wall-reflected-shortwave", choices=["on", "off"], default="on",
                    help="Include shortwave reflected off route-visible WALLS and "
                         "ROOFS (default on). Sunlit facades are a real shortwave "
                         "source for a pedestrian in a street canyon and were "
                         "previously omitted, leaving only ground reflection. Uses "
                         "the same 05a directional view weights and 05b per-facet "
                         "sun/sky exposure and albedo as the longwave surround, so "
                         "no new ray tracing is required. Requires "
                         "--facet-thermal-dir; without it the legacy ground-only "
                         "reflection is unchanged.")
    p.add_argument("--sensor-equivalent-outputs", choices=["on", "off"], default="on",
                    help="Also record what a four-component net radiometer would "
                         "read at --sensor-height-m (horizontal up/down shortwave "
                         "and longwave). These are instrument-equivalent "
                         "diagnostics for like-for-like comparison against field "
                         "measurements; they are never mixed into the "
                         "body-absorbed flux or MRT.")
    p.add_argument("--sensor-emulation", choices=["facet", "legacy"],
                   default="facet",
                   help="How the emulated radiometer forms its non-sky "
                        "channels. 'facet' (default): downwelling longwave and "
                        "reflected shortwave from the 05a up-facing view, and "
                        "upwelling shortwave from each footprint ground facet's "
                        "own irradiance; falls back to 'legacy' when the "
                        "thermal folder has no up-facing view. 'legacy': the "
                        "cylinder-weighted surround mean and footprint albedo "
                        "x sensor downwelling.")
    p.add_argument("--sensor-height-m", type=float, default=1.0,
                    help="Height above local ground of the emulated radiometer "
                         "(default 1.0 m). The receptor ray tracing is performed "
                         "at --z-height; when the two differ the sky-view and "
                         "shading state of the receptor are reused and the "
                         "difference is recorded in the metadata.")
    p.add_argument("--black-globe-outputs", choices=["on", "off"], default="on",
                    help="Also record what a BLACK-GLOBE THERMOMETER would read "
                         "at --sensor-height-m. A globe is a sphere, so its "
                         "direct-beam projected-area factor is a constant 0.25 "
                         "instead of the standing body's altitude-dependent one, "
                         "and it is a thermometer rather than a radiometer -- "
                         "wind pulls it toward air temperature. Both are handled "
                         "here. Like the radiometer channels this is an "
                         "instrument emulator: it is never mixed into the "
                         "body-absorbed flux, the pedestrian MRT, UTCI or JOS-3. "
                         "Requires --facet-thermal-dir.")
    p.add_argument("--globe-preset", default="campbell_blackglobe_l",
                    choices=sorted(GLOBE_PRESETS),
                    help="Which globe to emulate (default the Campbell "
                         "Scientific BLACKGLOBE-L used by the Lisbon campaigns: "
                         "152 mm copper sphere, emittance 0.957).")
    p.add_argument("--globe-diameter-m", type=float, default=None,
                    help="Override the emulated globe diameter [m].")
    p.add_argument("--globe-emissivity", type=float, default=None,
                    help="Override the emulated globe longwave emissivity.")
    p.add_argument("--globe-sw-absorptivity", type=float, default=None,
                    help="Override the emulated globe shortwave absorptivity.")
    p.add_argument("--globe-areal-heat-capacity", type=float, default=None,
                    help="Override the globe heat capacity per unit of its own "
                         "surface area [J m-2 K-1]. This sets the response lag, "
                         "which dominates a walking measurement -- see "
                         "README_black_globe.md before changing it.")
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
    p.add_argument("--wall-temperature-offset-K", type=float, default=0.0,
                   help="Diagnostic: add this offset to every WALL facet "
                        "temperature before the longwave surround is formed. "
                        "0.0 (default) leaves the solved field untouched. Used "
                        "to attribute the daytime radiative excess to a surface "
                        "class; the perturbation propagates into the globe, the "
                        "sensor channels and the body through the existing view "
                        "weights, so no angular conversion is involved. "
                        "Requires --facet-thermal-dir.")
    p.add_argument("--ground-temperature-offset-K", type=float, default=0.0,
                   help="As --wall-temperature-offset-K but for GROUND facets.")
    p.add_argument("--roof-temperature-offset-K", type=float, default=0.0,
                   help="As --wall-temperature-offset-K but for ROOF facets.")
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
    p.add_argument(
        "--radiation-csv",
        default=None,
        help=(
            "Optional measured/reference irradiance forcing with an 'hour' or "
            "'time' column and DNI_Wm2, DHI_Wm2, GHI_Wm2 columns. Values are "
            "periodically interpolated to model timesteps. When omitted, the "
            "existing pvlib clear-sky plus cloud-adjustment model is unchanged."
        ),
    )
    p.add_argument(
        "--radiation-forcing-config", default=None,
        help=("Optional case-level radiation_forcing.json. Supports clear-sky, "
              "independent component CSV, or a documented mobile-shortwave "
              "upper-envelope cloud estimate. --radiation-csv takes precedence."),
    )
    p.add_argument("--surface-temp-offset-day-c", type=float, default=8.0)
    p.add_argument("--cloud-cover-fraction", type=float, default=0.0)

    return p.parse_args()


SIGMA = 5.670374419e-8
SVF_CACHE_SCHEMA_VERSION = 1
SVF_CACHE_METADATA = "svf_cache_metadata.json"
SVF_CACHE_ARRAYS = (
    "svf_building_only.npy",
    "svf_planar.npy",
    "svf_standing.npy",
)


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    """Return a streaming SHA-256 digest without loading a large STL at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stl_cache_identity(path):
    """Stable identity for a mesh input used by the static SVF calculation."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(resolved),
    }


def _array_cache_identity(path, array):
    """Content identity for a generated NPY, independent of output directory."""
    resolved = Path(path)
    stat = resolved.stat()
    return {
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(resolved),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def build_svf_cache_metadata(args, path_xyz, path_xyz_file):
    """Describe every input that can change the geometry-only SVF fields."""
    return {
        "schema_version": SVF_CACHE_SCHEMA_VERSION,
        "algorithm": "compute_effective_svf_batched",
        "meshes": {
            "buildings": _stl_cache_identity(args.buildings_stl),
            "vegetation": _stl_cache_identity(args.vegetation_stl),
            "ground": _stl_cache_identity(args.ground_stl),
        },
        "path_xyz": _array_cache_identity(path_xyz_file, path_xyz),
        "sky_sampling": {
            "n_azimuth": int(args.sky_n_azimuth),
            "n_elevation": int(args.sky_n_elevation),
            "k_lad_diffuse": float(args.k_lad_diffuse),
        },
    }


def load_cached_svf(cache_dir, expected_metadata, n_points):
    """Load a complete, finite exact-match cache, otherwise return None."""
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / SVF_CACHE_METADATA
    if not metadata_path.is_file():
        return None
    try:
        with open(metadata_path, encoding="utf-8") as stream:
            actual_metadata = json.load(stream)
        if actual_metadata != expected_metadata:
            print("  SVF cache metadata mismatch; recomputing.")
            return None
        arrays = tuple(np.load(cache_dir / name) for name in SVF_CACHE_ARRAYS)
        for name, values in zip(SVF_CACHE_ARRAYS, arrays):
            if values.shape != (n_points,):
                print(f"  SVF cache shape mismatch in {name}; recomputing.")
                return None
            if not np.isfinite(values).all():
                print(f"  SVF cache contains non-finite values in {name}; recomputing.")
                return None
        print(f"  reusing cached SVF from {cache_dir}")
        return arrays
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"  SVF cache is incomplete or unreadable ({exc}); recomputing.")
        return None


def write_svf_cache(cache_dir, metadata, arrays):
    """Write SVF arrays first and the validity sidecar last."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, values in zip(SVF_CACHE_ARRAYS, arrays):
        np.save(cache_dir / name, values)
    with open(cache_dir / SVF_CACHE_METADATA, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"  Wrote exact-match SVF cache: {cache_dir}")


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


def load_radiation_csv(csv_path, model_times):
    """Load DNI/DHI/GHI forcing and interpolate periodically by decimal hour."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"radiation CSV not found: {path}")
    frame = pd.read_csv(path)
    lower = {column.lower(): column for column in frame.columns}
    if "hour" in lower:
        source_hours = frame[lower["hour"]].to_numpy(dtype=float)
    elif "time" in lower:
        parsed = pd.to_datetime(frame[lower["time"]], errors="raise")
        source_hours = (
            parsed.dt.hour
            + parsed.dt.minute / 60.0
            + parsed.dt.second / 3600.0
        ).to_numpy(dtype=float)
    else:
        raise ValueError("radiation CSV requires an 'hour' or 'time' column")

    def required_column(*aliases):
        for alias in aliases:
            if alias in lower:
                return frame[lower[alias]].to_numpy(dtype=float)
        raise ValueError(
            f"radiation CSV {path} is missing required column; accepted names: "
            f"{', '.join(aliases)}"
        )

    dni_source = required_column("dni_wm2", "dni", "kdir")
    dhi_source = required_column("dhi_wm2", "dhi", "kdiff")
    ghi_source = required_column("ghi_wm2", "ghi", "kdn")
    if len(source_hours) < 2:
        raise ValueError("radiation CSV needs at least two rows to interpolate")
    arrays = (source_hours, dni_source, dhi_source, ghi_source)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"radiation CSV {path} contains non-finite values")
    if np.any(dni_source < 0) or np.any(dhi_source < 0) or np.any(ghi_source < 0):
        raise ValueError(f"radiation CSV {path} contains negative irradiance")
    order = np.argsort(source_hours)
    source_hours = source_hours[order]
    target_hours = np.asarray(
        model_times.hour
        + model_times.minute / 60.0
        + model_times.second / 3600.0,
        dtype=float,
    )
    interpolate = lambda values: np.interp(
        target_hours, source_hours, values[order], period=24.0
    )
    return interpolate(dni_source), interpolate(dhi_source), interpolate(ghi_source)


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
                                 L_surround_override=None, lw_sky_frac=None,
                                 local_ground_albedo=None,
                                 reflected_source_fractions=None,
                                 L_surround_components=None,
                                 L_sky_override=None,
                                 wall_reflected_incident=None,
                                 wall_reflected_parts=None,
                                 return_contributions=False):
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
    if L_sky_override is None or not np.isfinite(L_sky_override):
        L_sky = sky_longwave_down(air_temp_C, rh_pct, cloud_fraction,
                                  clear_sky_model=args.clear_sky_emissivity)
    else:
        L_sky = float(L_sky_override)

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

    reflecting_albedo = (args.ground_albedo if local_ground_albedo is None
                         else np.asarray(local_ground_albedo, dtype=float))
    if (not np.isfinite(reflecting_albedo).all()
            or np.any((reflecting_albedo < 0) | (reflecting_albedo > 1))):
        raise ValueError("local ground albedo must be finite and in [0,1]")
    K_ground_reflected_abs = (args.person_sw_absorptivity * args.f_ground_reflected
                              * reflecting_albedo * k_global_local)

    # Wall/roof reflected shortwave. Sunlit facades are a genuine shortwave
    # source in a street canyon; omitting them (the previous behaviour) leaves
    # only ground reflection and under-states pedestrian shortwave where the
    # sky view is small and the wall view large -- exactly the canyon case.
    # The view weighting is 05a's, so no extra ray tracing is needed.
    if wall_reflected_incident is None:
        K_wall_reflected_abs = np.zeros_like(np.asarray(K_ground_reflected_abs,
                                                        dtype=float))
    else:
        K_wall_reflected_abs = (args.person_sw_absorptivity
                                * np.asarray(wall_reflected_incident, dtype=float))
        if not np.isfinite(K_wall_reflected_abs).all() \
                or np.any(K_wall_reflected_abs < 0):
            raise ValueError("wall-reflected shortwave must be finite and non-negative")

    K_reflected_abs = K_ground_reflected_abs + K_wall_reflected_abs
    K_shortwave_abs = K_direct_abs + K_diffuse_abs + K_reflected_abs
    # Longwave sky/surround blend. Use the FULL-SPHERE sky fraction when given
    # (facet-thermal path) so the ground below an open point is counted; the
    # upper-hemisphere svf_person is a fallback that discards it (legacy).
    sky_frac_lw = lw_sky_frac if lw_sky_frac is not None else svf_person
    L_sky_abs = args.person_emissivity * sky_frac_lw * L_sky
    L_surface_abs = args.person_emissivity * (1.0 - sky_frac_lw) * L_surround
    L_longwave_abs = L_sky_abs + L_surface_abs

    R_abs = L_longwave_abs + K_shortwave_abs
    tmrt_K = (R_abs / (args.person_emissivity * SIGMA)) ** 0.25
    if not return_contributions:
        return tmrt_K - 273.15, R_abs, K_shortwave_abs, L_longwave_abs

    template = np.asarray(K_reflected_abs, dtype=float)
    contributions = {
        "sw_direct_absorbed_Wm2": np.asarray(K_direct_abs, dtype=float),
        "sw_diffuse_sky_absorbed_Wm2": np.asarray(K_diffuse_abs, dtype=float),
        "sw_reflected_total_absorbed_Wm2": template,
        "lw_sky_absorbed_Wm2": np.asarray(L_sky_abs, dtype=float),
        "lw_surface_total_absorbed_Wm2": np.asarray(L_surface_abs, dtype=float),
        "sw_total_absorbed_Wm2": np.asarray(K_shortwave_abs, dtype=float),
        "lw_total_absorbed_Wm2": np.asarray(L_longwave_abs, dtype=float),
        "total_absorbed_radiant_flux_Wm2": np.asarray(R_abs, dtype=float),
    }

    # Material attribution of reflected shortwave. The GROUND part is split by
    # the same route-visible ground-facet albedo weights that produced
    # ``local_ground_albedo``; the WALL/ROOF part is attributed directly from
    # the 05a view weights of those facets. Together they close exactly onto
    # sw_reflected_total_absorbed_Wm2.
    ground_template = np.asarray(K_ground_reflected_abs, dtype=float)
    sw_fractions = reflected_source_fractions or {
        "sw_reflected_generic_ground_absorbed_Wm2": np.ones_like(ground_template)
    }
    for key in SW_SOURCE_COLUMNS:
        fraction = np.asarray(sw_fractions.get(key, np.zeros_like(ground_template)),
                              dtype=float)
        contributions[key] = ground_template * fraction
    if wall_reflected_parts is not None:
        absorptivity = args.person_sw_absorptivity
        contributions["sw_reflected_building_wall_absorbed_Wm2"] = (
            contributions.get("sw_reflected_building_wall_absorbed_Wm2", 0.0)
            + absorptivity * np.asarray(wall_reflected_parts["wall"], dtype=float))
        contributions["sw_reflected_roof_absorbed_Wm2"] = (
            contributions.get("sw_reflected_roof_absorbed_Wm2", 0.0)
            + absorptivity * np.asarray(wall_reflected_parts["roof"], dtype=float))

    if L_surround_components:
        for key in LW_SOURCE_COLUMNS:
            incident = np.asarray(
                L_surround_components.get(key, np.zeros_like(template)), dtype=float)
            contributions[key] = (
                args.person_emissivity * (1.0 - sky_frac_lw) * incident)
    else:
        # The legacy single-surround model has no hit-source identity. Retain
        # exact conservation and label that unresolved energy explicitly.
        for key in LW_SOURCE_COLUMNS:
            contributions[key] = np.zeros_like(template)
        contributions["lw_other_surface_absorbed_Wm2"] = np.asarray(
            L_surface_abs, dtype=float)
    return tmrt_K - 273.15, R_abs, K_shortwave_abs, L_longwave_abs, contributions


def sensor_radiometer_quantities(dni, dhi, elevation_deg, tau_direct,
                                 svf_planar, L_sky, L_surround,
                                 ground_albedo, ground_emitted_Wm2,
                                 ground_emissivity, surround_sw_radiance,
                                 args, longwave_down_override=None,
                                 reflected_down_override=None,
                                 shortwave_up_override=None):
    """Emulate a four-component net radiometer at the sensor height.

    Returns the four channels such an instrument reports, on ITS OWN angular
    weighting -- a horizontal, cosine-weighted upward and downward pair --
    not the human-body weighting used for MRT:

        shortwave_down = beam on the horizontal + sky diffuse + surround
                         reflected sunlight arriving from above the horizon
        shortwave_up   = footprint ground albedo * shortwave_down
        longwave_down  = sky share of the upper hemisphere + the surround
                         radiosity filling the rest of it
        longwave_up    = footprint ground emission + reflected downwelling
                         longwave

    The two UPWELLING channels use the instrument's own downward footprint
    (``FacetLongwave.build_sensor_ground_footprint``), which is a few metres
    across at a 1 m sensor height -- not the body's cylinder view of the ground
    out to the culling distance.

    Documented approximations (they are why this is a diagnostic, not a
    second authoritative product):

    * The sky/surface split of each hemisphere uses the PLANAR sky-view
      factor, which is the correct cosine weighting for a horizontal sensor.
      The radiance filling the non-sky part is taken from the cylinder-
      weighted facet mean already computed for the longwave surround, i.e.
      the surround's mean radiosity is reused while only the SPLIT is planar.
      A fully rigorous version would need a second, planar-weighted view
      matrix from 05a.
    * With ``--sensor-emulation facet`` (the default when 05a wrote the
      up-facing view) the two approximations above are removed: the
      downwelling longwave and the surround-reflected shortwave come from the
      up-facing view (``*_override`` arguments), and the upwelling shortwave
      is the footprint average of each ground facet's own reflected sunlight.
    * The ray tracing is performed once, at ``--z-height``. When
      ``--sensor-height-m`` differs, the receptor's shading and sky view are
      reused; over the ~0.1 m offsets involved this is negligible in the open
      and small beside a facade, and both heights are recorded in metadata.
    """
    sin_el = np.sin(np.deg2rad(np.maximum(elevation_deg, 0.0)))
    svf_planar = np.asarray(svf_planar, dtype=float)
    beam_horizontal = np.asarray(tau_direct, dtype=float) * float(dni) * sin_el
    diffuse_sky = svf_planar * float(dhi)
    if surround_sw_radiance is None:
        surround_sw = np.zeros_like(svf_planar)
    else:
        surround_sw = (1.0 - svf_planar) * np.asarray(surround_sw_radiance,
                                                      dtype=float)
    if reflected_down_override is not None:
        surround_sw = np.asarray(reflected_down_override, dtype=float)
    shortwave_down = beam_horizontal + diffuse_sky + surround_sw

    albedo = (np.full_like(svf_planar, float(args.ground_albedo or 0.18))
              if ground_albedo is None
              else np.asarray(ground_albedo, dtype=float))
    shortwave_up = albedo * shortwave_down
    if shortwave_up_override is not None:
        override = np.asarray(shortwave_up_override, dtype=float)
        shortwave_up = np.where(np.isfinite(override), override, shortwave_up)

    longwave_down = (svf_planar * np.asarray(L_sky, dtype=float)
                     + (1.0 - svf_planar) * np.asarray(L_surround, dtype=float))
    if longwave_down_override is not None:
        longwave_down = np.asarray(longwave_down_override, dtype=float)

    if ground_emitted_Wm2 is None:
        # No resolved ground footprint: the surround radiosity is the best
        # available estimate of what the downward sensor would see.
        longwave_up = np.asarray(L_surround, dtype=float) * np.ones_like(svf_planar)
    else:
        eps_g = (np.full_like(svf_planar, 0.95) if ground_emissivity is None
                 else np.asarray(ground_emissivity, dtype=float))
        # Already footprint-weighted RADIOSITY (eps*sigma*T^4 averaged over the
        # instrument's own downward kernel), not a temperature to be raised to
        # the fourth power here -- see FacetLongwave.sensor_ground_emitted.
        emitted = np.asarray(ground_emitted_Wm2, dtype=float)
        fallback = np.asarray(L_surround, dtype=float) * np.ones_like(svf_planar)
        eps_g = np.where(np.isfinite(eps_g), eps_g, 0.95)
        longwave_up = np.where(np.isfinite(emitted),
                               emitted + (1.0 - eps_g) * longwave_down,
                               fallback)
    return {
        "sensor_shortwave_down_Wm2": shortwave_down,
        "sensor_shortwave_up_Wm2": shortwave_up,
        "sensor_longwave_down_Wm2": longwave_down,
        "sensor_longwave_up_Wm2": longwave_up,
    }


def globe_radiation_args(args, spec):
    """An args view that makes ``estimate_mrt_from_radiation`` describe a SPHERE.

    Reusing that function rather than reimplementing the radiation load is
    deliberate: the globe must see exactly the same traced shading, sky view,
    facet longwave surround and wall-reflected shortwave as the pedestrian does,
    or any model-versus-instrument difference would be contaminated by the two
    receptors having been given different scenes. Only three things change --
    the beam projected-area factor becomes the sphere's constant 0.25, and the
    absorptivity/emissivity become the globe's paint rather than human skin and
    clothing. Diffuse, reflected and longwave angular factors are ~0.5/0.5/
    isotropic for a sphere and a standing cylinder alike (VDI 3787), so they are
    correctly left untouched.

    This costs no extra ray tracing: it is a second pass over already-traced
    per-point quantities.
    """
    shim = copy.copy(args)
    shim.projected_area_model = "sphere"
    shim.f_projected_direct = sphere_projected_area_factor()
    shim.person_sw_absorptivity = spec.sw_absorptivity
    shim.person_emissivity = spec.emissivity
    return shim


class FacetLongwave:
    """Assembles the per-point longwave surround from ray-traced facet
    radiosities and surface temperatures (outputs of 05a + 05b).

    Per timestep it computes, at each traced (coarse) route point:

        L_surround = [ sum_f W_pf * J_f(t)                          (facets)
                       + w_veg * J_veg(t)                            (canopy)
                       + w_def * J_environment(t) ]                 (culled)
                     / (1 - w_sky)

    where grey-surface radiosity from 05b is
    ``J = eps*sigma*T^4 + (1-eps)*G``. Only stage-05a first-hit facets visible
    from the route within its distance cap are included. Older thermal folders
    without an authoritative grey-radiosity manifest use the emitted-only
    legacy calculation for backward compatibility."""

    def __init__(self, thermal_dir, n_points, n_times, args):
        import scipy.sparse as sp
        d = Path(thermal_dir)
        self.W = sp.load_npz(d / "lw_view_matrix.npz")
        pw = np.load(d / "lw_point_weights.npz")
        # Up-facing radiometer view from 05a: the same rays re-weighted for a
        # horizontal cosine receiver over the upper hemisphere. Optional, so
        # thermal folders written before it existed still load.
        self.W_up = None
        up_matrix = d / "sensor_up_view_matrix.npz"
        up_weights = d / "sensor_up_point_weights.npz"
        if up_matrix.is_file() and up_weights.is_file():
            self.W_up = sp.load_npz(up_matrix)
            uw = np.load(up_weights)
            self.w_sky_up = uw["w_sky"]
            self.w_veg_up = uw["w_veg"]
            if self.W_up.shape != self.W.shape:
                raise ValueError("up-facing sensor view does not match the "
                                 "05a view matrix")
        self.w_sky = pw["w_sky"]
        self.w_veg = pw["w_veg"]
        self.w_def = pw["w_default"]
        self.point_map = np.load(d / "point_map.npy")
        self.facet_T = np.load(d / "facet_T_matrix_K.npy")
        self.facet_eps = np.load(d / "facet_eps.npy")
        self.local_ground_albedo = None
        self.reflected_source_fractions = None
        facet_albedo_path = d / "facet_albedo.npy"
        facets_path = d / "facets.npz"
        self.facet_material_name = np.full(self.W.shape[1], "other_surface", dtype="U32")
        facet_class = None
        if facets_path.is_file():
            facet_meta = np.load(facets_path)
            facet_class = facet_meta["cls"]
            # Diagnostic surface-class temperature perturbation. Applied to the
            # SOLVED facet field before any radiosity or view weighting, so the
            # offset reaches the globe (sphere-weighted), the emulated
            # radiometer (horizontal cosine-weighted) and the body
            # (cylinder-weighted) through their own existing weights. Adding an
            # offset to a receptor's absorbed flux instead would convert between
            # angular conventions by a fixed factor, which is exactly what this
            # framework refuses to do elsewhere.
            self._class_temperature_offsets = [
                (c, v) for c, v in (
                    (0, float(getattr(args, "ground_temperature_offset_K", 0.0) or 0.0)),
                    (1, float(getattr(args, "wall_temperature_offset_K", 0.0) or 0.0)),
                    (2, float(getattr(args, "roof_temperature_offset_K", 0.0) or 0.0)))
                if v != 0.0]
            if "material_name" in facet_meta.files:
                self.facet_material_name = facet_meta["material_name"].astype(str)
            else:
                self.facet_material_name = np.where(
                    facet_class == 0, "ground",
                    np.where(facet_class == 2, "roof", "wall"))
        if facet_albedo_path.is_file() and facet_class is not None:
            facet_albedo = np.load(facet_albedo_path).astype(float)
            if facet_albedo.shape != (self.W.shape[1],):
                raise ValueError("facet albedo count does not match view matrix")
            ground_mask = facet_class == 0
            if ground_mask.any():
                ground_weight = np.asarray(
                    self.W[:, ground_mask].sum(axis=1)).ravel()
                weighted_albedo = np.asarray(
                    self.W[:, ground_mask] @ facet_albedo[ground_mask]).ravel()
                coarse_albedo = np.where(
                    ground_weight > 1e-12,
                    weighted_albedo / np.maximum(ground_weight, 1e-12),
                    getattr(args, "ground_albedo", 0.18))
                self.local_ground_albedo = coarse_albedo[self.point_map]
                source_weighted_albedo = {}
                for material in sorted(set(self.facet_material_name[ground_mask])):
                    source = canonical_sw_source(material)
                    material_mask = ground_mask & (self.facet_material_name == material)
                    numerator = np.asarray(
                        self.W[:, material_mask] @ facet_albedo[material_mask]).ravel()
                    source_weighted_albedo[source] = (
                        source_weighted_albedo.get(source, 0.0) + numerator)
                fractions = {}
                nonzero = weighted_albedo > 1e-12
                for source, numerator in source_weighted_albedo.items():
                    fraction = np.zeros_like(weighted_albedo)
                    fraction[nonzero] = numerator[nonzero] / weighted_albedo[nonzero]
                    fractions[source] = fraction[self.point_map]
                # A point with no route-visible classified ground uses the
                # same scalar fallback albedo as the existing model. Attribute
                # that unresolved reflection to generic ground for closure.
                generic = "sw_reflected_generic_ground_absorbed_Wm2"
                fractions.setdefault(generic, np.zeros(n_points, dtype=float))
                fractions[generic][~nonzero[self.point_map]] = 1.0
                self.reflected_source_fractions = fractions
        # ---- shortwave reflection off walls/roofs, and the local ground state
        # a downward-facing radiometer would see. Both reuse the SAME 05a
        # directional view weights as the longwave surround, so a facade that
        # already contributes longwave to this pedestrian now also contributes
        # its reflected sunlight, with no additional ray tracing.
        self.wall_reflect_mask = None
        self.facet_albedo = None
        self.facet_cos_theta = None
        self.facet_tau_dir = None
        self.facet_f_sky = None
        # Downward-radiometer footprint state. These are the SENSOR's view of
        # the ground and are kept strictly apart from local_ground_albedo, which
        # is the BODY's cylinder-weighted view and feeds the pedestrian's
        # reflected shortwave. Two receptors, two weightings; conflating them is
        # exactly the mistake this footprint exists to undo.
        self._footprint = None
        self._footprint_total = None
        self._footprint_ground_index = None
        self.sensor_ground_albedo = None
        self.sensor_ground_emissivity = None
        self.sensor_footprint_coverage = 0.0
        self.sensor_footprint_radius_m = None
        self._facet_centroid = None
        self._facet_area = None
        self._facet_normal_ground = None
        if facets_path.is_file():
            facet_meta = np.load(facets_path)
            if {"centroid", "area", "normal"} <= set(facet_meta.files):
                self._facet_centroid = facet_meta["centroid"].astype(float)
                self._facet_area = facet_meta["area"].astype(float)
                ground_normals = facet_meta["normal"].astype(float)
                if facet_class is not None:
                    # Stored as-wound; build_sensor_ground_footprint orients
                    # them upward at the point of use.
                    self._facet_normal_ground = ground_normals[facet_class == 0]
        if facet_class is not None and facet_albedo_path.is_file():
            albedo_all = np.load(facet_albedo_path).astype(float)
            self.facet_albedo = albedo_all
            self.wall_reflect_mask = facet_class != 0        # walls + roofs
            tau_path = d / "tau_dir_facet.npy"
            fsky_path = d / "f_sky_facet.npy"
            if tau_path.is_file() and fsky_path.is_file():
                self.facet_tau_dir = np.load(tau_path, mmap_mode="r")
                self.facet_f_sky = np.load(fsky_path).astype(float)
                if self.facet_tau_dir.shape != (n_times, self.W.shape[1]):
                    raise ValueError(
                        "tau_dir_facet does not match this run's times/facets; "
                        "re-run 05b for this thermal folder")
                normals = facet_meta["normal"].astype(float)
                self.facet_normal = normals
            # Local ground state beneath the receptor, view-weighted over the
            # same route-visible ground facets that set local_ground_albedo.
            ground_mask = facet_class == 0
            if ground_mask.any():
                gw = np.asarray(self.W[:, ground_mask].sum(axis=1)).ravel()
                self._ground_mask = ground_mask
                self._ground_weight = gw
                eps_ground = np.load(d / "facet_eps.npy").astype(float)
                weighted_eps = np.asarray(
                    self.W[:, ground_mask] @ eps_ground[ground_mask]).ravel()
                coarse_eps = np.where(gw > 1e-12,
                                      weighted_eps / np.maximum(gw, 1e-12), 0.95)
                self.local_ground_emissivity = coarse_eps[self.point_map]

        self.facet_J = None
        self.environment_J = None
        radiosity_model = "legacy"
        manifest_path = d / MATERIALS_MANIFEST
        if manifest_path.is_file():
            with open(manifest_path) as f:
                manifest = json.load(f)
            radiosity_model = manifest.get(
                "_longwave_radiosity", {}).get("model", "legacy")
        if radiosity_model == "grey":
            radiosity_path = d / RADIOSITY_MATRIX
            environment_path = d / RADIOSITY_ENVIRONMENT
            if not radiosity_path.is_file() or not environment_path.is_file():
                raise FileNotFoundError(
                    "05b manifest declares grey radiosity but its radiosity "
                    "outputs are missing; rerun 05b for this thermal folder")
            self.facet_J = np.load(radiosity_path)
            self.environment_J = np.load(environment_path)
        self.radiosity_model = radiosity_model
        self.args = args
        # Apply the diagnostic surface-class offset now that BOTH the
        # temperature field and the grey radiosity (if 05b wrote one) are
        # loaded. surround_at consumes facet_J when it exists, so perturbing
        # facet_T alone would be silently inert -- the offset must move the
        # emitted term of the radiosity as well.
        #
        #   J = eps*sigma*T^4 + (1-eps)*G,  so  dJ = eps*sigma*((T+dT)^4 - T^4)
        #
        # holding the solved irradiance G fixed. That is a first-order
        # treatment: it neglects the re-reflection of the perturbation between
        # facets, which for the offsets used here is a small correction to an
        # already diagnostic experiment.
        for cls_id, delta in getattr(self, "_class_temperature_offsets", []):
            mask = facet_class == cls_id if facet_class is not None else None
            if mask is None or not mask.any():
                continue
            T_old = self.facet_T[:, mask].astype(float)
            T_new = T_old + delta
            if self.facet_J is not None:
                self.facet_J = np.array(self.facet_J, dtype=float, copy=True)
                self.facet_J[:, mask] += (self.facet_eps[mask]
                                          * SIGMA * (T_new ** 4 - T_old ** 4))
            self.facet_T = np.array(self.facet_T, dtype=float, copy=True)
            self.facet_T[:, mask] = T_new
            print(f"  [diagnostic] facet class {cls_id} temperature offset "
                  f"{delta:+.2f} K applied to {int(mask.sum())} facets "
                  f"(radiosity model: {radiosity_model})", flush=True)
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
        if self.facet_J is not None:
            if self.facet_J.shape != self.facet_T.shape:
                raise ValueError("grey-radiosity matrix must match facet temperatures")
            if self.environment_J.shape != (n_times,):
                raise ValueError("radiosity environment must have one value per time step")
            if (not np.isfinite(self.facet_J).all()
                    or not np.isfinite(self.environment_J).all()):
                raise ValueError("grey-radiosity outputs contain non-finite values")
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
              f"{self.sky_frac.mean():.3f}); radiosity={self.radiosity_model}")
        if self.local_ground_albedo is not None:
            print(f"  Local route-visible ground albedo: "
                  f"{self.local_ground_albedo.min():.3f}.."
                  f"{self.local_ground_albedo.max():.3f}")

    def facet_incident_shortwave(self, it, dni, dhi, sun_vec):
        """Global shortwave incident on every route-visible facet, W m^-2.

        Same construction 05b uses to heat those facets: attenuated direct
        beam on the facet's own tilt, plus its sky-view share of the diffuse.
        The second-bounce (environment-reflected) term 05b adds is deliberately
        omitted here, so this stays a single-bounce reflection to the
        pedestrian rather than an unbounded inter-reflection series.
        """
        if self.facet_tau_dir is None:
            return None
        incident = self.facet_f_sky * float(dhi)
        if sun_vec is not None and dni > 0.0:
            cos_theta = np.clip(self.facet_normal @ np.asarray(sun_vec, float),
                                0.0, None)
            incident = incident + (np.asarray(self.facet_tau_dir[it], dtype=float)
                                   * float(dni) * cos_theta)
        return incident

    def wall_reflected_shortwave(self, it, dni, dhi, sun_vec):
        """Shortwave reflected off walls/roofs onto each route point, W m^-2.

        Mirrors the longwave surround exactly: the 05a view weight of a facet
        multiplies what that facet sends toward the pedestrian. For longwave
        that is its radiosity; here it is its reflected sunlight,
        ``albedo * incident``. Returns ``(total, {'wall': .., 'roof': ..})``
        as incident radiance at the body (absorptivity is applied by the
        caller, as for the ground term).
        """
        incident = self.facet_incident_shortwave(it, dni, dhi, sun_vec)
        if incident is None or self.wall_reflect_mask is None:
            return None, None
        reflected = self.facet_albedo * incident
        parts = {}
        totals = np.zeros(len(self.point_map), dtype=float)
        for label, mask in (("wall", self.wall_reflect_mask
                             & (self.facet_material_name != "roof")),
                            ("roof", self.wall_reflect_mask
                             & (self.facet_material_name == "roof"))):
            if not mask.any():
                parts[label] = np.zeros(len(self.point_map), dtype=float)
                continue
            coarse = np.asarray(self.W[:, mask] @ reflected[mask]).ravel()
            fine = coarse[self.point_map]
            parts[label] = fine
            totals = totals + fine
        return totals, parts

    def build_sensor_ground_footprint(self, path_xyz, sensor_height_m,
                                      receptor_height_m,
                                      maximum_radius_m=20.0):
        """Build the view a DOWNWARD-FACING radiometer actually has.

        The previous implementation reused the standing-cylinder view matrix
        from 05a, weighted out to the 300 m culling distance. That is the wrong
        instrument. A cylinder's longwave weighting emphasises the horizon,
        so the "ground below the receptor" it produced was in truth a wide-area
        average dominated by DISTANT ground -- it would report sunlit plaza a
        hundred metres away as if it were under the sensor's feet.

        A downward pyrgeometer at height h sees a compact footprint. For a
        ground element of area dA at slant distance d, the contribution is
        ``L * cos(theta_sensor) * cos(theta_ground) * dA / d^2``, and over flat
        ground that reduces to ``h^2 / (r^2 + h^2)^2 dA``: 50% of the signal
        comes from within r = h, 90% from within 3h, 99% from within 10h. At the
        1.0 m sensor height that is a few metres across, not a few hundred.

        The full 3-D form is used here rather than the flat-ground reduction, so
        sloping terrain and tilted facets are handled correctly:

            w_i = max(cos_s, 0) * max(cos_g, 0) * A_i / d_i^2

        Documented approximations:

        * Each facet is sampled at its centroid. Ground facets along a route are
          small relative to the footprint, and the kernel is bounded at r = 0,
          so this cannot blow up -- but a very large facet close to the sensor
          has its weight placed at its centre rather than spread.
        * No occlusion test. Within the few metres that carry the weight, a
          receptor standing on the ground has unobstructed sight of it; the
          weight that leaks past a wall at 10 m+ is under 1%.

        Built only when the sensor channels are requested, since it is pure
        overhead for a run that does not emit them.
        """
        import scipy.sparse as sp
        from scipy.spatial import cKDTree

        if getattr(self, "_ground_mask", None) is None:
            return False
        ground_index = np.flatnonzero(self._ground_mask)
        if ground_index.size == 0 or self._facet_centroid is None:
            return False
        centroids = self._facet_centroid[ground_index]
        areas = self._facet_area[ground_index]
        # Orient every ground normal upward at the point of use. A terrain mesh
        # can carry either winding, and a downward-wound facet would fail the
        # cos_ground > 0 test and silently drop out of the footprint.
        normals = np.array(self._facet_normal_ground, dtype=float, copy=True)
        normals[normals[:, 2] < 0.0] *= -1.0

        sensor_xyz = np.asarray(path_xyz, dtype=float).copy()
        # path_xyz sits at the pedestrian receptor height; the instrument sits
        # at its own height above the same local ground.
        sensor_xyz[:, 2] += float(sensor_height_m) - float(receptor_height_m)

        tree = cKDTree(centroids[:, :2])
        neighbourhoods = tree.query_ball_point(sensor_xyz[:, :2],
                                               float(maximum_radius_m))
        rows, cols, data = [], [], []
        for point, (position, neighbours) in enumerate(
                zip(sensor_xyz, neighbourhoods)):
            if not neighbours:
                continue
            neighbours = np.asarray(neighbours, dtype=int)
            offset = position - centroids[neighbours]          # facet -> sensor
            distance_sq = np.einsum("ij,ij->i", offset, offset)
            distance_sq = np.maximum(distance_sq, 1e-6)
            distance = np.sqrt(distance_sq)
            cos_sensor = offset[:, 2] / distance               # facet below = +
            cos_ground = np.einsum("ij,ij->i",
                                   normals[neighbours], offset) / distance
            weight = (np.maximum(cos_sensor, 0.0) * np.maximum(cos_ground, 0.0)
                      * areas[neighbours] / distance_sq)
            keep = weight > 0.0
            if not keep.any():
                continue
            rows.append(np.full(keep.sum(), point, dtype=np.int32))
            cols.append(neighbours[keep].astype(np.int32))
            data.append(weight[keep])
        if not rows:
            return False
        footprint = sp.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(sensor_xyz.shape[0], ground_index.size))
        total = np.asarray(footprint.sum(axis=1)).ravel()
        self._footprint = footprint
        self._footprint_total = total
        self._footprint_ground_index = ground_index
        covered = total > 1e-12
        # Where no ground facet is resolved the SHORTWAVE-up channel still has
        # to produce a finite number, so it falls back to the body's view of the
        # ground albedo. Only the longwave-up channel is left NaN, because it has
        # its own documented fallback to the surround radiosity.
        albedo_fallback = (self.local_ground_albedo
                           if self.local_ground_albedo is not None
                           else np.full(total.shape,
                                        getattr(self.args, "ground_albedo", 0.18)))
        self.sensor_ground_albedo = np.where(
            covered,
            np.asarray(footprint @ self.facet_albedo[ground_index]).ravel()
            / np.maximum(total, 1e-12),
            albedo_fallback)
        self.sensor_ground_emissivity = np.where(
            covered,
            np.asarray(footprint @ self.facet_eps[ground_index]).ravel()
            / np.maximum(total, 1e-12),
            0.95)
        self.sensor_footprint_coverage = float(np.mean(covered))
        # Radius holding half the weight -- the honest statement of what the
        # emulated instrument is averaging over.
        self.sensor_footprint_radius_m = float(sensor_height_m)
        return True

    def sensor_ground_emitted(self, it):
        """Footprint-weighted EMITTED longwave from the ground, W/m2.

        Note this averages radiosity, not temperature. The previous code
        averaged T over its view and only then raised the mean to the fourth
        power. A radiometer integrates radiance: because ``sigma*T^4`` is
        convex, ``<T>^4`` sits below ``<T^4>``, so temperature-averaging
        understated a spatially varying surface. Small next to the view error
        it accompanied, but wrong in its own right.
        """
        if getattr(self, "_footprint", None) is None:
            return None
        index = self._footprint_ground_index
        temps = np.asarray(self.facet_T[it], dtype=float)[index]
        emitted = self.facet_eps[index] * SIGMA * temps ** 4
        weighted = np.asarray(self._footprint @ emitted).ravel()
        return np.where(self._footprint_total > 1e-12,
                        weighted / np.maximum(self._footprint_total, 1e-12),
                        np.nan)

    def _radiosities(self, it, air_temp_C, elevation_deg, sky_longwave_Wm2):
        """(facet, vegetation, environment) radiosity at one time step, on
        the same convention ``surround_at`` uses."""
        a = self.args
        air_K = air_temp_C + 273.15
        if self.facet_J is not None:
            J_facet = self.facet_J[it].astype(float)
            J_environment = float(self.environment_J[it])
            G_vegetation = 0.5 * (sky_longwave_Wm2 + J_environment)
            J_vegetation = (a.vegetation_emissivity * SIGMA * air_K ** 4
                            + (1.0 - a.vegetation_emissivity) * G_vegetation)
        else:
            sin_el = np.sin(np.deg2rad(max(elevation_deg, 0.0)))
            legacy_K = air_K + a.surface_temp_offset_day_c * max(sin_el, 0.0)
            J_facet = self.facet_eps * SIGMA * self.facet_T[it].astype(float) ** 4
            J_vegetation = a.vegetation_emissivity * SIGMA * air_K ** 4
            J_environment = a.surrounding_emissivity * SIGMA * legacy_K ** 4
        return J_facet, J_vegetation, J_environment

    @property
    def has_sensor_up_view(self):
        return self.W_up is not None

    def sensor_downwelling_at(self, it, air_temp_C, elevation_deg,
                              sky_longwave_Wm2, facet_incident_sw=None):
        """What an UP-FACING horizontal radiometer receives, W m^-2.

        Returns ``(longwave_down, reflected_shortwave_down)`` per full route
        point. Both use the 05a up-facing view, so the part of the upper
        hemisphere that is not sky is filled by the surfaces actually above
        the horizon -- walls, roofs, canopy -- at their own radiosity, instead
        of by the body's cylinder-weighted mean, which includes the (hot by
        day) ground below the horizon that this sensor cannot see.

        Vegetation reflection of shortwave is not represented (the canopy
        underside a sensor sees from below is largely shaded).
        """
        J_facet, J_veg, _ = self._radiosities(
            it, air_temp_C, elevation_deg, sky_longwave_Wm2)
        lw = (self.w_sky_up * float(sky_longwave_Wm2)
              + self.W_up @ J_facet + self.w_veg_up * J_veg)
        if facet_incident_sw is None or self.facet_albedo is None:
            sw = np.zeros_like(lw)
        else:
            sw = self.W_up @ (self.facet_albedo * facet_incident_sw)
        return (np.asarray(lw).ravel()[self.point_map],
                np.asarray(sw).ravel()[self.point_map])

    def sensor_ground_reflected(self, facet_incident_sw):
        """Footprint-weighted shortwave reflected by the ground, W m^-2.

        Each ground facet reflects ``albedo * (its OWN incident shortwave)``:
        its own ray-traced beam transmission on its own tilt plus its own sky
        share of the diffuse. The earlier emulation multiplied the footprint
        albedo by the SENSOR's downwelling irradiance, which assigns the
        sensor's shade state to ground up to a few metres away -- wrong
        whenever a shadow edge lies inside the footprint.
        """
        if (getattr(self, "_footprint", None) is None
                or facet_incident_sw is None or self.facet_albedo is None):
            return None
        index = self._footprint_ground_index
        reflected = self.facet_albedo[index] * np.asarray(
            facet_incident_sw, dtype=float)[index]
        weighted = np.asarray(self._footprint @ reflected).ravel()
        return np.where(self._footprint_total > 1e-12,
                        weighted / np.maximum(self._footprint_total, 1e-12),
                        np.nan)

    def surround_at(self, it, air_temp_C, elevation_deg,
                    sky_longwave_Wm2=None, return_components=False):
        a = self.args
        air_K = air_temp_C + 273.15
        sin_el = np.sin(np.deg2rad(max(elevation_deg, 0.0)))
        legacy_K = air_K + a.surface_temp_offset_day_c * max(sin_el, 0.0)
        if self.facet_J is not None:
            if sky_longwave_Wm2 is None or not np.isfinite(sky_longwave_Wm2):
                raise ValueError("grey radiosity requires finite sky longwave")
            J_facet = self.facet_J[it].astype(float)
            J_environment = float(self.environment_J[it])
            # Leaves have high emissivity and see both sky and the local route
            # enclosure. Their 2% reflected part is retained for consistency.
            G_vegetation = 0.5 * (sky_longwave_Wm2 + J_environment)
            J_vegetation = (
                a.vegetation_emissivity * SIGMA * air_K ** 4
                + (1.0 - a.vegetation_emissivity) * G_vegetation)
            facet_radiosity = J_facet
            num = (self.W @ facet_radiosity
                   + self.w_veg * J_vegetation
                   + self.w_def * J_environment)
            legacy_L = J_environment
        else:
            facet_radiosity = self.facet_eps * SIGMA * self.facet_T[it].astype(float) ** 4
            J_vegetation = a.vegetation_emissivity * SIGMA * air_K ** 4
            J_environment = a.surrounding_emissivity * SIGMA * legacy_K ** 4
            num = (self.W @ facet_radiosity
                   + self.w_veg * a.vegetation_emissivity * SIGMA * air_K ** 4
                   + self.w_def * a.surrounding_emissivity * SIGMA * legacy_K ** 4)
            legacy_L = a.surrounding_emissivity * SIGMA * legacy_K ** 4
        L_coarse = np.where(self.w_surf > 1e-6,
                            num / np.maximum(self.w_surf, 1e-6), legacy_L)
        full = L_coarse[self.point_map]
        if not return_components:
            return full

        source_numerators = {}
        for material in sorted(set(self.facet_material_name)):
            source = canonical_lw_source(material)
            mask = self.facet_material_name == material
            numerator = np.asarray(self.W[:, mask] @ facet_radiosity[mask]).ravel()
            source_numerators[source] = source_numerators.get(source, 0.0) + numerator
        source_numerators["lw_tree_canopy_absorbed_Wm2"] = (
            source_numerators.get("lw_tree_canopy_absorbed_Wm2", 0.0)
            + self.w_veg * J_vegetation)
        source_numerators["lw_other_surface_absorbed_Wm2"] = (
            source_numerators.get("lw_other_surface_absorbed_Wm2", 0.0)
            + self.w_def * J_environment)
        components = {}
        visible = self.w_surf > 1e-6
        for source, numerator in source_numerators.items():
            normalized = np.zeros_like(self.w_surf, dtype=float)
            normalized[visible] = np.asarray(numerator)[visible] / self.w_surf[visible]
            components[source] = normalized[self.point_map]
        # Only relevant to non-fullsphere legacy configurations: if the 05a
        # view contains no resolved surface but stage 05 still assigns a
        # surround fraction, retain the legacy surround in Other.
        missing = ~visible[self.point_map]
        if missing.any():
            components.setdefault("lw_other_surface_absorbed_Wm2",
                                  np.zeros(len(self.point_map), dtype=float))
            components["lw_other_surface_absorbed_Wm2"][missing] = full[missing]
        return full, components


def main():
    args = parse_args()
    contribution_config = load_contribution_config(args.radiant_flux_config)
    record_contributions = bool(contribution_config["enabled"])
    if record_contributions:
        print("Absorbed radiant-flux contribution recording: enabled")
    else:
        print("Absorbed radiant-flux contribution recording: disabled")

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

    path_xyz_file = out_dir / "path_xyz.npy"
    np.save(path_xyz_file, path_xyz)
    np.save(out_dir / "path_segment_id.npy", segment_id)

    print("\n" + "=" * 70)
    print("Computing static effective sky-view factor...")
    sky_directions, sky_w_planar, sky_w_cyl = make_sky_directions(
        args.sky_n_azimuth, args.sky_n_elevation)
    print(f"  Sky directions: {len(sky_directions)}")

    svf_metadata = None
    cached_svf = None
    if args.svf_cache:
        svf_metadata = build_svf_cache_metadata(args, path_xyz, path_xyz_file)
        if args.force_svf:
            print("  --force-svf set; bypassing any existing SVF cache.")
        else:
            cached_svf = load_cached_svf(args.svf_cache, svf_metadata, n_points)
    if cached_svf is None:
        svf_building_only, svf_planar, svf_standing = compute_effective_svf_batched(
            path_xyz, sky_directions, sky_w_planar, sky_w_cyl,
            building_intersector, vegetation_intersector,
            args.k_lad_diffuse, args.svf_batch_size,
        )
        if args.svf_cache:
            write_svf_cache(
                args.svf_cache, svf_metadata,
                (svf_building_only, svf_planar, svf_standing),
            )
    else:
        svf_building_only, svf_planar, svf_standing = cached_svf
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
    print("Solar position and irradiance forcing...")
    times = pd.date_range(
        start=f"{args.date} 00:00", end=f"{args.date} 23:50",
        freq=f"{args.dt_min}min", tz=args.timezone,
    )
    location = pvlib.location.Location(latitude=args.latitude, longitude=args.longitude,
                                        tz=args.timezone)
    solar = pvlib.solarposition.get_solarposition(times, args.latitude, args.longitude)
    elev = solar["apparent_elevation"].values
    azim = solar["azimuth"].values
    clearsky = location.get_clearsky(times, model="ineichen")
    if args.radiation_csv:
        dni, dhi, ghi = load_radiation_csv(args.radiation_csv, times)
        radiation_source = str(Path(args.radiation_csv).resolve())
        cloud_fraction_time = np.full(len(times), args.cloud_cover_fraction)
        lwin_time = np.full(len(times), np.nan)
        forcing_metadata = {
            "mode": "legacy_radiation_csv",
            "source_file": radiation_source,
            "configured_cloud_fraction": args.cloud_cover_fraction,
            "note": "DNI/DHI/GHI from --radiation-csv; sky longwave remains parameterized.",
        }
        print(f"  Irradiance forcing: measured/reference CSV {radiation_source}")
    else:
        resolved_forcing = resolve_radiation_forcing(
            args.radiation_forcing_config, times,
            clearsky["dni"].values, clearsky["dhi"].values,
            clearsky["ghi"].values, elev, args.cloud_cover_fraction,
        )
        dni = resolved_forcing.dni_wm2
        dhi = resolved_forcing.dhi_wm2
        ghi = resolved_forcing.ghi_wm2
        cloud_fraction_time = resolved_forcing.cloud_fraction
        lwin_time = resolved_forcing.lwin_wm2
        radiation_source = resolved_forcing.source
        forcing_metadata = resolved_forcing.metadata
        print(f"  Irradiance forcing: {radiation_source}")
        if "inferred_case_cloud_fraction" in forcing_metadata:
            print("  Mobile upper-envelope cloud estimate: "
                  f"{forcing_metadata['inferred_case_cloud_fraction']:.3f} "
                  f"from {forcing_metadata['n_usable_daytime_samples']} usable samples")
            for warning in forcing_metadata.get("warnings", []):
                print(f"  WARNING: {warning}")
    (out_dir / "radiation_forcing_metadata.json").write_text(
        json.dumps(forcing_metadata, indent=2) + "\n", encoding="utf-8")
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

    # This is intentionally written before the expensive direct-sun/MRT loop:
    # 05b needs only this forcing table, and --prep-only is the preparation
    # half of the merged facet-thermal stage. The column order and formatting
    # are unchanged from a normal full run.
    times_df = pd.DataFrame({
        "time": times, "azimuth_deg": azim, "elevation_deg": elev,
        "DNI_Wm2": dni, "DHI_Wm2": dhi, "GHI_Wm2": ghi, "air_temp_C": air_temp_C_time,
        # Carried through so downstream stages inherit identical forcing.
        "rh_pct": rh_pct_time, "wind_ms": wind_ms_time,
        "cloud_fraction": cloud_fraction_time,
        "LWin_Wm2": lwin_time,
        "radiation_source": radiation_source,
    })
    # Optional separate INLET (free-stream) wind series. 05b scales the
    # step-3 potential-flow field by this when present, so the convective
    # boundary condition varies through the day instead of using one constant.
    # Without it, wind_ms continues to serve as both, exactly as before.
    if weather.has_free_stream_wind():
        # Carried into times.csv so 05b can drive convection with the
        # experiment-derived free stream instead of the sheltered wind.
        times_df["wind_freestream_ms"] = weather.free_stream_wind_ms(
            np.asarray(hour_of_day, dtype=float))
    if weather.has_inlet_wind():
        times_df["wind_inlet_ms"] = weather.inlet_wind_ms(
            np.asarray(hour_of_day, dtype=float))
        print(f"    inlet wind        <- csv wind_inlet_ms "
              f"({times_df['wind_inlet_ms'].min():.2f}.."
              f"{times_df['wind_inlet_ms'].max():.2f} m/s, carried to times.csv)")
    times_df.to_csv(out_dir / "times.csv", index=False)

    if args.prep_only:
        print("\n" + "=" * 70)
        print("Preparation-only run complete; direct-sun/MRT loop was not run.")
        print("  path_xyz.npy                     -- (n_points, 3) point coordinates")
        print("  times.csv                        -- solar position + radiation per time step")
        print("  svf_effective.npy                -- (n_points,) static sky view factor")
        print("  tmrt_matrix_C.npy                -- NOT WRITTEN in --prep-only mode")
        print(f"\n[mrt_prep_result] n_points={n_points} n_times={nt} output_dir={out_dir}")
        return

    print("\n" + "=" * 70)
    print("Running direct-sun ray tracing + MRT for each time step...")
    environment = EnvironmentField(
        weather, args.microclimate_dir, args.microclimate_receptor_height_m)
    print(f"Air temperature / velocity forcing: {environment.describe()}")
    facet_lw = None
    if args.facet_thermal_dir:
        facet_lw = FacetLongwave(args.facet_thermal_dir, n_points, nt, args)
    tmrt_matrix = np.zeros((nt, n_points), dtype=np.float32)
    direct_transmission_matrix = np.zeros((nt, n_points), dtype=np.float32)
    local_air_min = np.zeros(nt, dtype=float)
    local_air_mean = np.zeros(nt, dtype=float)
    local_air_max = np.zeros(nt, dtype=float)
    local_wind_mean = np.zeros(nt, dtype=float)
    local_wind_max = np.zeros(nt, dtype=float)
    # Instrument-equivalent channels need the resolved surround and the ground
    # surface temperature below the receptor, so they are recorded only on the
    # facet-thermal path. Emitting zeros without that data would be worse than
    # emitting nothing.
    record_sensor = (args.sensor_equivalent_outputs == "on"
                     and facet_lw is not None)
    if args.sensor_equivalent_outputs == "on" and facet_lw is None:
        print("  NOTE: sensor-equivalent radiometer channels need "
              "--facet-thermal-dir (resolved surround + ground temperature); "
              "skipping them for this run.")
    if record_sensor:
        print(f"  Sensor-equivalent radiometer channels at "
              f"{args.sensor_height_m:.2f} m"
              + ("" if abs(args.sensor_height_m - args.z_height) < 1e-9 else
                 f" (receptor ray tracing at {args.z_height:.2f} m; "
                 f"shading/sky view reused)"))
        # The downward-facing channels need the instrument's OWN footprint, not
        # the body's cylinder view of the ground out to the culling distance.
        if facet_lw.build_sensor_ground_footprint(
                path_xyz, args.sensor_height_m, args.z_height):
            print(f"    downward footprint: half the weight within "
                  f"{args.sensor_height_m:.2f} m of the sensor, "
                  f"{facet_lw.sensor_footprint_coverage:.1%} of route points "
                  f"covered by resolved ground facets")
        else:
            print("    NOTE: no resolved ground facets for the downward "
                  "footprint; the upwelling channels will fall back to the "
                  "surround radiosity.")
    # The black globe needs the same resolved surround the radiometer does, so
    # it rides on the same precondition.
    record_globe = (args.black_globe_outputs == "on" and facet_lw is not None)
    globe_spec = None
    globe_args = None
    if record_globe:
        globe_spec = resolve_globe_spec(
            args.globe_preset,
            diameter_m=args.globe_diameter_m,
            emissivity=args.globe_emissivity,
            sw_absorptivity=args.globe_sw_absorptivity,
            areal_heat_capacity_J_m2K=args.globe_areal_heat_capacity)
        globe_args = globe_radiation_args(args, globe_spec)
        print(f"  Black-globe emulation: {describe_globe(globe_spec)}")
    elif args.black_globe_outputs == "on" and facet_lw is None:
        print("  NOTE: black-globe emulation needs --facet-thermal-dir "
              "(resolved surround); skipping it for this run.")
    contribution_matrices = None
    if record_contributions:
        stored_keys = PRIMARY_COLUMNS + TOTAL_COLUMNS
        if contribution_config["record_material_resolved_sources"]:
            stored_keys = stored_keys + SW_SOURCE_COLUMNS + LW_SOURCE_COLUMNS
        if record_sensor:
            stored_keys = stored_keys + SENSOR_COLUMNS
        if record_globe:
            stored_keys = stored_keys + GLOBE_COLUMNS
        contribution_matrices = {
            key: np.zeros((nt, n_points), dtype=np.float32)
            for key in stored_keys
        }
        validation_cfg = contribution_config["validation"]

    t_loop_start = time.time()
    for it, (t, el, az) in enumerate(zip(times, elev, azim)):
        local_environment = environment.sample(path_xyz, float(hour_of_day[it]))
        local_air_c = local_environment.air_temperature_c
        local_wind_ms = local_environment.wind_speed_ms
        local_air_min[it] = float(np.min(local_air_c))
        local_air_mean[it] = float(np.mean(local_air_c))
        local_air_max[it] = float(np.max(local_air_c))
        local_wind_mean[it] = float(np.mean(local_wind_ms))
        local_wind_max[it] = float(np.max(local_wind_ms))
        if el <= 0.0:
            tau_direct = np.zeros(n_points)
        else:
            sun_vec = sun_vector_enu(az, el)
            tau_direct = direct_solar_transmission_batched(
                path_xyz, sun_vec, building_intersector, vegetation_intersector,
                args.k_lad_direct, args.sun_batch_size,
            )

        L_surround_override = None
        L_surround_components = None
        lw_sky_frac = None
        local_ground_albedo = None
        reflected_source_fractions = None
        wall_reflected_incident = None
        wall_reflected_parts = None
        sun_vec_now = sun_vector_enu(az, el) if el > 0.0 else None
        if facet_lw is not None:
            sky_lw_current = (float(lwin_time[it]) if np.isfinite(lwin_time[it])
                              else float(sky_longwave_down(
                                  air_temp_C_time[it], rh_pct_time[it],
                                  cloud_fraction_time[it],
                                  clear_sky_model=args.clear_sky_emissivity)))
            surround_result = facet_lw.surround_at(
                it, air_temp_C_time[it], el,
                sky_longwave_Wm2=sky_lw_current,
                return_components=record_contributions)
            if record_contributions:
                L_surround_override, L_surround_components = surround_result
            else:
                L_surround_override = surround_result
            if args.lw_sky_fraction == "fullsphere":
                lw_sky_frac = facet_lw.sky_frac
            local_ground_albedo = facet_lw.local_ground_albedo
            reflected_source_fractions = facet_lw.reflected_source_fractions
            if args.wall_reflected_shortwave == "on":
                wall_reflected_incident, wall_reflected_parts = (
                    facet_lw.wall_reflected_shortwave(
                        it, dni[it], dhi[it], sun_vec_now))

        radiation_result = estimate_mrt_from_radiation(
            dni[it], dhi[it], ghi[it], el, tau_direct, svf_person, svf_ground,
            local_air_c, rh_pct_time[it], cloud_fraction_time[it], args,
            L_surround_override=L_surround_override, lw_sky_frac=lw_sky_frac,
            local_ground_albedo=local_ground_albedo,
            reflected_source_fractions=reflected_source_fractions,
            L_surround_components=L_surround_components,
            L_sky_override=lwin_time[it],
            wall_reflected_incident=wall_reflected_incident,
            wall_reflected_parts=wall_reflected_parts,
            return_contributions=record_contributions,
        )
        if record_contributions:
            tmrt_C, R_abs, K_sw, L_lw, contributions = radiation_result
            validate_contribution_arrays(
                contributions,
                absolute_tolerance=float(validation_cfg["absolute_tolerance_Wm2"]),
                relative_tolerance=float(validation_cfg["relative_tolerance"]),
                expected_mrt_c=tmrt_C,
                person_emissivity=args.person_emissivity,
                sigma=SIGMA,
                mrt_tolerance_c=float(validation_cfg["mrt_absolute_tolerance_C"]),
            )
            if record_sensor:
                # Mean reflected-shortwave radiance of the surround, reused by
                # the horizontal sensor for the part of its hemisphere that is
                # not sky. Derived from the same wall/roof reflection already
                # computed for the body.
                surround_sw_radiance = None
                if wall_reflected_incident is not None:
                    non_sky = np.maximum(1.0 - facet_lw.sky_frac, 1e-6)
                    surround_sw_radiance = wall_reflected_incident / non_sky
                # The upwelling pair uses the INSTRUMENT's footprint view of the
                # ground; the downwelling pair and the body keep their own.
                sensor_albedo = (facet_lw.sensor_ground_albedo
                                 if facet_lw.sensor_ground_albedo is not None
                                 else local_ground_albedo)
                lw_down_up = sw_refl_up = sw_up_facet = None
                if (args.sensor_emulation == "facet"
                        and facet_lw.has_sensor_up_view):
                    incident_sw = facet_lw.facet_incident_shortwave(
                        it, dni[it], dhi[it], sun_vec_now)
                    lw_down_up, sw_refl_up = facet_lw.sensor_downwelling_at(
                        it, air_temp_C_time[it], el, sky_lw_current,
                        incident_sw)
                    sw_up_facet = facet_lw.sensor_ground_reflected(incident_sw)
                sensor = sensor_radiometer_quantities(
                    dni[it], dhi[it], el, tau_direct, svf_planar,
                    sky_lw_current, L_surround_override,
                    sensor_albedo,
                    facet_lw.sensor_ground_emitted(it),
                    facet_lw.sensor_ground_emissivity,
                    surround_sw_radiance, args,
                    longwave_down_override=lw_down_up,
                    reflected_down_override=sw_refl_up,
                    shortwave_up_override=sw_up_facet)
                contributions.update(sensor)
            if record_globe:
                # Second pass over the SAME traced scene with sphere weighting
                # and globe optics -- no extra rays are cast.
                _, globe_flux, _, _ = estimate_mrt_from_radiation(
                    dni[it], dhi[it], ghi[it], el, tau_direct,
                    svf_person, svf_ground,
                    local_air_c, rh_pct_time[it], cloud_fraction_time[it],
                    globe_args,
                    L_surround_override=L_surround_override,
                    lw_sky_frac=lw_sky_frac,
                    local_ground_albedo=local_ground_albedo,
                    reflected_source_fractions=None,
                    L_surround_components=None,
                    L_sky_override=lwin_time[it],
                    wall_reflected_incident=wall_reflected_incident,
                    wall_reflected_parts=None,
                    return_contributions=False,
                )
                contributions["globe_absorbed_flux_Wm2"] = globe_flux
                contributions["globe_radiative_equilibrium_C"] = (
                    radiative_equilibrium_temperature_C(
                        globe_flux, globe_spec.emissivity))
                contributions["globe_steady_temperature_C"] = (
                    steady_globe_temperature_C(
                        globe_flux, local_air_c, local_wind_ms, globe_spec))
            for key, matrix in contribution_matrices.items():
                matrix[it, :] = contributions[key]
        else:
            tmrt_C, R_abs, K_sw, L_lw = radiation_result
        tmrt_matrix[it, :] = tmrt_C
        direct_transmission_matrix[it, :] = tau_direct

        if (it + 1) % 24 == 0 or it == nt - 1:
            elapsed = time.time() - t_loop_start
            print(f"  step {it + 1}/{nt} ({t.strftime('%H:%M')}) -- {elapsed:.0f}s elapsed")

    np.save(out_dir / "tmrt_matrix_C.npy", tmrt_matrix)
    np.save(out_dir / "direct_transmission_matrix.npy", direct_transmission_matrix)
    if record_contributions:
        closure = validate_contribution_arrays(
            contribution_matrices,
            absolute_tolerance=float(validation_cfg["absolute_tolerance_Wm2"]),
            relative_tolerance=float(validation_cfg["relative_tolerance"]),
            expected_mrt_c=tmrt_matrix,
            person_emissivity=args.person_emissivity,
            sigma=SIGMA,
            mrt_tolerance_c=float(validation_cfg["mrt_absolute_tolerance_C"]),
        )
        np.savez_compressed(out_dir / CONTRIBUTION_ARCHIVE, **contribution_matrices)
        write_contribution_metadata(out_dir / CONTRIBUTION_METADATA, {
            "schema_version": 1,
            "units": "absorbed W m^-2",
            "n_times": nt,
            "n_points": n_points,
            "person_shortwave_absorptivity": args.person_sw_absorptivity,
            "person_longwave_emissivity": args.person_emissivity,
            "projected_area_model": args.projected_area_model,
            "sky_view_body": args.sky_view_body,
            "primary_columns": PRIMARY_COLUMNS,
            "reflected_shortwave_source_columns": (
                SW_SOURCE_COLUMNS if contribution_config["record_material_resolved_sources"] else []),
            "surface_longwave_source_columns": (
                LW_SOURCE_COLUMNS if contribution_config["record_material_resolved_sources"] else []),
            "total_columns": TOTAL_COLUMNS,
            "shortwave_source_attribution": (
                "route-visible ground-facet albedo weights from the current local-ground reflection model"),
            "longwave_source_attribution": (
                "stage-05a first-hit facet material, canopy weight, or unresolved enclosure weight"),
            "mrt_equation": "Tmrt_K=(total_absorbed_flux/(person_emissivity*sigma))**0.25",
            "sensor_columns": SENSOR_COLUMNS if record_sensor else [],
            "sensor_height_m": args.sensor_height_m if record_sensor else None,
            "globe_columns": GLOBE_COLUMNS if record_globe else [],
            # Downstream stages need the exact globe to integrate the transient
            # response along a route, so the spec travels with the archive
            # rather than being re-guessed from CLI defaults.
            "black_globe": globe_spec.as_metadata() if record_globe else None,
            "closure": closure,
            "configuration": contribution_config,
        })
        print(f"  Saved absorbed-flux archive: {out_dir / CONTRIBUTION_ARCHIVE}")
    if facet_lw is not None and facet_lw.local_ground_albedo is not None:
        np.save(out_dir / "local_ground_albedo.npy",
                facet_lw.local_ground_albedo.astype(np.float32))
    print("\n" + "=" * 70)
    print("Saving lightweight summary (safe at any scale)...")
    summary_rows = []
    for it, t in enumerate(times):
        summary_rows.append({
            "time": t.isoformat(),
            "elevation_deg": elev[it],
            "DNI_Wm2": dni[it], "DHI_Wm2": dhi[it], "GHI_Wm2": ghi[it],
            "air_temp_C": local_air_mean[it],
            "air_temp_min_C": local_air_min[it],
            "air_temp_max_C": local_air_max[it],
            "wind_mean_ms": local_wind_mean[it],
            "wind_max_ms": local_wind_max[it],
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
            sub_environment = environment.sample(
                path_xyz[sub_idx], float(hour_of_day[it]))
            for local_index, ip in enumerate(sub_idx):
                records.append({
                    "time": t.isoformat(), "point_index": int(ip),
                    "x": path_xyz[ip, 0], "y": path_xyz[ip, 1], "z": path_xyz[ip, 2],
                    "svf_effective": svf_person[ip],
                    "direct_transmission": direct_transmission_matrix[it, ip],
                    "air_temp_C": float(
                        sub_environment.air_temperature_c[local_index]),
                    "wind_ms": float(
                        sub_environment.wind_speed_ms[local_index]),
                    "wind_u_ms": float(
                        sub_environment.velocity_u_ms[local_index]),
                    "wind_v_ms": float(
                        sub_environment.velocity_v_ms[local_index]),
                    "wind_w_ms": float(
                        sub_environment.velocity_w_ms[local_index]),
                    "tmrt_C": tmrt_matrix[it, ip],
                })
        pd.DataFrame(records).to_csv(out_dir / "detailed_subsample.csv", index=False)

    print("\n" + "=" * 70)
    print("Done. Key outputs in", out_dir)
    print("  path_xyz.npy                     -- (n_points, 3) point coordinates")
    print("  svf_effective.npy                -- (n_points,) static sky view factor")
    print("  tmrt_matrix_C.npy                -- (n_times, n_points) MRT, deg C")
    print("  direct_transmission_matrix.npy   -- (n_times, n_points) direct sun factor")
    if record_contributions:
        print(f"  {CONTRIBUTION_ARCHIVE:35s} -- absorbed flux components, W m^-2")
        print(f"  {CONTRIBUTION_METADATA:35s} -- units, conventions, and closure report")
    print("  times.csv                        -- solar position + radiation per time step")
    print("  summary_by_time.csv              -- lightweight per-timestep stats (always small)")
    if args.save_subsample_csv > 0:
        print("  detailed_subsample.csv           -- full per-point-per-time for a "
              "representative subsample")
    print(f"\n[mrt_result] n_points={n_points} n_times={nt} output_dir={out_dir}")


if __name__ == "__main__":
    main()
