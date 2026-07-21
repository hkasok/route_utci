"""
thermal_common.py -- shared primitives for the facet surface-temperature
pipeline (05a facet selection / 05b energy balance / 05 MRT integration).

Everything here is a small, pure, unit-testable function. The analytic
verification for each lives in verify_thermal_pipeline.py:

  make_sphere_directions      -> weights sum to 1; isothermal enclosure
                                 reproduces sigma*T^4 exactly
  solve_tridiagonal_batched   -> matches dense np.linalg.solve to ~1e-12
  vegetation_transmission...  -> identical function already validated in 05
  nearest_hit_multi           -> checked on a synthetic scene (known hits)
"""

import json

import numpy as np

SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W m^-2 K^-4

# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH FOR SURFACE RADIATIVE PROPERTIES
#
# These values are consumed by BOTH sides of the radiation calculation:
#
#   05b_facet_energy_balance.py  -- how much shortwave a ground facet ABSORBS
#                                   (drives its surface temperature)
#   05_mrt_network_raytrace.py   -- how much shortwave the ground REFLECTS
#                                   onto the pedestrian
#
# Before this was centralised the two disagreed: the ground reflected 20% of
# incident shortwave toward a pedestrian while absorbing as though it
# reflected only 12%. That is not a tuning difference, it is a violation of
# energy conservation for the same physical surface, and it biased ground
# facets warm while simultaneously overstating reflected load on the person.
# Change the value HERE and both stages follow.
#
# GROUND_ALBEDO reference points:
#   0.12  previous 05b value -- darker than any standard urban class
#   0.16  grass / unmanaged vegetation
#   0.18  dark asphalt   <-- default; matches the SOLWEIG land-cover class
#   0.20  cobble stone / concrete, previous 05 pedestrian-side value
# ---------------------------------------------------------------------------
GROUND_ALBEDO = 0.18

DEFAULT_MATERIALS = {
    "ground": {"k": 1.00, "C": 2.0e6, "depth": 0.50, "n_layers": 8,
               "albedo": GROUND_ALBEDO, "emissivity": 0.95,
               "bottom_bc": "fixed"},     # fixed T_deep at depth
    "wall":   {"k": 1.40, "C": 1.8e6, "depth": 0.25, "n_layers": 6,
               "albedo": 0.30, "emissivity": 0.90,
               "bottom_bc": "interior"},  # convective to indoor air
    "roof":   {"k": 1.00, "C": 1.6e6, "depth": 0.25, "n_layers": 6,
               "albedo": 0.15, "emissivity": 0.92,
               "bottom_bc": "interior"},
}

# Filename written by 05b into its output directory and read back by 05, so
# the two stages can be checked against each other rather than trusted.
MATERIALS_MANIFEST = "materials_used.json"


def load_materials(material_json=None):
    """Resolve the material table, applying an optional JSON override.

    Returns a deep-ish copy so callers cannot mutate the module defaults.
    """
    materials = {k: dict(v) for k, v in DEFAULT_MATERIALS.items()}
    if material_json:
        with open(material_json) as f:
            overrides = json.load(f)
        for name, over in overrides.items():
            if name not in materials:
                raise KeyError(
                    f"--material-json references unknown surface class "
                    f"'{name}'; expected one of {sorted(materials)}")
            materials[name].update(over)
    return materials


def resolve_ground_albedo(args, facet_thermal_dir=None):
    """Ground albedo for the pedestrian-side reflected-shortwave term.

    Precedence:
      1. An explicit --ground-albedo on the command line (wins, but is
         cross-checked below and warned about if it disagrees).
      2. The manifest written by 05b, if a facet-thermal directory is given.
         This is authoritative because it is what actually heated the ground.
      3. --material-json, if supplied.
      4. The module default, GROUND_ALBEDO.
    """
    from pathlib import Path

    explicit = getattr(args, "ground_albedo", None)
    source = "module default"
    albedo = GROUND_ALBEDO

    if getattr(args, "material_json", None):
        albedo = load_materials(args.material_json)["ground"]["albedo"]
        source = f"--material-json ({args.material_json})"

    manifest_albedo = None
    if facet_thermal_dir:
        manifest = Path(facet_thermal_dir) / MATERIALS_MANIFEST
        if manifest.is_file():
            with open(manifest) as f:
                manifest_albedo = json.load(f)["ground"]["albedo"]
            albedo = manifest_albedo
            source = f"05b manifest ({manifest})"

    if explicit is not None:
        if manifest_albedo is not None and abs(explicit - manifest_albedo) > 1e-9:
            print(f"  WARNING: --ground-albedo {explicit:.3f} disagrees with "
                  f"the {manifest_albedo:.3f} that 05b actually used to heat "
                  f"the ground facets. The reflected-shortwave term and the "
                  f"ground energy balance will not conserve energy. Drop "
                  f"--ground-albedo to inherit the consistent value.")
        albedo = explicit
        source = "--ground-albedo (explicit)"

    return float(albedo), source

# Facet class codes (kept as small ints so they can live in compact arrays)
CLASS_GROUND = 0
CLASS_WALL = 1
CLASS_ROOF = 2
CLASS_VEGETATION = 3   # virtual: no energy balance; radiates at ~air temp
CLASS_NAMES = {CLASS_GROUND: "ground", CLASS_WALL: "wall",
               CLASS_ROOF: "roof", CLASS_VEGETATION: "vegetation"}


def get_intersector(mesh, quiet=False):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        if not quiet:
            print("  Using pyembree ray intersector.")
    except Exception:
        from trimesh.ray.ray_triangle import RayMeshIntersector
        if not quiet:
            print("  Using trimesh triangle ray intersector (slower).")
    return RayMeshIntersector(mesh)


def make_sphere_directions(n_azimuth, n_elevation, body="cylinder"):
    """Full-sphere direction set for LONGWAVE view sampling around a
    pedestrian, with weights normalized to sum to exactly 1.

    Elevation bands are uniform in elevation angle over (-pi/2, +pi/2)
    (band centers at (ie+0.5)*pi/n_el - pi/2), azimuth uniform over 2*pi.

    Weights = (solid angle of the cell, proportional to cos(el)) times a
    body projection factor:
      body="sphere"   : projection 1        (spherical receptor)
      body="cylinder" : projection cos(el)  (standing person; lateral
                        directions dominate, zenith/nadir de-emphasized --
                        the same idea as RayMan's cylindric human model)

    NOTE ON VERIFIABILITY: because the weights are normalized, an
    isothermal blackbody environment at temperature T yields
    sum_i w_i * sigma*T^4 = sigma*T^4 for ANY body factor -- this is the
    first unit test, and it also guarantees the exact backward-
    compatibility property used in test T2 of the verifier.
    """
    dirs, w = [], []
    for ie in range(n_elevation):
        el = (ie + 0.5) * np.pi / n_elevation - 0.5 * np.pi
        for ia in range(n_azimuth):
            az = 2.0 * np.pi * (ia + 0.5) / n_azimuth
            dirs.append([np.cos(el) * np.sin(az),
                         np.cos(el) * np.cos(az),
                         np.sin(el)])
            solid_angle = np.cos(el)          # d(omega) ~ cos(el) del daz
            proj = np.cos(el) if body == "cylinder" else 1.0
            w.append(solid_angle * proj)
    dirs = np.asarray(dirs, dtype=float)
    w = np.asarray(w, dtype=float)
    return dirs, w / w.sum()


def make_hemisphere_directions_about_normal(normals, n_dirs=64, seed=12345):
    """Cosine-weighted hemisphere directions about each facet's outward
    normal, for computing the facet's sky fraction f_sky.

    Cosine weighting means each returned direction carries EQUAL weight
    (1/n_dirs) in the irradiance integral over the facet -- the standard
    Monte-Carlo estimator for a Lambertian receiver. The same fixed
    low-discrepancy-ish sample (deterministic per seed) is used for every
    facet so results are reproducible run to run.

    Returns array of shape (n_facets, n_dirs, 3).
    """
    rng = np.random.default_rng(seed)
    # Stratified (u1,u2) for lower variance than plain uniform sampling
    n_side = int(np.ceil(np.sqrt(n_dirs)))
    u1g, u2g = np.meshgrid((np.arange(n_side) + 0.5) / n_side,
                           (np.arange(n_side) + 0.5) / n_side)
    u1 = u1g.ravel()[:n_dirs]
    u2 = u2g.ravel()[:n_dirs]
    # jitter within strata (deterministic)
    u1 = np.clip(u1 + (rng.random(n_dirs) - 0.5) / n_side, 1e-6, 1 - 1e-6)
    u2 = (u2 + (rng.random(n_dirs) - 0.5) / n_side) % 1.0

    # cosine-weighted local hemisphere (z = local normal)
    r = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    local = np.column_stack([r * np.cos(phi), r * np.sin(phi),
                             np.sqrt(np.maximum(0.0, 1.0 - u1))])

    normals = np.asarray(normals, dtype=float)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    # Build an orthonormal frame (t1, t2, n) per facet, vectorized
    helper = np.where(np.abs(normals[:, 2:3]) < 0.9,
                      np.tile([0.0, 0.0, 1.0], (len(normals), 1)),
                      np.tile([1.0, 0.0, 0.0], (len(normals), 1)))
    t1 = np.cross(helper, normals)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)
    # (nf,1,3)*(1,nd,1) sums -> (nf, nd, 3)
    out = (t1[:, None, :] * local[None, :, 0:1]
           + t2[:, None, :] * local[None, :, 1:2]
           + normals[:, None, :] * local[None, :, 2:3])
    return out


def nearest_hit_multi(intersectors, origins, directions, max_distance,
                      min_distance=1e-4):
    """First (nearest) hit for each ray across several meshes.

    intersectors: list of (mesh_id, RayMeshIntersector)
    Returns (hit_mesh_id, hit_face, hit_dist), each length n_rays;
    hit_mesh_id = -1 where nothing was hit within max_distance.

    Verified in the synthetic-scene test (rays with known first surface).
    """
    n = len(origins)
    best_dist = np.full(n, np.inf)
    best_mesh = np.full(n, -1, dtype=np.int32)
    best_face = np.full(n, -1, dtype=np.int64)
    for mesh_id, inter in intersectors:
        loc, iray, itri = inter.intersects_location(
            origins, directions, multiple_hits=True)
        if len(iray) == 0:
            continue
        d = np.einsum("ij,ij->i", loc - origins[iray], directions[iray])
        ok = (d > min_distance) & (d <= max_distance)
        iray, itri, d = iray[ok], itri[ok], d[ok]
        if len(iray) == 0:
            continue
        # keep nearest hit per ray within this mesh
        order = np.lexsort((d, iray))
        iray, itri, d = iray[order], itri[order], d[order]
        first = np.concatenate(([True], iray[1:] != iray[:-1]))
        iray, itri, d = iray[first], itri[first], d[first]
        closer = d < best_dist[iray]
        upd = iray[closer]
        best_dist[upd] = d[closer]
        best_mesh[upd] = mesh_id
        best_face[upd] = itri[closer]
    return best_mesh, best_face, best_dist


def vegetation_transmission_from_intersections(vegetation_intersector, origins,
                                               directions, k_lad,
                                               min_distance=1e-6,
                                               unique_tol=1e-5):
    """Beer-Lambert transmission through vegetation along each ray.
    IDENTICAL algorithm to the version already validated byte-for-byte in
    05_mrt_network_raytrace.py (kept here so 05a/05b can share it)."""
    n_rays = origins.shape[0]
    L_veg = np.zeros(n_rays, dtype=float)

    locations, index_ray, _ = vegetation_intersector.intersects_location(
        origins, directions, multiple_hits=True)
    if len(index_ray) == 0:
        return np.ones(n_rays, dtype=float), L_veg

    dist = np.einsum("ij,ij->i", locations - origins[index_ray],
                     directions[index_ray])
    valid = dist > min_distance
    index_ray, dist = index_ray[valid], dist[valid]
    if len(index_ray) == 0:
        return np.ones(n_rays, dtype=float), L_veg

    order = np.lexsort((dist, index_ray))
    index_ray, dist = index_ray[order], dist[order]

    same_ray_as_prev = np.concatenate(([False], index_ray[1:] == index_ray[:-1]))
    close_to_prev = np.concatenate(([False], (dist[1:] - dist[:-1]) <= unique_tol))
    keep = ~(same_ray_as_prev & close_to_prev)
    index_ray, dist = index_ray[keep], dist[keep]
    if len(index_ray) == 0:
        return np.ones(n_rays, dtype=float), L_veg

    group_change = np.concatenate(([True], index_ray[1:] != index_ray[:-1]))
    idx_arr = np.arange(len(index_ray))
    group_start = np.maximum.accumulate(np.where(group_change, idx_arr, 0))
    position_in_group = idx_arr - group_start

    group_ids, group_counts = np.unique(index_ray, return_counts=True)
    size_lookup = np.zeros(n_rays, dtype=int)
    size_lookup[group_ids] = group_counts
    group_size = size_lookup[index_ray]
    is_last_in_odd_group = ((position_in_group == group_size - 1)
                            & (group_size % 2 == 1))

    sign = np.where(position_in_group % 2 == 0, -1.0, 1.0)
    sign[is_last_in_odd_group] = 0.0
    L_veg = np.bincount(index_ray, weights=sign * dist, minlength=n_rays)
    L_veg = np.maximum(L_veg, 0.0)
    return np.exp(-k_lad * L_veg), L_veg


def solve_tridiagonal_batched(a, b, c, d):
    """Thomas algorithm, vectorized over a batch of independent systems.

    a: (m, n) sub-diagonal   (a[:,0] unused)
    b: (m, n) diagonal
    c: (m, n) super-diagonal (c[:,-1] unused)
    d: (m, n) right-hand side
    Returns x of shape (m, n).

    Verified against np.linalg.solve on random diagonally-dominant
    systems (test T3 in verify_thermal_pipeline.py).
    """
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    c = np.asarray(c, dtype=float).copy()
    d = np.asarray(d, dtype=float).copy()
    m, n = b.shape
    for i in range(1, n):
        w = a[:, i] / b[:, i - 1]
        b[:, i] -= w * c[:, i - 1]
        d[:, i] -= w * d[:, i - 1]
    x = np.empty_like(d)
    x[:, -1] = d[:, -1] / b[:, -1]
    for i in range(n - 2, -1, -1):
        x[:, i] = (d[:, i] - c[:, i] * x[:, i + 1]) / b[:, i]
    return x


def sun_vector_enu(azimuth_deg, elevation_deg):
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    v = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az),
                  np.sin(el)], dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(3)


def sky_longwave_down(air_temp_C, cloud_fraction):
    """Same clear/cloudy sky emissivity model already used in 05, factored
    out so 05 and 05b are guaranteed consistent."""
    air_K = np.asarray(air_temp_C, dtype=float) + 273.15
    cloud = np.clip(cloud_fraction, 0.0, 1.0)
    eps_sky = 0.78 * (1.0 - cloud) + 0.98 * cloud
    return eps_sky * SIGMA * air_K ** 4
