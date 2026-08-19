"""Full-surface shortwave and hybrid longwave solver for TREC-Route.

Unlike the pedestrian receptor ray tracer, this module evaluates every mesh
facet.  Expensive geometric quantities are precomputed once; transient calls
are vectorized except for batched BVH visibility queries.  Route-zone rows of
the surface view-factor operator are stored in CSR form, while ambient facets
use a sky/canyon closure.

Sign convention: every returned flux is positive *into* the surface.  Thus
``longwave_net_Wm2`` may be negative and
``net_radiative_flux_Wm2 = sw_direct + sw_diffuse + longwave_net``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import trimesh

from thermal_common import SIGMA, get_intersector, make_hemisphere_directions_about_normal


@dataclass(frozen=True)
class RadiationResult:
    """Per-facet absorbed/net fluxes in W m-2."""

    sw_direct_absorbed_Wm2: np.ndarray
    sw_diffuse_absorbed_Wm2: np.ndarray
    lw_sky_absorbed_Wm2: np.ndarray
    lw_surface_absorbed_Wm2: np.ndarray
    lw_emitted_Wm2: np.ndarray
    longwave_net_Wm2: np.ndarray
    net_radiative_flux_Wm2: np.ndarray
    sunlit_area_fraction: np.ndarray


class UrbanRadiationEngine:
    """Radiation solver over a material-labelled triangulated neighborhood.

    Parameters are arrays in final-face order.  ``route_zone`` selects rows
    receiving ray-checked facet-to-facet view factors.  All facets remain ray
    occluders and all facets receive direct, diffuse, and longwave flux.
    """

    CACHE_VERSION = 1

    def __init__(self, mesh: trimesh.Trimesh, albedo: np.ndarray,
                 emissivity: np.ndarray, route_zone: np.ndarray,
                 *, surface_offset_m: float = 2.0e-3,
                 ray_batch_size: int = 200_000):
        self.mesh = mesh
        self.centroids = np.asarray(mesh.triangles_center, dtype=float)
        self.normals = np.asarray(mesh.face_normals, dtype=float)
        self.areas = np.asarray(mesh.area_faces, dtype=float)
        self.albedo = self._facet_array(albedo, "albedo")
        self.emissivity = self._facet_array(emissivity, "emissivity")
        self.route_zone = np.asarray(route_zone, dtype=bool)
        if self.route_zone.shape != (len(self.mesh.faces),):
            raise ValueError("route_zone must contain one flag per mesh face")
        if np.any((self.albedo < 0) | (self.albedo > 1)):
            raise ValueError("facet albedo must lie in [0,1]")
        if np.any((self.emissivity <= 0) | (self.emissivity > 1)):
            raise ValueError("facet emissivity must lie in (0,1]")
        if np.any(~np.isfinite(self.centroids)) or np.any(self.areas <= 0):
            raise ValueError("radiation mesh contains invalid facets")
        self.absorptivity = 1.0 - self.albedo
        self.surface_offset_m = float(surface_offset_m)
        self.ray_batch_size = int(ray_batch_size)
        self.intersector = get_intersector(mesh, quiet=True)
        self.sky_view_factor: np.ndarray | None = None
        self.view_factors: sparse.csr_matrix | None = None

    def _facet_array(self, values: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.mesh.faces),) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite value for every face")
        return values

    @staticmethod
    def _four_samples(triangles: np.ndarray) -> np.ndarray:
        """Three inward-nudged vertices and centroid, shape (faces,4,3)."""
        tri = np.asarray(triangles, dtype=float)
        centroid = tri.mean(axis=1)
        # 96% vertex + 2% of each other vertex stays inside the triangle.
        nudged = np.stack([
            0.96 * tri[:, 0] + 0.02 * tri[:, 1] + 0.02 * tri[:, 2],
            0.02 * tri[:, 0] + 0.96 * tri[:, 1] + 0.02 * tri[:, 2],
            0.02 * tri[:, 0] + 0.02 * tri[:, 1] + 0.96 * tri[:, 2],
        ], axis=1)
        return np.concatenate([nudged, centroid[:, None, :]], axis=1)

    def _unoccluded(self, origins: np.ndarray, directions: np.ndarray,
                    maximum_distance: np.ndarray | float | None = None) -> np.ndarray:
        """Any-hit visibility in bounded batches (True means no blocker)."""
        origins = np.asarray(origins, dtype=float)
        directions = np.asarray(directions, dtype=float)
        visible = np.ones(len(origins), dtype=bool)
        for start in range(0, len(origins), self.ray_batch_size):
            stop = min(start + self.ray_batch_size, len(origins))
            loc, iray, _ = self.intersector.intersects_location(
                origins[start:stop], directions[start:stop], multiple_hits=False)
            if not len(iray):
                continue
            distance = np.einsum(
                "ij,ij->i", loc - origins[start:stop][iray], directions[start:stop][iray])
            if maximum_distance is None:
                blocked = distance > self.surface_offset_m * 0.5
            else:
                limit = (np.asarray(maximum_distance)[start:stop][iray]
                         if np.ndim(maximum_distance) else float(maximum_distance))
                # A hit at the destination facet is not an obstruction.
                blocked = ((distance > self.surface_offset_m * 0.5)
                           & (distance < limit - 2.0 * self.surface_offset_m))
            visible[start + iray[blocked]] = False
        return visible

    def direct_sunlit_fraction(self, sun_vector: np.ndarray) -> np.ndarray:
        """Four-point area estimator for direct-beam visibility per facet."""
        sun = np.asarray(sun_vector, dtype=float)
        if sun.shape != (3,) or not np.isfinite(sun).all():
            raise ValueError("sun_vector must be one finite 3-vector")
        magnitude = np.linalg.norm(sun)
        if magnitude <= 0:
            raise ValueError("sun_vector must be non-zero")
        sun /= magnitude
        cosine = self.normals @ sun
        active = np.flatnonzero(cosine > 0)
        result = np.zeros(len(self.mesh.faces), dtype=float)
        if not len(active):
            return result
        facet_batch = max(1, self.ray_batch_size // 4)
        triangles = np.asarray(self.mesh.triangles)
        for start in range(0, len(active), facet_batch):
            ids = active[start:start + facet_batch]
            points = self._four_samples(triangles[ids])
            origins = (points + self.surface_offset_m * self.normals[ids, None, :]).reshape(-1, 3)
            directions = np.repeat(sun[None, :], len(origins), axis=0)
            clear = self._unoccluded(origins, directions).reshape(len(ids), 4)
            result[ids] = clear.mean(axis=1)
        return result

    def precompute_sky_view_factor(self, n_directions: int = 64,
                                   facet_batch_size: int = 4000) -> np.ndarray:
        """Cosine-weighted outward-hemisphere sky fraction for every facet."""
        if n_directions < 4:
            raise ValueError("at least four sky directions are required")
        result = np.empty(len(self.mesh.faces), dtype=float)
        for start in range(0, len(result), facet_batch_size):
            stop = min(start + facet_batch_size, len(result))
            normals = self.normals[start:stop]
            directions = make_hemisphere_directions_about_normal(
                normals, n_dirs=n_directions).reshape(-1, 3)
            origins = np.repeat(
                self.centroids[start:stop] + self.surface_offset_m * normals,
                n_directions, axis=0)
            result[start:stop] = self._unoccluded(origins, directions).reshape(
                stop - start, n_directions).mean(axis=1)
        self.sky_view_factor = np.clip(result, 0.0, 1.0)
        return self.sky_view_factor

    def precompute_route_view_factors(self, maximum_distance_m: float = 40.0,
                                      cutoff: float = 1.0e-5,
                                      maximum_neighbors: int = 96,
                                      receiver_batch_size: int = 5000) -> sparse.csr_matrix:
        """Build ray-checked differential view factors in route-zone rows.

        Candidate sources are the nearest geometrically relevant facets.  Row
        sums are capped to the non-sky enclosure fraction; any unresolved
        fraction is closed with the canyon mean during transient evaluation.
        """
        if (maximum_distance_m <= 0 or cutoff < 0 or maximum_neighbors < 1
                or receiver_batch_size < 1):
            raise ValueError("invalid route view-factor controls")
        tree = cKDTree(self.centroids)
        route_ids = np.flatnonzero(self.route_zone)
        n = len(self.mesh.faces)
        if not len(route_ids):
            matrix = sparse.csr_matrix((n, n), dtype=float)
            self.view_factors = matrix
            return matrix
        # Vectorized fixed-size nearest-neighbor candidate tables. Batching
        # bounds peak memory without reverting to one BVH call per receiver.
        k = min(n, maximum_neighbors + 1)
        all_rows, all_cols, all_data = [], [], []
        for start in range(0, len(route_ids), receiver_batch_size):
            receivers_batch = route_ids[start:start + receiver_batch_size]
            distance, candidate = tree.query(
                self.centroids[receivers_batch], k=k,
                distance_upper_bound=maximum_distance_m, workers=-1)
            if k == 1:
                distance, candidate = distance[:, None], candidate[:, None]
            receiver = np.repeat(receivers_batch, k)
            candidate = candidate.reshape(-1)
            distance = distance.reshape(-1)
            valid = ((candidate < n) & (candidate != receiver) & np.isfinite(distance)
                     & (distance > 4.0 * self.surface_offset_m))
            receiver, candidate, distance = receiver[valid], candidate[valid], distance[valid]
            delta = self.centroids[candidate] - self.centroids[receiver]
            direction = delta / distance[:, None]
            cos_i = np.einsum("ij,ij->i", direction, self.normals[receiver])
            cos_j = -np.einsum("ij,ij->i", direction, self.normals[candidate])
            geometric = (cos_i > 0) & (cos_j > 0)
            receiver, candidate, distance = (value[geometric] for value in
                                             (receiver, candidate, distance))
            direction = direction[geometric]
            cos_i, cos_j = cos_i[geometric], cos_j[geometric]
            origins = self.centroids[receiver] + self.surface_offset_m * self.normals[receiver]
            visible = self._unoccluded(origins, direction, distance)
            factor = cos_i * cos_j * self.areas[candidate] / (np.pi * distance**2)
            keep = visible & (factor >= cutoff)
            all_rows.append(receiver[keep])
            all_cols.append(candidate[keep])
            all_data.append(factor[keep])
        nonempty = [index for index, values in enumerate(all_data) if len(values)]
        if nonempty:
            matrix = sparse.coo_matrix(
                (np.concatenate([all_data[i] for i in nonempty]),
                 (np.concatenate([all_rows[i] for i in nonempty]),
                  np.concatenate([all_cols[i] for i in nonempty]))),
                shape=(n, n)).tocsr()
        else:
            matrix = sparse.csr_matrix((n, n), dtype=float)
        # Differential factors can over-close a row when large nearby source
        # triangles are treated as point patches. Cap, never inflate, to the
        # non-sky enclosure fraction. The remainder is the canyon closure.
        row_sum = np.asarray(matrix.sum(axis=1)).ravel()
        target = (1.0 - self.sky_view_factor if self.sky_view_factor is not None
                  else np.ones(n))
        scale = np.ones(n)
        over = row_sum > target
        scale[over] = np.divide(target[over], row_sum[over],
                                out=np.zeros(np.sum(over)), where=row_sum[over] > 0)
        matrix = (sparse.diags(scale) @ matrix).tocsr()
        self.view_factors = matrix
        return matrix

    def evaluate(self, *, sun_vector: np.ndarray, direct_normal_Wm2: float,
                 diffuse_horizontal_Wm2: float, sky_temperature_K: float,
                 surface_temperature_K: np.ndarray,
                 canyon_temperature_K: float | None = None,
                 sunlit_area_fraction: np.ndarray | None = None) -> RadiationResult:
        """Evaluate one transient forcing state after geometric precompute."""
        if self.sky_view_factor is None:
            raise RuntimeError("precompute_sky_view_factor must be called first")
        temperature = self._facet_array(surface_temperature_K, "surface temperature")
        if np.any(temperature <= 0) or sky_temperature_K <= 0:
            raise ValueError("radiative temperatures must be in kelvin and positive")
        if direct_normal_Wm2 < 0 or diffuse_horizontal_Wm2 < 0:
            raise ValueError("solar irradiance cannot be negative")
        sun = np.asarray(sun_vector, dtype=float)
        sun /= max(np.linalg.norm(sun), 1e-15)
        cosine = np.maximum(self.normals @ sun, 0.0)
        fraction = (self.direct_sunlit_fraction(sun) if sunlit_area_fraction is None
                    else self._facet_array(sunlit_area_fraction, "sunlit area fraction"))
        if np.any((fraction < 0) | (fraction > 1)):
            raise ValueError("sunlit area fractions must lie in [0,1]")
        direct = self.absorptivity * cosine * float(direct_normal_Wm2) * fraction
        diffuse = self.absorptivity * self.sky_view_factor * float(diffuse_horizontal_Wm2)

        emitted_black = SIGMA * temperature**4
        sky_black = SIGMA * float(sky_temperature_K)**4
        if canyon_temperature_K is None:
            canyon_temperature_K = float(np.average(temperature, weights=self.areas))
        canyon_black = SIGMA * float(canyon_temperature_K)**4
        sky_absorbed = self.emissivity * self.sky_view_factor * sky_black
        surface_incident_black = (1.0 - self.sky_view_factor) * canyon_black
        if self.view_factors is not None and self.route_zone.any():
            row_sum = np.asarray(self.view_factors.sum(axis=1)).ravel()
            unresolved = np.maximum(0.0, 1.0 - self.sky_view_factor - row_sum)
            environment_irradiation = (self.sky_view_factor * sky_black
                                       + unresolved * canyon_black)
            # Grey radiosity: emitted plus repeatedly reflected longwave. The
            # matrix has non-zero route rows only; ambient radiosity uses its
            # direct sky/canyon closure and remains available as a source.
            ambient_irradiation = (self.sky_view_factor * sky_black
                                   + (1.0 - self.sky_view_factor) * canyon_black)
            radiosity = (self.emissivity * emitted_black
                         + (1.0 - self.emissivity) * ambient_irradiation)
            for _ in range(20):
                mutual = np.asarray(self.view_factors @ radiosity).ravel()
                irradiation = environment_irradiation + mutual
                updated = self.emissivity * emitted_black + (1.0 - self.emissivity) * irradiation
                change = np.max(np.abs(updated[self.route_zone] - radiosity[self.route_zone]))
                radiosity[self.route_zone] = updated[self.route_zone]
                if change < 1.0e-7:
                    break
            resolved = np.asarray(self.view_factors @ radiosity).ravel()
            surface_incident_black[self.route_zone] = (
                resolved[self.route_zone] + unresolved[self.route_zone] * canyon_black)
        surface_absorbed = self.emissivity * surface_incident_black
        emitted = self.emissivity * emitted_black
        longwave_net = sky_absorbed + surface_absorbed - emitted
        net = direct + diffuse + longwave_net
        arrays = (direct, diffuse, sky_absorbed, surface_absorbed, emitted,
                  longwave_net, net, fraction)
        if not all(np.isfinite(value).all() for value in arrays):
            raise FloatingPointError("non-finite urban radiation result")
        return RadiationResult(*arrays)

    def save_precomputation(self, directory: str | Path) -> None:
        """Persist reusable SVF and CSR view factors."""
        if self.sky_view_factor is None or self.view_factors is None:
            raise RuntimeError("SVF and route view factors must both be precomputed")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "facet_sky_view_factor.npy", self.sky_view_factor.astype(np.float32))
        sparse.save_npz(directory / "route_zone_view_factors.npz", self.view_factors)
