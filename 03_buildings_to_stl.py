"""
03_buildings_to_stl.py -- building points -> LoD1 building solids (STL).

Two models are available (``--lod1-model``):

segmented (default)
    A level-of-detail-1 model with one flat roof per BUILDING PART, where a
    part is a connected patch of roof whose measured height varies by no more
    than one storey (``--height-tolerance``, default 3 m). It replaces the
    block model below, which gave a whole merged city block one roof and one
    floor and was wrong in the way that matters on hills: in the Lisbon scenes
    the block roofs sat a median 3-6 m above the measured roof surface, and
    downhill walls stood on a floor taken from the block centroid.

      1. Rasterise the building returns (``--raster-res``, 0.5 m): per cell the
         highest return is the roof surface; terrain comes from the ground
         returns, interpolated under the buildings.
      2. Occupancy = cells with a return at least ``--min-height`` above the
         terrain; one closing pass bridges single empty cells; holes smaller
         than ``--min-hole-area`` are filled (sampling gaps), larger ones are
         kept (courtyards and light wells -- the block model filled every
         enclosed hole); parts smaller than ``--min-part-area`` are dropped.
      3. Parts: cells are split wherever adjacent roof heights differ by more
         than half the tolerance (a facade step; a 45-degree pitched roof rises
         only 0.5 m between 0.5 m cells, so it is never cut). Any part whose
         height range still exceeds the tolerance (a terrace stepping down a
         slope) is cut into tolerance-high bands. Parts below
         ``--min-part-area`` are merged into the neighbour they share the
         longest edge with.
      4. Roof height = median measured roof height of the part (robust to
         chimneys and stray returns; for a pitched roof it is the mid-pitch
         height, close to the volume-preserving value).
      5. Outlines are the exact cell boundaries, simplified as a polygon
         COVERAGE (``--simplify-tolerance``) so neighbouring parts keep
         identical shared edges and no slit opens between them.
      6. Each part is extruded from below the lowest terrain under it to its
         roof. Where two parts meet, only the step between their roofs is
         walled, so there are no internal faces whose centroid would sit
         inside a neighbouring solid. All faces are wound outward (roofs up,
         walls away from the solid, floors down), which stage 05a relies on.

block (legacy, reproduces the geometry used before 11 Oct 2026)
    Cluster the points into connected blocks, take each block's outer contour
    with enclosed holes filled, one roof at the ``--roof-percentile`` height
    of the whole block and one floor at the mean terrain near its centroid.

Run:
    python3 03_buildings_to_stl.py \
        --input split/building_points.npy \
        --output buildings.stl \
        --ground-npy split/ground_and_water_points.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import shapely
import trimesh
from grid_cluster import grid_cluster_2d
from skimage.measure import find_contours, approximate_polygon
from skimage.morphology import remove_small_holes, remove_small_objects
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import binary_closing, binary_fill_holes
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import mapbox_earcut as earcut


def parse_args():
    p = argparse.ArgumentParser(description="Building points -> LoD1 building solids STL")
    p.add_argument("--input", required=True, help="Path to building_points.npy (Nx3 array)")
    p.add_argument("--output", required=True, help="Output STL path")
    p.add_argument("--ground-npy", required=True,
                    help="Path to ground_and_water_points.npy (terrain under and "
                         "around the buildings)")
    p.add_argument("--lod1-model", choices=["segmented", "block"], default="segmented",
                    help="'segmented' (default): one roof per building part of "
                         "near-uniform height, courtyards kept, floors on the "
                         "local terrain. 'block': the legacy one-roof-per-block "
                         "model, kept only to reproduce earlier geometry.")
    p.add_argument("--raster-res", type=float, default=0.5,
                    help="Raster cell size, meters (default: 0.5)")
    p.add_argument("--simplify-tolerance", type=float, default=0.75,
                    help="Outline simplification tolerance, meters (default: 0.75)")
    # segmented model
    p.add_argument("--height-tolerance", type=float, default=3.0,
                    help="segmented: largest roof-height range within one "
                         "building part, meters -- one storey (default: 3.0)")
    p.add_argument("--min-height", type=float, default=0.5,
                    help="segmented: returns less than this above the terrain "
                         "are not treated as building, meters (default: 0.5)")
    p.add_argument("--min-hole-area", type=float, default=4.0,
                    help="segmented: enclosed gaps smaller than this are sampling "
                         "holes and are filled; larger ones are courtyards and "
                         "are kept, m^2 (default: 4.0)")
    p.add_argument("--min-part-area", type=float, default=6.0,
                    help="segmented: smallest building or building part, m^2 "
                         "(default: 6.0)")
    p.add_argument("--floor-margin", type=float, default=0.3,
                    help="segmented: walls reach this far below the terrain "
                         "beside them, meters (default: 0.3)")
    p.add_argument("--ground-stl", default=None,
                    help="segmented: the terrain mesh the simulation will use. "
                         "Strongly recommended: every outer wall then reaches "
                         "below THIS surface along its whole length, so no ray "
                         "can pass under a building. Without it the terrain is "
                         "interpolated from --ground-npy, which can sit above a "
                         "coarser terrain mesh at a steep drop.")
    p.add_argument("--report", default=None,
                    help="segmented: write a JSON fit report here (default: "
                         "next to the STL)")
    # block (legacy) model
    p.add_argument("--cell-size", type=float, default=0.5,
                    help="block: clustering grid cell size, meters (default: 0.5)")
    p.add_argument("--connect-radius", type=int, default=1,
                    help="block: grid cells of dilation before connecting "
                         "components (default: 1)")
    p.add_argument("--min-cluster-points", type=int, default=50,
                    help="block: skip clusters with fewer points (default: 50)")
    p.add_argument("--roof-percentile", type=float, default=90.0,
                    help="block: percentile of cluster Z used as roof height "
                         "(default: 90.0)")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Segmented LoD1 model
# ----------------------------------------------------------------------------

def thin_lowest(points, spacing):
    """Lowest point per horizontal cell -- a terrain-preserving thinning."""
    ix = np.floor((points[:, 0] - points[:, 0].min()) / spacing).astype(np.int64)
    iy = np.floor((points[:, 1] - points[:, 1].min()) / spacing).astype(np.int64)
    key = iy * (int(ix.max()) + 1) + ix
    order = np.lexsort((points[:, 2], key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    return points[order[first]]


class Grid:
    def __init__(self, xy, res, pad=3):
        self.res = float(res)
        self.x0 = np.floor(xy[:, 0].min() / res) * res - pad * res
        self.y0 = np.floor(xy[:, 1].min() / res) * res - pad * res
        self.nx = int(np.ceil((xy[:, 0].max() - self.x0) / res)) + pad + 1
        self.ny = int(np.ceil((xy[:, 1].max() - self.y0) / res)) + pad + 1

    def index(self, xy):
        ix = np.floor((xy[:, 0] - self.x0) / self.res).astype(np.int64)
        iy = np.floor((xy[:, 1] - self.y0) / self.res).astype(np.int64)
        return iy, ix

    def centres(self, iy, ix):
        return np.column_stack([self.x0 + (ix + 0.5) * self.res,
                                self.y0 + (iy + 0.5) * self.res])


def terrain_at(ground_pts, xy):
    """Terrain height at xy: linear interpolation of the ground returns,
    nearest-neighbour outside their hull (the same rule as 04)."""
    if len(ground_pts) > 400_000:
        ground_pts = thin_lowest(ground_pts, 1.0)
    linear = LinearNDInterpolator(ground_pts[:, :2], ground_pts[:, 2])
    z = linear(xy)
    missing = ~np.isfinite(z)
    if missing.any():
        z[missing] = NearestNDInterpolator(ground_pts[:, :2], ground_pts[:, 2])(xy[missing])
    return z


def neighbour_pairs(labels):
    """(a, b, n) for every 4-adjacent pair of differing labels, with the number
    of shared cell edges n. Labels are integers, 0 = background."""
    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        d = a != b
        pairs.append(np.column_stack([a[d], b[d]]))
    p = np.vstack(pairs)
    p = np.sort(p, axis=1)
    uniq, counts = np.unique(p, axis=0, return_counts=True)
    return uniq, counts


def segment_parts(mask, roof_s, res, tolerance, min_part_cells):
    """Label building parts: split at facade steps, then into height bands,
    then merge fragments. Returns an int label raster (0 = background)."""
    ny, nx = mask.shape
    idx = -np.ones(mask.shape, dtype=np.int64)
    cells = np.flatnonzero(mask.ravel())
    idx.ravel()[cells] = np.arange(len(cells))
    step = 0.5 * tolerance
    rows, cols = [], []
    for sl_a, sl_b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        a_i, b_i = idx[sl_a], idx[sl_b]
        ok = (a_i >= 0) & (b_i >= 0) & (np.abs(roof_s[sl_a] - roof_s[sl_b]) <= step)
        rows.append(a_i[ok]); cols.append(b_i[ok])
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                       shape=(len(cells), len(cells)))
    _, comp = connected_components(graph, directed=False)

    # Height bands inside any part whose range exceeds the tolerance.
    z = roof_s.ravel()[cells]
    order = np.argsort(comp, kind="stable")
    bounds = np.flatnonzero(np.r_[True, comp[order][1:] != comp[order][:-1], True])
    band = np.zeros(len(cells), dtype=np.int64)
    n_banded = 0
    for s, e in zip(bounds[:-1], bounds[1:]):
        members = order[s:e]
        lo, hi = np.percentile(z[members], [2.0, 98.0])
        if hi - lo > tolerance:
            band[members] = np.clip(np.floor((z[members] - lo) / tolerance), 0, None)
            n_banded += 1
    key = comp.astype(np.int64) * 1000 + band
    _, key_index = np.unique(key, return_inverse=True)
    # A band of one part need not be connected: split every (part, band) into
    # its 4-connected pieces. Edges are kept only between cells of the same
    # (part, band), so the graph components are exactly those pieces.
    keep = key_index[rows] == key_index[cols]
    graph = coo_matrix((np.ones(int(keep.sum()), dtype=np.int8),
                        (rows[keep], cols[keep])), shape=(len(cells), len(cells)))
    _, piece = connected_components(graph, directed=False)
    labels = np.zeros(mask.shape, dtype=np.int64)
    labels.ravel()[cells] = piece + 1

    # Merge fragments into the neighbour with the longest shared edge.
    for _ in range(50):
        sizes = np.bincount(labels.ravel(), minlength=labels.max() + 1)
        small = np.flatnonzero((sizes > 0) & (sizes < min_part_cells))
        small = small[small > 0]
        if len(small) == 0:
            break
        pairs, counts = neighbour_pairs(labels)
        keep = (pairs[:, 0] > 0) & (pairs[:, 1] > 0)
        pairs, counts = pairs[keep], counts[keep]
        target = np.arange(labels.max() + 1)
        best = np.zeros(labels.max() + 1)
        is_small = np.zeros(labels.max() + 1, dtype=bool)
        is_small[small] = True
        for (a, b), c in zip(pairs, counts):
            for s, o in ((a, b), (b, a)):
                if is_small[s] and c > best[s] and (not is_small[o] or sizes[o] > sizes[s]):
                    best[s] = c
                    target[s] = o
        changed = target != np.arange(len(target))
        if not changed.any():
            break
        # resolve chains a->b->c
        for _ in range(10):
            target = target[target]
        labels = target[labels]
    # Fragments with no neighbour at all are free-standing and too small.
    sizes = np.bincount(labels.ravel(), minlength=labels.max() + 1)
    labels[(sizes[labels] < min_part_cells)] = 0
    # Compact the label numbering.
    _, inv = np.unique(labels, return_inverse=True)
    labels = inv.reshape(labels.shape)
    return labels, n_banded


def lattice_edges(labels, grid):
    """Unit cell-boundary segments wherever the label changes (world xy)."""
    segs = []
    x0, y0, r = grid.x0, grid.y0, grid.res
    # vertical edges between (i, j) and (i, j+1)
    d = labels[:, :-1] != labels[:, 1:]
    i, j = np.nonzero(d)
    x = x0 + (j + 1) * r
    segs.append(np.stack([np.column_stack([x, y0 + i * r]),
                          np.column_stack([x, y0 + (i + 1) * r])], axis=1))
    # horizontal edges between (i, j) and (i+1, j)
    d = labels[:-1, :] != labels[1:, :]
    i, j = np.nonzero(d)
    y = y0 + (i + 1) * r
    segs.append(np.stack([np.column_stack([x0 + j * r, y]),
                          np.column_stack([x0 + (j + 1) * r, y])], axis=1))
    return np.concatenate(segs)


def part_polygons(labels, grid):
    """Exact cell-boundary polygons of every part (label > 0), as a list of
    (label, Polygon). Background faces (courtyards) are discarded."""
    segs = lattice_edges(labels, grid)
    lines = shapely.linestrings(segs)
    merged = shapely.line_merge(shapely.union_all(lines))
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(merged)))
    reps = shapely.point_on_surface(faces)
    rxy = shapely.get_coordinates(reps)
    iy, ix = grid.index(rxy)
    lab = labels[np.clip(iy, 0, labels.shape[0] - 1), np.clip(ix, 0, labels.shape[1] - 1)]
    keep = lab > 0
    return lab[keep], faces[keep]


def ring_segments(poly):
    """Segments of an oriented polygon (exterior CCW, holes CW): the solid is
    always on the LEFT of the direction of travel."""
    out = []
    for ring in [poly.exterior, *poly.interiors]:
        c = np.asarray(ring.coords)[:, :2]
        out.append(np.stack([c[:-1], c[1:]], axis=1))
    return np.concatenate(out)


def triangulate(poly):
    """earcut triangulation of a polygon with holes -> (verts2d, faces) with
    counter-clockwise faces (normal +z)."""
    rings = [np.asarray(poly.exterior.coords)[:-1, :2]]
    rings += [np.asarray(h.coords)[:-1, :2] for h in poly.interiors]
    verts = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = np.asarray(earcut.triangulate_float64(verts, ends), dtype=np.int64).reshape(-1, 3)
    if len(tri):
        a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        flip = cross < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
    return verts, tri


def build_segmented(pts, ground_pts, args):
    res = args.raster_res
    grid = Grid(pts[:, :2], res)
    iy, ix = grid.index(pts[:, :2])
    shape = (grid.ny, grid.nx)
    roof = np.full(shape, -np.inf)
    np.maximum.at(roof, (iy, ix), pts[:, 2])
    has_return = np.isfinite(roof)
    print(f"[bld] raster {grid.nx} x {grid.ny} at {res} m; {has_return.sum():,} cells with returns")

    # terrain under every cell that matters (returns plus a 2-cell margin)
    near = ndimage.binary_dilation(has_return, iterations=2)
    cy, cx = np.nonzero(near)
    terrain = np.full(shape, np.nan)
    terrain[cy, cx] = terrain_at(ground_pts, grid.centres(cy, cx))

    occ = has_return & (roof - terrain >= args.min_height)
    n_low = int((has_return & ~occ).sum())
    occ = binary_closing(occ, structure=np.ones((3, 3)), iterations=1)
    occ &= near
    min_hole = int(round(args.min_hole_area / res ** 2))
    min_part = max(1, int(round(args.min_part_area / res ** 2)))
    holes_before = binary_fill_holes(occ) & ~occ
    occ = remove_small_holes(occ, max_size=max(min_hole - 1, 0))
    occ = remove_small_objects(occ, max_size=max(min_part - 1, 0), connectivity=1)
    courtyard_cells = int((binary_fill_holes(occ) & ~occ).sum())

    # roof height in cells added by closing/hole filling: nearest measured cell
    measured = occ & has_return & (roof - terrain >= args.min_height)
    _, (ny_i, nx_i) = ndimage.distance_transform_edt(~measured, return_indices=True)
    roof_f = np.where(measured, roof, roof[ny_i, nx_i])
    roof_f[~occ] = np.nan
    roof_s = ndimage.median_filter(np.where(occ, roof_f, 0.0), size=3)
    roof_s = np.where(occ, roof_s, np.nan)

    labels, n_banded = segment_parts(occ, roof_s, res, args.height_tolerance, min_part)
    n_parts = int(labels.max())
    print(f"[bld] {n_parts:,} building parts ({n_banded} height-banded); "
          f"{courtyard_cells * res * res:,.0f} m2 of courtyard kept; "
          f"{n_low:,} low cells (< {args.min_height} m) not building")

    flat = labels.ravel()
    roof_m = np.where(measured, roof, np.nan).ravel()
    good = (flat > 0) & np.isfinite(roof_m)
    order = np.argsort(flat[good], kind="stable")
    lab_sorted = flat[good][order]
    z_sorted = roof_m[good][order]
    part_roof = np.full(n_parts + 1, np.nan)
    bounds = np.flatnonzero(np.r_[True, lab_sorted[1:] != lab_sorted[:-1], True])
    for s, e in zip(bounds[:-1], bounds[1:]):
        part_roof[lab_sorted[s]] = np.median(z_sorted[s:e])
    missing = ~np.isfinite(part_roof[1:])
    if missing.any():   # parts made only of filled cells: use their filled heights
        fill = ndimage.median(np.nan_to_num(roof_f), labels, index=np.arange(1, n_parts + 1))
        part_roof[1:][missing] = np.asarray(fill)[missing]
    dil = ndimage.grey_dilation(labels, size=(3, 3))
    terr = np.where(np.isfinite(terrain), terrain, np.nanmax(terrain))
    part_floor = np.asarray(ndimage.minimum(terr, labels=np.maximum(labels, (labels == 0) * dil),
                                            index=np.arange(1, n_parts + 1)))
    part_floor = np.r_[np.nan, part_floor - args.floor_margin]

    # outlines: exact cell boundaries, simplified as a coverage
    labs, faces = part_polygons(labels, grid)
    simplified = shapely.coverage_simplify(faces, tolerance=args.simplify_tolerance)
    polys, triangulations = [], []
    n_dropped = 0
    for lab, geom in zip(labs, simplified):
        for part in shapely.get_parts(geom):
            if part.geom_type != "Polygon" or part.area <= 0.5 * res * res \
                    or not np.isfinite(part_roof[lab]):
                n_dropped += 1
                continue
            part = shapely.orient_polygons(part)
            v2, tri = triangulate(part)
            if len(tri) == 0:
                n_dropped += 1
                continue
            polys.append((int(lab), part))
            triangulations.append((v2, tri))
    # Only parts that will be built may own a shared edge, otherwise a
    # neighbour would leave its wall open against a part that is not there.
    seg_owner = {}
    for k, (lab, poly) in enumerate(polys):
        for (x1, y1), (x2, y2) in ring_segments(poly):
            seg_owner[(round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6))] = k

    # Pass 1: every ring segment, its neighbour part (if any) and whether it is
    # an OUTER wall (open air or courtyard on its far side).
    plan = []
    outer_pts, outer_owner = [], []
    for k, (lab, poly) in enumerate(polys):
        segs = ring_segments(poly)
        neighbour = np.full(len(segs), -1)
        for s, ((x1, y1), (x2, y2)) in enumerate(segs):
            other = seg_owner.get((round(x2, 6), round(y2, 6), round(x1, 6), round(y1, 6)))
            if other is not None:
                neighbour[s] = other
        plan.append((lab, poly, segs, neighbour))
        outer = np.flatnonzero(neighbour < 0)
        for s in outer:
            p, q = segs[s]
            length = float(np.hypot(*(q - p)))
            n_samp = max(2, int(np.ceil(length / (0.5 * res))) + 1)
            t = np.linspace(0.0, 1.0, n_samp)[:, None]
            along = p + t * (q - p)
            normal = np.array([q[1] - p[1], p[0] - q[0]]) / max(length, 1e-9)
            outer_pts.append(along + 0.1 * normal)       # just outside the wall
            outer_owner.append(np.full(n_samp, len(outer_owner)))
    # Terrain beside every outer wall: the simulation's terrain mesh when given.
    outer_xy = np.vstack(outer_pts)
    outer_seg = np.concatenate(outer_owner)
    z_beside = terrain_at(ground_pts, outer_xy)
    terrain_source = "interpolated ground returns"
    if args.ground_stl:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        gmesh = trimesh.load(args.ground_stl, force="mesh", process=False)
        origins = np.column_stack([outer_xy, np.full(len(outer_xy), 1.0e5)])
        loc, ray, _ = RayMeshIntersector(gmesh).intersects_location(
            origins, np.tile([0.0, 0.0, -1.0], (len(origins), 1)), multiple_hits=False)
        z_mesh = np.full(len(outer_xy), np.nan)
        z_mesh[ray] = loc[:, 2]
        # The lower of the two, so a wall clears both surfaces.
        z_beside = np.fmin(z_beside, z_mesh)
        terrain_source = "terrain mesh and interpolated ground returns (lower of the two)"
    seg_bottom = np.full(len(outer_owner), np.inf)
    np.minimum.at(seg_bottom, outer_seg, z_beside)
    seg_bottom -= args.floor_margin

    V, F, roles = [], [], []
    nv = 0
    n_outer = 0
    n_sunken = 0
    for (lab, poly, segs, neighbour), (v2, tri) in zip(plan, triangulations):
        top = part_roof[lab]
        outer = neighbour < 0
        bottoms = seg_bottom[n_outer:n_outer + int(outer.sum())]
        n_outer += int(outer.sum())
        base = min(part_floor[lab], float(bottoms.min()) if len(bottoms) else np.inf)
        if base > top - args.min_height:
            # Roof at or below the terrain beside it (a structure against a
            # retaining wall). Keep it, with a floor below its roof.
            base = top - max(args.min_height, 1.0)
            bottoms = np.minimum(bottoms, base)
            n_sunken += 1
        n2 = len(v2)
        V.append(np.column_stack([v2, np.full(n2, top)]))
        F.append(tri + nv); roles.append(np.full(len(tri), 2)); nv += n2
        V.append(np.column_stack([v2, np.full(n2, base)]))
        F.append(tri[:, [0, 2, 1]] + nv); roles.append(np.full(len(tri), 3)); nv += n2
        low = np.empty(len(segs))
        low[outer] = bottoms
        inner = ~outer
        if inner.any():
            low[inner] = np.maximum(base, part_roof[[polys[o][0] for o in neighbour[inner]]])
        wall = low < top - 1e-6
        segs, low = segs[wall], low[wall]
        if len(segs) == 0:
            continue
        p, q = segs[:, 0], segs[:, 1]
        m = len(segs)
        quad = np.concatenate([np.column_stack([p, low]), np.column_stack([q, low]),
                               np.column_stack([q, np.full(m, top)]),
                               np.column_stack([p, np.full(m, top)])])
        a = np.arange(m) + nv
        F.append(np.concatenate([np.column_stack([a, a + m, a + 2 * m]),
                                 np.column_stack([a, a + 2 * m, a + 3 * m])]))
        roles.append(np.full(2 * m, 1))
        V.append(quad); nv += 4 * m
    vertices = np.vstack(V)
    faces_arr = np.vstack(F)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_arr, process=False)
    # Weld coincident vertices so each building block is ONE connected object
    # (its parts meet along shared edges). Faces and winding are unchanged.
    mesh.merge_vertices()

    # Fit report: modelled roof against the measured roof surface, per cell.
    cells = measured & (labels > 0)
    model_roof = part_roof[labels[cells]]
    residual = model_roof - roof[cells]
    report = {
        "model": "segmented LoD1",
        "raster_res_m": res,
        "height_tolerance_m": args.height_tolerance,
        "min_height_m": args.min_height,
        "min_hole_area_m2": args.min_hole_area,
        "min_part_area_m2": args.min_part_area,
        "simplify_tolerance_m": args.simplify_tolerance,
        "floor_margin_m": args.floor_margin,
        "n_parts": n_parts,
        "n_height_banded_parts": int(n_banded),
        "n_polygons": len(polys),
        "n_polygons_dropped": int(n_dropped),
        "n_parts_below_adjacent_terrain": int(n_sunken),
        "wall_bottom_reference": terrain_source,
        "n_faces": int(len(faces_arr)),
        "building_area_m2": float((labels > 0).sum() * res * res),
        "courtyard_area_kept_m2": float(courtyard_cells * res * res),
        "enclosed_gap_area_before_filling_m2": float(holes_before.sum() * res * res),
        "roof_residual_model_minus_measured_m": {
            "median": float(np.median(residual)),
            "mean": float(np.mean(residual)),
            "mae": float(np.mean(np.abs(residual))),
            "p10": float(np.percentile(residual, 10)),
            "p90": float(np.percentile(residual, 90)),
        },
    }
    return mesh, roles_concat(roles), report


def roles_concat(roles):
    return np.concatenate(roles) if roles else np.zeros(0, dtype=int)


# ----------------------------------------------------------------------------
# Legacy block model
# ----------------------------------------------------------------------------

def local_ground_elevation(building_xy_center, ground_pts, kdtree, k=50):
    """Mean of the k nearest ground points to the block centroid."""
    dists, idx = kdtree.query(building_xy_center, k=min(k, len(ground_pts)))
    return float(np.mean(ground_pts[idx, 2]))


def extract_footprint_polygon(cluster_pts, raster_res, simplify_tolerance):
    """Outer contour of a rasterised cluster, enclosed holes filled."""
    x, y = cluster_pts[:, 0], cluster_pts[:, 1]
    xmin, ymin = x.min(), y.min()
    nx = int(np.ceil((x.max() - xmin) / raster_res)) + 3
    ny = int(np.ceil((y.max() - ymin) / raster_res)) + 3
    grid = np.zeros((ny, nx), dtype=bool)
    col = np.clip(((x - xmin) / raster_res).astype(int) + 1, 0, nx - 1)
    row = np.clip(((y - ymin) / raster_res).astype(int) + 1, 0, ny - 1)
    grid[row, col] = True
    grid = binary_closing(grid, structure=np.ones((3, 3)), iterations=2)
    grid = binary_fill_holes(grid)
    contours = find_contours(grid.astype(float), level=0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    simplified = approximate_polygon(contour, tolerance=simplify_tolerance / raster_res)
    poly_x = simplified[:, 1] * raster_res + xmin - raster_res
    poly_y = simplified[:, 0] * raster_res + ymin - raster_res
    poly = np.column_stack([poly_x, poly_y])
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if len(poly) < 3:
        return None
    return poly


def extrude_footprint(poly_xy, z_base, z_top):
    n = len(poly_xy)
    verts2d = poly_xy.astype(np.float64)
    rings = np.array([n], dtype=np.uint32)
    tri_idx = np.asarray(earcut.triangulate_float64(verts2d, rings)).reshape(-1, 3)
    if len(tri_idx) == 0:
        return None
    floor_verts = np.column_stack([poly_xy, np.full(n, z_base)])
    roof_verts = np.column_stack([poly_xy, np.full(n, z_top)])
    all_verts = np.vstack([floor_verts, roof_verts])
    faces = []
    for tri in tri_idx:
        faces.append([tri[0], tri[2], tri[1]])
        faces.append([tri[0] + n, tri[1] + n, tri[2] + n])
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
    return trimesh.Trimesh(vertices=all_verts, faces=np.array(faces), process=True)


def build_block(pts, ground_pts, args):
    ground_kdtree = cKDTree(ground_pts[:, :2])
    labels, _ = grid_cluster_2d(pts[:, :2], cell_size=args.cell_size,
                                connect_radius_cells=args.connect_radius)
    meshes = []
    for label in sorted(set(labels)):
        cluster_pts = pts[labels == label]
        if len(cluster_pts) < args.min_cluster_points:
            continue
        poly = extract_footprint_polygon(cluster_pts, args.raster_res, args.simplify_tolerance)
        if poly is None:
            continue
        z_base = local_ground_elevation(cluster_pts[:, :2].mean(axis=0), ground_pts, ground_kdtree)
        z_top = float(np.percentile(cluster_pts[:, 2], args.roof_percentile))
        if z_top <= z_base:
            continue
        mesh = extrude_footprint(poly, z_base, z_top)
        if mesh is not None:
            meshes.append(mesh)
    return trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()


def main():
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.load(args.input)
    ground_pts = np.load(args.ground_npy)
    print(f"[bld] Loaded {len(pts):,} building points, {len(ground_pts):,} ground points; "
          f"model: {args.lod1_model}")
    if len(pts) == 0:
        trimesh.Trimesh().export(str(out_path))
        print(f"[bld_result] n_buildings=0 total_faces=0 output={out_path}")
        return
    if args.lod1_model == "block":
        mesh = build_block(pts, ground_pts, args)
        mesh.export(str(out_path))
        print(f"[bld_result] model=block total_faces={len(mesh.faces)} output={out_path}")
        return
    mesh, roles, report = build_segmented(pts, ground_pts, args)
    mesh.export(str(out_path))
    report["roles_face_counts"] = {"wall": int((roles == 1).sum()),
                                   "roof": int((roles == 2).sum()),
                                   "floor": int((roles == 3).sum())}
    report_path = Path(args.report) if args.report else out_path.with_name(
        out_path.stem + "_lod1_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    r = report["roof_residual_model_minus_measured_m"]
    print(f"[bld] roof fit (model - measured, per cell): median {r['median']:+.2f} m, "
          f"MAE {r['mae']:.2f} m, p10..p90 {r['p10']:+.2f}..{r['p90']:+.2f} m")
    print(f"[bld_result] model=segmented n_parts={report['n_parts']} "
          f"total_faces={len(mesh.faces)} output={out_path}")


if __name__ == "__main__":
    main()
