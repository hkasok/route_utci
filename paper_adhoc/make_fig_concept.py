"""Generate fig_concept.tex (paper Figure 1a): the route-first idea on a
synthetic district.

The highlighting is computed, not drawn by hand: from route points every
2 m at 1.1 m height the same 24 x 18 full-sphere ray set as stage 05a
(thermal_common.make_sphere_directions) is cast to 300 m against the whole
scene, and every building or ground triangle hit first is a solved facet.
Crowns are occluders (rays stopping there are vegetation, not facets).

Units: 1 unit = 10 m; a 3 x 3-block district.  Output: TikZ, oblique view, painter's order.

Usage: python -I make_fig_concept.py OUT.tex
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thermal_common import make_sphere_directions  # noqa: E402

RNG = np.random.default_rng(7)
G = 0.4                    # grid: tiles, gaps and footprints are multiples
H_PED = 0.11               # 1.1 m
STEP = 0.2                 # route point spacing, 2 m (every 8th of 0.25 m)
RANGE = 30.0               # 300 m

# projection used in the TikZ (x, y, z unit vectors on the page, cm)
SCALE = 1.00              # natural width about the paper's text width
PX = SCALE * np.array([[0.60, 0.26, 0.00],
                       [-0.10, 0.42, 0.30]])
# direction towards the viewer: null space of PX, with +z component
_n = np.cross(PX[0], PX[1]); VIEW = _n / np.linalg.norm(_n)
if VIEW[2] < 0:
    VIEW = -VIEW

# ------------------------------------------------------------- scene -----
BW, BD, ST = 3.6, 2.4, 1.2          # block width, depth, street width
NCOL, NROW = 3, 3
X0, Y0 = -0.4, -0.4                 # domain origin (edge strip)
XMAX = NCOL * (BW + ST) - ST + 0.4
YMAX = NROW * (BD + ST) - ST + 0.4


def block_origin(i, j):
    return i * (BW + ST), j * (BD + ST)


buildings = []        # (x0, x1, y0, y1, h)
trees = []            # (x, y, crown radius, crown centre height)
PARK = {(1, 1)}       # open block with trees, inside the route's bend

for i in range(NCOL):
    for j in range(NROW):
        bx, by = block_origin(i, j)
        if (i, j) in PARK:
            for tx in np.arange(bx + 0.7, bx + BW - 0.3, 1.1):
                for ty in np.arange(by + 0.6, by + BD - 0.3, 1.2):
                    trees.append((tx + RNG.uniform(-.1, .1),
                                  ty + RNG.uniform(-.1, .1), 0.36, 0.80))
            continue
        # 2 x 2 buildings with a 0.4 gap
        xs = [bx, bx + 1.6, bx + 2.0, bx + BW]
        ys = [by, by + 1.0, by + 1.4, by + BD]
        for a in range(2):
            for b in range(2):
                h = float(RNG.choice([0.8, 1.2, 1.6, 2.0]))
                buildings.append((xs[2 * a], xs[2 * a + 1],
                                  ys[2 * b], ys[2 * b + 1], h))

# one tall building in the far row
bx, by = block_origin(0, 2)
buildings = [b for b in buildings
             if not (b[0] == bx + 2.0 and b[2] == by + 1.4)]
buildings.append((bx + 2.0, bx + BW, by + 1.4, by + BD, 3.2))

# street trees along the last leg
for tx in np.arange(9.4, 12.8, 1.1):
    trees.append((tx, 6.0 + 0.25, 0.30, 0.72))

# route: east on the y=3.0 street, north on x=4.2, east on y=6.6
ROUTE = np.array([[0.2, 3.15], [4.35, 3.15], [4.35, 6.75], [13.0, 6.75]])


def route_points():
    pts = []
    for a, b in zip(ROUTE[:-1], ROUTE[1:]):
        n = int(np.ceil(np.linalg.norm(b - a) / STEP))
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append(a + t * (b - a))
    pts.append(ROUTE[-1])
    p = np.asarray(pts)
    return np.column_stack([p, np.full(len(p), H_PED)])


# ------------------------------------------------- triangulated surfaces --
tris = []      # list of 3x3 arrays
kind = []      # 'ground' | 'wall' | 'roof' | 'veg'
owner = []     # building index, -1 ground, -2 - tree index
face_of = []   # (owner, face name) for walls/roofs, ground tile id


def add_quad(p0, p1, p2, p3, k, own, face):
    for tri in ((p0, p1, p2), (p0, p2, p3)):
        tris.append(np.array(tri, float)); kind.append(k)
        owner.append(own); face_of.append(face)


def inside_footprint(x, y):
    for (x0, x1, y0, y1, _) in buildings:
        if x0 - 1e-9 <= x <= x1 + 1e-9 and y0 - 1e-9 <= y <= y1 + 1e-9:
            return True
    return False


# ground tiles (G x G) not under buildings
for x in np.arange(X0, XMAX - 1e-9, G):
    for y in np.arange(Y0, YMAX - 1e-9, G):
        if inside_footprint(x + G / 2, y + G / 2):
            continue
        add_quad((x, y, 0), (x + G, y, 0), (x + G, y + G, 0), (x, y + G, 0),
                 'ground', -1, ('g', round(x, 2), round(y, 2)))


def wall_panels(a, b, h, normal):
    """Split the vertical rectangle from a to b (ground) up to h."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    L = np.linalg.norm(b - a)
    nu = max(1, int(round(L / 0.8)))
    nv = max(1, int(np.ceil(h / 0.6)))
    us = np.linspace(0, 1, nu + 1); vs = np.linspace(0, h, nv + 1)
    out = []
    for iu in range(nu):
        for iv in range(nv):
            p, q = a + us[iu] * (b - a), a + us[iu + 1] * (b - a)
            out.append(((p[0], p[1], vs[iv]), (q[0], q[1], vs[iv]),
                        (q[0], q[1], vs[iv + 1]), (p[0], p[1], vs[iv + 1])))
    return out


for k, (x0, x1, y0, y1, h) in enumerate(buildings):
    faces = {'S': ((x0, y0), (x1, y0)), 'E': ((x1, y0), (x1, y1)),
             'N': ((x1, y1), (x0, y1)), 'W': ((x0, y1), (x0, y0))}
    for name, (a, b) in faces.items():
        for quad in wall_panels(a, b, h, name):
            add_quad(*quad, 'wall', k, (k, name))
    add_quad((x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h),
             'roof', k, (k, 'T'))

for t, (tx, ty, r, zc) in enumerate(trees):
    s = trimesh.creation.icosphere(subdivisions=2, radius=r)
    for f in s.faces:
        tris.append(s.vertices[f] + [tx, ty, zc]); kind.append('veg')
        owner.append(-2 - t); face_of.append(('veg', t))

T = np.array(tris)
kind = np.array(kind); owner = np.array(owner)
mesh = trimesh.Trimesh(vertices=T.reshape(-1, 3),
                       faces=np.arange(len(T) * 3).reshape(-1, 3),
                       process=False)
ray = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

# ------------------------------------------------------- selection -------
P = route_points()
dirs, _ = make_sphere_directions(24, 18, body="cylinder")
orig = np.repeat(P, len(dirs), axis=0)
dvec = np.tile(dirs, (len(P), 1))
idx_tri, idx_ray, locs = ray.intersects_id(orig, dvec, multiple_hits=False,
                                           return_locations=True)
dist = np.linalg.norm(locs - orig[idx_ray], axis=1)
ok = dist <= RANGE
hit_tri = np.full(len(orig), -1)
hit_loc = np.full((len(orig), 3), np.nan)
hit_tri[idx_ray[ok]] = idx_tri[ok]
hit_loc[idx_ray[ok]] = locs[ok]
selected = np.zeros(len(T), bool)
sel_ids = hit_tri[(hit_tri >= 0)]
selected[sel_ids[kind[sel_ids] != 'veg']] = True
n_sky = int((hit_tri < 0).sum())
print(f"route points {len(P)}, rays {len(orig):,}, "
      f"solved triangles {selected.sum()} of {(kind != 'veg').sum()}, "
      f"sky rays {n_sky:,}")

# ------------------------------------------------------- drawing ---------
def depth(p):
    return float(np.dot(p, VIEW))


def visible_from_viewer(p):
    """True if nothing lies between scene point p and the viewer."""
    o = np.asarray(p, float) + VIEW * 1e-3
    return not ray.intersects_any(o[None], VIEW[None])[0]


def c(p):
    return f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"


out = []
w = out.append
w("% Figure 1a (route-first selection on a synthetic district); generated, do not edit by hand.")
w("% Solved facets are the triangles hit first by the 24x18 facet-selection ray set"
  " cast from route points every 2 m.")
w("\\begin{tikzpicture}[")
w(f"    x={{({PX[0,0]:.2f}cm,{PX[1,0]:.2f}cm)}}, "
  f"y={{({PX[0,1]:.2f}cm,{PX[1,1]:.2f}cm)}}, "
  f"z={{({PX[0,2]:.2f}cm,{PX[1,2]:.2f}cm)}},")
w("    font=\\footnotesize, line join=round,")
w("    g/.style={fill=black!5}, gs/.style={fill=cfSel!55!white},")
w("    wf/.style={fill=black!22}, ws/.style={fill=black!34},"
  " rf/.style={fill=black!9},")
w("    wfs/.style={fill=cfSel!80!white}, wss/.style={fill=cfSel!85!black},")
w("    mesh/.style={draw=white, line width=0.18pt},"
  " edge/.style={draw=black!60, line width=0.35pt},")
w("  ]")
w("  \\definecolor{cfRoute}{RGB}{0,105,95}")
w("  \\definecolor{cfRay}{RGB}{0,114,178}")
w("  \\definecolor{cfSel}{RGB}{230,159,0}")
w("  \\definecolor{cfTree}{RGB}{118,140,104}")

# ground: base and solved triangles
gidx = np.where(kind == "ground")[0]
w("  % ground")
w(f"  \\fill[g] ({X0},{Y0},0)--({XMAX},{Y0},0)--({XMAX},{YMAX},0)"
  f"--({X0},{YMAX},0)--cycle;")
for k in gidx:
    if selected[k]:
        w(f"  \\filldraw[gs, mesh] {c(T[k][0])}--{c(T[k][1])}--{c(T[k][2])}--cycle;")
# route line on the ground, under buildings and trees
w("  \\draw[cfRoute, line width=1.6pt, line cap=round, line join=round] "
  + "--".join(f"({x:.2f},{y:.2f},0)" for x, y in ROUTE) + ";")

# objects in painter's order
items = []
for k, (x0, x1, y0, y1, h) in enumerate(buildings):
    items.append((depth([(x0 + x1) / 2, (y0 + y1) / 2, 0]), 'b', k))
for t, (tx, ty, r, zc) in enumerate(trees):
    items.append((depth([tx, ty, 0]), 't', t))
items.sort()

FACE_NORMAL = {'S': (0, -1, 0), 'E': (1, 0, 0), 'N': (0, 1, 0),
               'W': (-1, 0, 0), 'T': (0, 0, 1)}
face_tris = {}
for k in np.where((kind == 'wall') | (kind == 'roof'))[0]:
    face_tris.setdefault(face_of[k], []).append(k)

w("  % buildings and trees, far to near")
for _, typ, k in items:
    if typ == 't':
        tx, ty, r, zc = trees[k]
        w(f"  \\draw[black!55, line width=0.8pt] ({tx:.2f},{ty:.2f},0)"
          f"--({tx:.2f},{ty:.2f},{zc - r * 0.6:.2f});")
        w(f"  \\filldraw[draw=black!60, fill=cfTree, line width=0.3pt] "
          f"({tx:.2f},{ty:.2f},{zc:.2f}) circle[radius={r * 0.55 * SCALE:.2f}cm];")
        continue
    x0, x1, y0, y1, h = buildings[k]
    for name in ('S', 'E', 'T', 'W', 'N'):
        if np.dot(FACE_NORMAL[name], VIEW) <= 0:
            continue
        base = {'S': 'wf', 'E': 'ws', 'W': 'ws', 'N': 'wf', 'T': 'rf'}[name]
        sel_style = {'S': 'wfs', 'E': 'wss', 'W': 'wss', 'N': 'wfs',
                     'T': 'wfs'}[name]
        for t in face_tris[(k, name)]:
            st = sel_style if selected[t] else base
            if name == 'T' and not selected[t]:
                w(f"  \\fill[{st}] {c(T[t][0])}--{c(T[t][1])}--{c(T[t][2])}--cycle;")
            else:
                w(f"  \\filldraw[{st}, mesh] {c(T[t][0])}--{c(T[t][1])}"
                  f"--{c(T[t][2])}--cycle;")
        corners = {'S': [(x0, y0), (x1, y0)], 'E': [(x1, y0), (x1, y1)],
                   'W': [(x0, y1), (x0, y0)], 'N': [(x1, y1), (x0, y1)]}
        if name == 'T':
            w(f"  \\draw[edge] ({x0},{y0},{h})--({x1},{y0},{h})--({x1},{y1},{h})"
              f"--({x0},{y1},{h})--cycle;")
        else:
            (ax, ay), (bx_, by_) = corners[name]
            w(f"  \\draw[edge] ({ax},{ay},0)--({bx_},{by_},0)--({bx_},{by_},{h})"
              f"--({ax},{ay},{h})--cycle;")

# --------------------------------------------- reverse rays (sampled) ----
# where the long sky rays end (page cm): the empty corner above the far
# right of the ground, below the top of the tallest building
_tops = np.array([[x, y, h] for (x0, x1, y0, y1, h) in buildings
                  for x in (x0, x1) for y in (y0, y1)])
_ground = np.array([[X0, Y0, 0], [XMAX, Y0, 0], [X0, YMAX, 0], [XMAX, YMAX, 0]])
_scr = np.vstack([_tops, _ground]) @ PX.T
SKY_LABEL = np.array([_scr[:, 0].max() - 0.9, _scr[:, 1].max() - 0.45])
SKY_OFFSET = [np.array([-0.35, -0.10]), np.array([0.15, -0.30])]
FAN_AT = [np.array([4.35, 4.9, H_PED]), np.array([7.0, 6.75, H_PED])]
w("  % reverse rays from two route points (subset of the 432 per point)")
fan_rays = []
for p in FAN_AT:
    ip = int(np.argmin(np.linalg.norm(P - p, axis=1)))
    p = P[ip]
    rows = range(ip * len(dirs), (ip + 1) * len(dirs))
    cand = []
    for r in rows:
        d = dvec[r]
        if hit_tri[r] < 0:
            continue
        q = hit_loc[r]
        # end only on faces turned towards the viewer, so the dot is seen
        if np.dot(mesh.face_normals[hit_tri[r]], VIEW) <= 0.05 and \
                kind[hit_tri[r]] != 'ground':
            continue
        L = np.linalg.norm(q - p)
        if L < 0.6 or L > 9:
            continue
        samples = [p + s * (q - p) for s in np.linspace(0.05, 0.97, 12)]
        if not all(visible_from_viewer(s) for s in samples):
            continue
        cand.append((np.degrees(np.arctan2(d[1], d[0])), r, q, kind[hit_tri[r]]))
    # spread over azimuth, prefer walls, keep one crown hit
    chosen, used = [], []
    for want in ('wall', 'ground', 'veg'):
        for az, r, q, kd in sorted(cand, key=lambda z: z[0]):
            if kd != want:
                continue
            if any(abs(((az - u + 180) % 360) - 180) < 22 for u in used):
                continue
            chosen.append((r, q, kd)); used.append(az)
            if want == 'veg' or len([1 for x in chosen if x[2] == want]) >= (4 if want == 'wall' else 3):
                break
    # one sky ray, drawn long: the escaping ray whose image points best at
    # the empty top-right corner of the figure, where "sky" is written
    ps = PX @ p
    aim = SKY_LABEL + SKY_OFFSET[len(fan_rays)] - ps
    best = None
    for r in rows:
        if hit_tri[r] >= 0 or dvec[r][2] < 0.15:
            continue
        u = PX @ dvec[r]
        ang = np.arccos(np.clip(u @ aim / np.linalg.norm(u) / np.linalg.norm(aim), -1, 1))
        if best is None or ang < best[0]:
            best = (ang, r, np.linalg.norm(aim) / np.linalg.norm(u))
    sky = [(best[1], best[2])]
    fan_rays.append((p, chosen, sky))
    print(f"fan at {p[:2]}: {[(x[2]) for x in chosen]}, sky ray off by "
          f"{np.degrees(best[0]):.1f} deg on the page")

for n_fan, (p, chosen, sky) in enumerate(fan_rays):
    for r, q, kd in chosen:
        w(f"  \\draw[cfRay, line width=0.45pt] {c(p)}--{c(q)};")
        mark = 'fill=cfRay' if kd != 'veg' else 'fill=white, draw=cfRay'
        w(f"  \\filldraw[{mark}, line width=0.3pt] {c(q)} circle[radius=0.045cm];")
    for r, L in sky:
        q = p + L * dvec[r]
        w(f"  \\draw[cfRay, line width=0.45pt, -{{Stealth[length=4pt]}}] "
          f"{c(p)}--{c(q)};")
    w(f"  \\filldraw[fill=white, draw=cfRoute, line width=0.6pt] {c(p)} "
      f"circle[radius=0.06cm];")

w(f"  \\node[anchor=west, text=cfRay, font=\\footnotesize\\itshape, inner sep=1pt] "
  f"at ({SKY_LABEL[0] + 0.3:.2f}cm,{SKY_LABEL[1] - 0.1:.2f}cm) {{sky}};")

# ---------------------------------------------------- walkers, sun -------
w("""  \\def\\cfWalker#1{%
    \\begin{scope}[shift={#1}, x=0.8cm, y=0.8cm]
      \\filldraw[draw=white, fill=cfRoute, line width=0.6pt] (0,0.2) circle[radius=0.2];
      \\draw[white, line width=0.8pt, line cap=round]
        (0,0.25)--(0,0.12) (-0.07,0.03)--(0,0.12)--(0.07,0.03)
        (-0.07,0.19)--(0,0.23)--(0.07,0.17);
      \\fill[white] (0,0.32) circle[radius=0.035];
    \\end{scope}}""")
s0, s1 = ROUTE[0], ROUTE[-1]
w(f"  \\cfWalker{{({s0[0]},{s0[1]},0)}}")
w(f"  \\node[fill=cfRoute, text=white, font=\\scriptsize\\bfseries, "
  f"inner sep=2pt, anchor=east] at ($({s0[0]},{s0[1]},0)+(-0.25,0.2)$) {{START}};")
w(f"  \\cfWalker{{({s1[0]},{s1[1]},0)}}")
w(f"  \\node[fill=cfRoute, text=white, font=\\scriptsize\\bfseries, "
  f"inner sep=2pt, anchor=west] at ($({s1[0]},{s1[1]},0)+(0.25,0.2)$) {{END}};")

# legend: two rows under the ground, in page centimetres
corners = np.array([[X0, Y0, 0], [XMAX, Y0, 0], [X0, YMAX, 0], [XMAX, YMAX, 0]])
scr = corners @ PX.T
lx, rx, by_ = scr[:, 0].min(), scr[:, 0].max(), scr[:, 1].min()
cw = (rx - lx) / 3.0
entries = [
    ("\\draw[cfRoute, line width=1.6pt, line cap=round] (0,0)--(0.45,0);",
     "pedestrian route"),
    ("\\draw[cfRay, line width=0.45pt] (0,0)--(0.40,0); "
     "\\filldraw[fill=cfRay, line width=0.3pt] (0.43,0) circle[radius=0.045cm];",
     "reverse ray, first hit"),
    ("\\filldraw[draw=black!60, fill=cfTree] (0.22,0) circle[radius=0.11];",
     "tree crown (occluder)"),
    ("\\filldraw[draw=black!60, fill=cfSel!80!white] (0,-0.11) rectangle (0.45,0.11);",
     "solved facet (ray hit)"),
    ("\\filldraw[draw=black!60, fill=black!22] (0,-0.11) rectangle (0.45,0.11);",
     "occluder only, not solved"),
]
w("  % legend")
w("  \\begin{scope}[x=1cm, y=1cm]")
for n, (sym, text) in enumerate(entries):
    col, row = n % 3, n // 3
    ox, oy = lx + 0.1 + (0.0, 4.0, 7.7)[col], by_ - 0.22 - 0.38 * row
    w(f"    \\begin{{scope}}[shift={{({ox:.2f},{oy:.2f})}}] {sym} "
      f"\\node[anchor=west, inner sep=1pt, font=\\scriptsize] at (0.55,0) {{{text}}}; \\end{{scope}}")
w("  \\end{scope}")
w("\\end{tikzpicture}")

Path(sys.argv[1]).write_text("\n".join(out) + "\n")
print(f"wrote {sys.argv[1]}: {len(out)} lines")
