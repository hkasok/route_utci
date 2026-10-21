import sys, os, numpy as np, importlib.util
P="/media/harshin/data_drive/route based UTCI"; sys.path.insert(0,P); os.chdir(P)
from scipy.spatial import ConvexHull
from ground_height_filter import build_ground_height_grid, height_above_ground
from grid_cluster import grid_cluster_2d
from tree_crown_segmentation import segment_tree_crowns, should_segment
from vertical_stratification import split_vertical_strata
spec=importlib.util.spec_from_file_location("veg","02_vegetation_to_stl.py"); veg=importlib.util.module_from_spec(spec)
sv,sys.argv=sys.argv,["x"]
try: spec.loader.exec_module(veg)
except SystemExit: pass
sys.argv=sv
d=sys.argv[1] if len(sys.argv)>1 else "out_crop_test/00_split"
pts=np.load(f"{d}/vegetation_points.npy"); gnd=np.load(f"{d}/ground_and_water_points.npy")
grid=build_ground_height_grid(gnd,cell_size=2.0); pts=pts[height_above_ground(pts,grid)>=1.5]
labels,_=grid_cluster_2d(pts[:,:2],cell_size=0.5,connect_radius_cells=1)
crowns=[]
for c in np.unique(labels):
    cl=pts[labels==c]; subs=[cl]
    if should_segment(cl,0.5,300,8.0):
        lab,_=segment_tree_crowns(cl,chm_res=0.25,smooth_sigma=2.0,min_tree_distance_m=3.0,min_fragment_points=50)
        subs=[cl[lab==k] for k in np.unique(lab)]
    for s in subs: crowns+=[x.points for x in split_vertical_strata(s)[0]]
def aspect(c): return np.ptp(c[:,2])/max(max(np.ptp(c[:,0]),np.ptp(c[:,1])),1e-6)
sel=[c for c in crowns if 1200<=len(c)<=6000 and 0.6<=aspect(c)<=1.6 and max(np.ptp(c[:,0]),np.ptp(c[:,1]))>=6.0]
sel.sort(key=len); p=sel[-1]
print(f"figure crown: {len(p)} points, {np.ptp(p[:,0]):.2f} x {np.ptp(p[:,1]):.2f} x {np.ptp(p[:,2]):.2f} m; candidates {len(sel)} of {len(crowns)}",flush=True)
ratios=[]
for c in crowns:
    if len(c)<10: continue
    try: axy,z,rad,_=veg.crown_radial_profile(c)
    except Exception: continue
    n=rad.shape[1]; az=2*np.pi*np.arange(n)/n
    for row in rad:
        xy=np.c_[row*np.cos(az),row*np.sin(az)]
        a=0.5*abs(np.dot(xy[:,0],np.roll(xy[:,1],-1))-np.dot(xy[:,1],np.roll(xy[:,0],-1)))
        ratios.append(a/ConvexHull(xy).volume)
r=np.array(ratios)
print(f"star-hull cross-sections: {len(r)} from {len(crowns)} crowns; share with area >= 95% of convex hull: {100*(r>=0.95).mean():.2f}%")
