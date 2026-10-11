import numpy as np, scipy.sparse as sp
for case in ["lisbon1","lisbon2","lisbon3","lisbon4","lisbon5","lisbon6","MMC"]:
    d=f"{case}/thermal_out"; m=f"{case}/mrt_facet_out"
    W=sp.load_npz(d+"/lw_view_matrix.npz"); f=np.load(d+"/facets.npz"); pw=np.load(d+"/lw_point_weights.npz"); pm=np.load(d+"/point_map.npy")
    cls=f["cls"]
    gw=np.asarray(W[:,cls==0].sum(1)).ravel()
    inside=(gw<0.05)&(pw['w_sky']<1e-3)&(pw['w_veg']<0.1)
    ins_full=inside[pm]
    T=np.load(m+"/tmrt_matrix_C.npy")
    for it in (78,54,132):
        print(case,"it",it,"inside-building pts %.1f%%: Tmrt %.1f vs others median %.1f"%(100*ins_full.mean(), np.median(T[it,ins_full]) if ins_full.any() else np.nan, np.median(T[it,~ins_full])))
