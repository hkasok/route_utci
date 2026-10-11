import numpy as np, scipy.sparse as sp, sys
for case in ["lisbon1","lisbon2","lisbon3","lisbon4","lisbon5","lisbon6","MMC"]:
    d=f"{case}/thermal_out"
    try:
        W=sp.load_npz(d+"/lw_view_matrix.npz"); f=np.load(d+"/facets.npz"); pw=np.load(d+"/lw_point_weights.npz")
    except Exception as e:
        print(case,e); continue
    cls=f["cls"]; cen=f["centroid"]
    gw=np.asarray(W[:,cls==0].sum(1)).ravel()
    ww=np.asarray(W[:,cls==1].sum(1)).ravel()
    rw=np.asarray(W[:,cls==2].sum(1)).ravel()
    print(case, "n",W.shape, "ground w mean %.3f p10 %.3f min %.3f | wall %.3f roof %.3f sky %.3f veg %.3f def %.4f"%(gw.mean(),np.percentile(gw,10),gw.min(),ww.mean(),rw.mean(),pw["w_sky"].mean(),pw["w_veg"].mean(),pw["w_default"].mean()))
