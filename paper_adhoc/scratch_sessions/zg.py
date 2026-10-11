import numpy as np, scipy.sparse as sp
for case in ["lisbon2","lisbon3","MMC"]:
    d=f"{case}/thermal_out"; m=f"{case}/mrt_facet_out"
    W=sp.load_npz(d+"/lw_view_matrix.npz"); f=np.load(d+"/facets.npz"); pw=np.load(d+"/lw_point_weights.npz")
    cls=f["cls"]
    gw=np.asarray(W[:,cls==0].sum(1)).ravel(); ww=np.asarray(W[:,cls==1].sum(1)).ravel(); rw=np.asarray(W[:,cls==2].sum(1)).ravel()
    sel=np.where(gw<0.05)[0]
    print(case,len(sel),"of",len(gw))
    ci=np.load(d+"/coarse_index.npy"); xyz=np.load(m+"/path_xyz.npy")
    svf=np.load(m+"/svf_planar.npy")
    for i in sel[:8]:
        print("  gw %.3f wall %.3f roof %.3f veg %.3f sky %.3f def %.3f svf_planar %.3f xyz %s"%(gw[i],ww[i],rw[i],pw['w_veg'][i],pw['w_sky'][i],pw['w_default'][i],svf[ci[i]],np.round(xyz[ci[i]],1)))
