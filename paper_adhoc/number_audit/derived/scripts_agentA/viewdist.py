import numpy as np, scipy.sparse as sp
R="/media/harshin/data_drive/route based UTCI/run_output"
for c in ['lisbon1','lisbon2','lisbon3','lisbon4','lisbon5','lisbon6','MMC']:
    t=f"{R}/{c}/thermal_out"; m=f"{R}/{c}/mrt_facet_out"
    W=sp.load_npz(f"{t}/lw_view_matrix.npz").tocoo()
    pw=np.load(f"{t}/lw_point_weights.npz")
    ci=np.load(f"{t}/coarse_index.npy")
    try: xyz=np.load(f"{m}/receptor_xyz.npy")
    except: xyz=np.load(f"{m}/path_xyz.npy")
    xyz0=np.load(f"{m}/path_xyz.npy")
    cen=np.load(f"{t}/facets.npz")["centroid"]
    out=[]
    for P in (xyz,):
        d=np.linalg.norm(cen[W.col]-P[ci][W.row],axis=1)
        wsurf=np.asarray(sp.csr_matrix(W).sum(1)).ravel()
        tot_pooled=W.data.sum(); tot_pooled_def=tot_pooled+pw['w_default'].sum()
        r=[]
        for lo,hi in ((0,5),(50,1e9),(100,1e9)):
            sel=(d>=lo)&(d<hi)
            pooled=W.data[sel].sum()/tot_pooled
            pooled_d=(W.data[sel].sum()+(pw['w_default'].sum() if lo>=50 else 0))/tot_pooled_def
            per=np.bincount(W.row[sel],W.data[sel],minlength=len(ci))
            permean=np.mean(per[wsurf>0]/wsurf[wsurf>0])
            r.append((lo,round(pooled*100,2),round(pooled_d*100,2),round(permean*100,2)))
        print(c, r, 'wdef mean',pw['w_default'].mean(), 'n',len(ci))
