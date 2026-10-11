import numpy as np, scipy.sparse as sp
for c in ['lisbon1','lisbon2','lisbon3','lisbon4','lisbon5','lisbon6','MMC']:
    th=f'{c}/thermal_out'
    W=sp.load_npz(th+'/lw_view_matrix.npz').tocoo(); f=np.load(th+'/facets.npz')
    ci=np.load(th+'/coarse_index.npy'); xyz=np.load(f'{c}/mrt_facet_out/path_xyz.npy')[ci]
    s=np.einsum('ij,ij->i', xyz[W.row]-f['centroid'][W.col], f['normal'][W.col])
    back=s<0
    bld=f['mesh_id'][W.col]==0
    fsky=np.load(th+'/f_sky_facet.npy')
    # outside points: points with sky view > 0
    pw=np.load(th+'/lw_point_weights.npz'); outside=pw['w_sky'][W.row]>0.01
    bad=back&bld&outside
    nb=np.unique(W.col[bad])
    tot_bld_w=W.data[bld&outside].sum()
    print(c,'W weight on building facets seen from their BACK side by open-sky points: %.2f%% of building-facet weight; facets %d (f_sky mean %.2f vs all-wall mean %.2f)'%(100*W.data[bad].sum()/tot_bld_w, len(nb), fsky[nb].mean() if len(nb) else np.nan, fsky[f['cls']==1].mean()))
