import numpy as np, scipy.sparse as sp
for c in ['lisbon2','lisbon3','lisbon4','lisbon5','lisbon6','MMC']:
    th=f'{c}/thermal_out'
    W=sp.load_npz(th+'/lw_view_matrix.npz').tocoo(); f=np.load(th+'/facets.npz')
    ci=np.load(th+'/coarse_index.npy'); xyz=np.load(f'{c}/mrt_facet_out/path_xyz.npy')[ci]
    s=np.einsum('ij,ij->i', xyz[W.row]-f['centroid'][W.col], f['normal'][W.col])
    pw=np.load(th+'/lw_point_weights.npz')
    bad=(s<0)&(f['mesh_id'][W.col]==0)
    per=np.bincount(W.row[bad],weights=W.data[bad],minlength=W.shape[0])
    out=pw['w_sky']>0.01
    p=per[out]
    print(c,'outdoor coarse pts: flipped-facet weight (full-sphere) mean %.3f, p90 %.3f, max %.3f, pts with >0.05: %.1f%%'%(p.mean(),np.percentile(p,90),p.max(),100*(p>0.05).mean()))
    # temperature of flipped facets vs other walls at 15:00
    T=np.load(th+'/facet_T_matrix_K.npy', mmap_mode='r')[90]-273.15
    nb=np.unique(W.col[bad]); walls=np.setdiff1d(np.where(f['cls']==1)[0],nb)
    print('    15:00 T flipped facets mean %.1f C vs other walls %.1f C'%(T[nb].mean(),T[walls].mean()))
