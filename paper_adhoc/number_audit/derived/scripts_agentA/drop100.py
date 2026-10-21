import numpy as np, scipy.sparse as sp, pandas as pd
R="/media/harshin/data_drive/route based UTCI/run_output"
SIG=5.670374419e-8; den=4*SIG*300**3; ak,ep=0.70,0.97
res={}
for c in ['lisbon1','lisbon2','lisbon3','lisbon4','lisbon5','lisbon6','MMC']:
    t=f"{R}/{c}/thermal_out"; m=f"{R}/{c}/mrt_facet_out"
    W=sp.load_npz(f"{t}/lw_view_matrix.npz").tocsr(); Wc=W.tocoo()
    ci=np.load(f"{t}/coarse_index.npy"); pm=np.load(f"{t}/point_map.npy")
    xyz=np.load(f"{m}/receptor_xyz.npy")
    F=np.load(f"{t}/facets.npz"); cen=F["centroid"]; nrm=F["normal"]
    d=np.linalg.norm(cen[Wc.col]-xyz[ci][Wc.row],axis=1)
    far=d>100
    Wfar=sp.csr_matrix((Wc.data*far,(Wc.row,Wc.col)),shape=W.shape)
    wfar=np.asarray(Wfar.sum(1)).ravel()
    J=np.load(f"{t}/facet_radiosity_matrix_Wm2.npy"); Jenv=np.load(f"{t}/radiosity_environment_Wm2.npy")
    Jsw=np.load(f"{t}/sw_radiosity_environment_Wm2.npy")
    alb=np.load(f"{t}/facet_albedo.npy"); fsky=np.load(f"{t}/f_sky_facet.npy"); tau=np.load(f"{t}/tau_dir_facet.npy",mmap_mode='r')
    tm=pd.read_csv(f"{m}/times.csv")
    day=tm.elevation_deg.values>0
    dl=[];ds=[]
    for it in np.where(day)[0]:
        lw=wfar*Jenv[it]-Wfar@J[it].astype(float)
        az=np.deg2rad(tm.azimuth_deg[it]); el=np.deg2rad(tm.elevation_deg[it])
        sv=np.array([np.sin(az)*np.cos(el),np.cos(az)*np.cos(el),np.sin(el)])
        inc=fsky*tm.DHI_Wm2[it]+(1-fsky)*Jsw[it]+tau[it]*tm.DNI_Wm2[it]*np.clip(nrm@sv,0,None)
        sw=wfar*Jsw[it]-Wfar@(alb*inc)
        dl.append(lw[pm]); ds.append(sw[pm])
    dl=np.array(dl); ds=np.array(ds)
    for name,dT in (("LW+SW",(dl+ak/ep*ds)/den),):
        a=np.abs(dT); res[c]=dT; print(c,name,"p95abs %.3f p95signed %.3f p5 %.3f max %.3f"%(np.percentile(a,95),np.percentile(dT,95),np.percentile(dT,5),a.max()))
allv=np.concatenate([np.abs(v).ravel() for k,v in res.items()]); print("pooled p95abs %.3f"%np.percentile(allv,95)); al=np.concatenate([np.abs(v).ravel() for k,v in res.items() if k!="MMC"]); print("lisbon pooled %.3f"%np.percentile(al,95))
