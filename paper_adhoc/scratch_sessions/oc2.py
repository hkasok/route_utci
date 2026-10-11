import pandas as pd, numpy as np, json, scipy.sparse as sp, sys
sys.path.insert(0,'.')
from black_globe import resolve_globe_spec
spec=resolve_globe_spec("campbell_blackglobe_l")
S=5.670374419e-8
allb=[];allg=[]
for c in ['lisbon1','lisbon2','lisbon3','lisbon4','lisbon5','lisbon6']:
    o=json.load(open(f'input/{c}/case.json'))['coordinates']
    d=pd.read_csv(f'run_output/{c}/validation/mrt_lisbon/radiant_flux_comparison_points.csv'); d=d[d.period=='day']
    m=f'run_output/{c}/mrt_facet_out'; th=f'run_output/{c}/thermal_out'
    xyz=np.load(m+'/path_xyz.npy'); pm=np.load(th+'/point_map.npy')
    from scipy.spatial import cKDTree
    _,idx=cKDTree(xyz[:,:2]).query(np.c_[d.x_projected_m-o['local_origin_x'],d.y_projected_m-o['local_origin_y']])
    f=np.load(th+'/facets.npz'); g=f['cls']==0
    W=sp.load_npz(th+'/lw_view_matrix.npz'); Wg=sp.load_npz(th+'/globe_view_matrix.npz')
    alb=np.load(th+'/facet_albedo.npy')
    gw=np.asarray(W[:,g].sum(1)).ravel()[pm]; gwg=np.asarray(Wg[:,g].sum(1)).ravel()
    albg=np.where(gwg>1e-12, np.asarray(Wg[:,g]@alb[g]).ravel()/np.maximum(gwg,1e-12), 0.18)[pm]; gwg=gwg[pm]
    t=pd.read_csv(m+'/times.csv'); tau=np.load(m+'/direct_transmission_matrix.npy'); svfp=np.load(m+'/svf_planar.npy')
    z=np.load(m+'/radiant_flux_contributions.npz')
    it=np.clip(np.round(d.arrival_hour_local.values*6).astype(int),0,143)
    sin=np.sin(np.deg2rad(np.maximum(t.elevation_deg.values[it],0)))
    Kloc=tau[it,idx]*t.DNI_Wm2.values[it]*sin+svfp[idx]*t.DHI_Wm2.values[it]
    # body
    tot=z['sw_reflected_total_absorbed_Wm2'][it,idx]; wr=z['sw_reflected_building_wall_absorbed_Wm2'][it,idx]+z['sw_reflected_roof_absorbed_Wm2'][it,idx]
    ob=(tot-wr)*np.clip(1-2*gw[idx],0,None)
    R=z['total_absorbed_radiant_flux_Wm2'][it,idx].astype(float)
    dT=(R/(0.97*S))**0.25-((R-ob)/(0.97*S))**0.25
    og=spec.sw_absorptivity*0.5*albg[idx]*Kloc*np.clip(1-2*gwg[idx],0,None)
    Rg=z['globe_absorbed_flux_Wm2'][it,idx].astype(float)
    dTg=(Rg/(spec.emissivity*S))**0.25-((Rg-og)/(spec.emissivity*S))**0.25
    print(c,'day samples',len(d),'body overcount mean %.1f W/m2 (Tmrt %.2f K, p90 %.2f) | globe overcount mean %.1f W/m2 (rad-eq T %.2f K)'%(ob.mean(),dT.mean(),np.percentile(dT,90),og.mean(),dTg.mean()))
