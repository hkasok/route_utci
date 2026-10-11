import numpy as np, scipy.sparse as sp
S=5.670374419e-8
for case in ["lisbon1","lisbon2","lisbon3","lisbon4","lisbon5","lisbon6","MMC"]:
    d=f"{case}/thermal_out"; m=f"{case}/mrt_facet_out"
    W=sp.load_npz(d+"/lw_view_matrix.npz"); f=np.load(d+"/facets.npz"); pm=np.load(d+"/point_map.npy")
    gw=np.asarray(W[:,f["cls"]==0].sum(1)).ravel()[pm]
    z=np.load(m+"/radiant_flux_contributions.npz")
    it=78  # 13:00
    tot=z["sw_reflected_total_absorbed_Wm2"][it].astype(float)
    wr=z["sw_reflected_building_wall_absorbed_Wm2"][it]+z["sw_reflected_roof_absorbed_Wm2"][it]
    g=tot-wr
    over=g*np.clip(1-2*gw,0,None)
    R=z["total_absorbed_radiant_flux_Wm2"][it].astype(float)
    T0=(R/(0.97*S))**0.25; T1=((R-over)/(0.97*S))**0.25
    print(case,"13:00 ground-refl abs mean %.1f, wall-refl %.1f, overcount mean %.1f p90 %.1f W/m2 -> dTmrt mean %.2f p90 %.2f max %.2f K; zero-gw pts %.1f%%"%(g.mean(),wr.mean(),over.mean(),np.percentile(over,90),(T0-T1).mean(),np.percentile(T0-T1,90),(T0-T1).max(),100*(gw<0.05).mean()))
