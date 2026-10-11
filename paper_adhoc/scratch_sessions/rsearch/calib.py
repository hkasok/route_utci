"""Calibrate a screening proxy: core rise ~ c + a*duration + b*beam dose, on the
full-model departure sweep (fixed clothing) of the four campus routes."""
import pickle, numpy as np, pandas as pd
from scipy.spatial import cKDTree
P="/media/harshin/data_drive/route based UTCI"; R=f"{P}/run_output/MMC/mrt_facet_out"
OUT="/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/rsearch"
E=pickle.load(open(f"{OUT}/edge_sun.pkl","rb"))
t=pd.read_csv(f"{R}/times.csv"); hh=(pd.to_datetime(t.time).dt.hour+pd.to_datetime(t.time).dt.minute/60).values
tau_r=np.load(f"{R}/direct_transmission_matrix.npy"); xyz=np.load(f"{R}/path_xyz.npy"); seg=np.load(f"{R}/path_segment_id.npy")
fp=lambda el: 0.308*np.cos(np.deg2rad(el*(0.998-el**2/50000)))
U=1.3
def dose(tau_at, hours_grid, elev, dni, cum, h0):
    """beam load on a standing body integrated along the walk (kJ m-2)."""
    ta=h0+cum/U/3600; out=0.0
    j=np.clip(np.searchsorted(hours_grid,ta)-1,0,len(hours_grid)-2); w=(ta-hours_grid[j])/(hours_grid[j+1]-hours_grid[j])
    tauv=(1-w)*tau_at[j,np.arange(len(cum))]+w*tau_at[j+1,np.arange(len(cum))]
    load=tauv*np.maximum(0,(1-w)*fp(elev[j])*dni[j]*(elev[j]>0)+w*fp(elev[j+1])*dni[j+1]*(elev[j+1]>0))
    dt=np.gradient(cum)/U
    return float((load*dt).sum()/1000)
sw=pd.read_csv(f"{P}/run_output/MMC/postprocessing/departure_sweep/departure_sweep_fixedclothing.csv")
rows=[]
tree=cKDTree(E["xy"])
for r in range(1,5):
    m=seg==(r-1); pxy=xyz[m,:2]; cum=np.concatenate([[0],np.cumsum(np.hypot(*np.diff(pxy,axis=0).T))])
    _,nn=tree.query(pxy)
    for h0 in sorted(sw.departure_hour.unique()):
        d_model=dose(tau_r[:,m],hh,t.elevation_deg.values,t.DNI_Wm2.values,cum,h0)
        d_edge=dose(E["tau"][:,nn],E["hours"],E["elev"],E["dni"],cum,h0)
        q=sw[(sw.route_id==r)&(sw.departure_hour==h0)].iloc[0]
        rows.append(dict(route=r,h0=h0,dur=cum[-1]/U/60,dose_model=d_model,dose_edge=d_edge,rise=q.tcore_rise_c,rad=q.radiation_attributable_rise_c))
D=pd.DataFrame(rows)
print("segment ids:",np.unique(seg), " edge-proxy vs pipeline beam dose: r = %.3f, mean ratio %.3f"%(np.corrcoef(D.dose_model,D.dose_edge)[0,1],(D.dose_edge/D.dose_model).mean()))
X=np.column_stack([np.ones(len(D)),D.dur,D.dose_model,D.h0,D.h0**2]); b,*_=np.linalg.lstsq(X,D.rise,rcond=None)
pred=X@b; print("fit rise ~ c + a*dur + b*dose + h0 terms: coef",b.round(5)," R2 %.3f  RMSE %.4f C"%(1-((D.rise-pred)**2).sum()/((D.rise-D.rise.mean())**2).sum(), np.sqrt(((D.rise-pred)**2).mean())))
Xr=np.column_stack([np.ones(len(D)),D.dose_model]); br,*_=np.linalg.lstsq(Xr,D.rad,rcond=None); pr=Xr@br
print("radiant part ~ dose: coef",br.round(5)," R2 %.3f"%(1-((D.rad-pr)**2).sum()/((D.rad-D.rad.mean())**2).sum()))
D.to_csv(f"{OUT}/calib.csv",index=False); pickle.dump(dict(b=b,br=br),open(f"{OUT}/calib.pkl","wb"))
