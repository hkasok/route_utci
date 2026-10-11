import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, black_globe as bg
S=5.670374419e-8
P=pd.read_pickle('/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/clipP.pkl').reset_index(drop=True)
act=P.solweig_globe_load_uncorrected_Wm2
clipped=((act/(0.957*S))**0.25-273.15)>79.99
unclip=np.where(clipped, np.maximum(P.est-11.9, act), act)
res={}
for i in range(1,7):
    t=pd.read_csv(f'run_output/lisbon{i}/mrt_facet_out/times.csv'); tt=pd.to_datetime(t.time); th=(tt.dt.hour+tt.dt.minute/60).to_numpy(float)
    m=(P.case==i).to_numpy()
    h=P.arrival_hour_local[m].to_numpy(float)
    az=np.radians(np.interp(h,th,t.azimuth_deg)); el=np.interp(h,th,t.elevation_deg); I=np.interp(h,th,t.DNI_Wm2)
    hh=np.radians(np.clip(el,0,90))
    box=(np.sin(hh)+np.cos(hh)*(np.abs(np.sin(az))+np.abs(np.cos(az))))/6
    corr=np.where(el>0,0.95*I*P.solweig_shadow[m]*(0.25-box),0)
    P.loc[m,'load_beamfix']=act[m]+corr
    P.loc[m,'load_both']=unclip[m]+corr
for col in ('load_beamfix','load_both'):
    P['T_'+col]=np.nan
    for _,g in P.groupby(['case','route_id','period']):
        g=g.sort_values('seq'); el_s=(g.arrival_hour_local-g.arrival_hour_local.iloc[0]).to_numpy()*3600
        T,_=bg.integrate_globe_temperature_C(el_s,g[col].to_numpy(),g.trec_route_air_temperature_c.to_numpy(),g.globe_ventilation_ms.to_numpy(),bg.DEFAULT_GLOBE)
        P.loc[g.index,'T_'+col]=T
G=P[(P.period=='day')&~P.globe_spinup_affected.astype(bool)]
for c in ('solweig_globe_C','T_load_beamfix','T_load_both','globe_transient_temperature_C'):
    d=G[c]-G.measured_black_globe_temperature_c
    print(c,'MBE %.2f RMSE %.2f cRMSE %.2f'%(d.mean(),np.sqrt((d**2).mean()),d.std(ddof=0)))
for cs,g in G.groupby('case'):
    f=lambda c: np.sqrt(((g[c]-g.measured_black_globe_temperature_c)**2).mean())
    print(cs,'RMSE sol old %.2f both-fixed %.2f trec %.2f'%(f('solweig_globe_C'),f('T_load_both'),f('globe_transient_temperature_C')))
