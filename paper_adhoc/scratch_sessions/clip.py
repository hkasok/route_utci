import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, black_globe as bg
S=5.670374419e-8
out=[]
for i in range(1,7):
    P=pd.read_csv(f'run_output/lisbon{i}/solweig/solweig_comparison_points.csv')
    t=pd.read_csv(f'run_output/lisbon{i}/mrt_facet_out/times.csv'); tt=pd.to_datetime(t.time); th=(tt.dt.hour+tt.dt.minute/60).to_numpy(float)
    h=P.arrival_hour_local.to_numpy(float)
    az=np.radians(np.interp(h,th,t.azimuth_deg)); el=np.interp(h,th,t.elevation_deg); I=np.interp(h,th,t.DNI_Wm2)
    hh=np.radians(np.clip(el,0,90)); sh=P.solweig_shadow.to_numpy()
    beamI=np.where(el>0,I*sh*np.cos(hh),0)
    Kd,Ku,Ld,Lu=P.solweig_kdown,P.solweig_kup,P.solweig_ldown,P.solweig_lup
    Sst=0.97*S*(P.solweig_tmrt+273.15)**4
    Lside=2*(Ld+Lu)   # assumption: each wall sees the mean of up and down
    X=(Sst-0.97*(0.06*(Ld+Lu)+0.22*Lside))/0.70
    X=(X-0.06*(Kd+Ku)-0.28*beamI)/0.22   # sum of diffuse+reflected side shortwave
    est=0.95/6*(Kd+Ku+X+beamI*(np.abs(np.sin(az))+np.abs(np.cos(az))))+0.957/6*(Ld+Lu+Lside)
    P['est']=est; P['case']=i; out.append(P)
P=pd.concat(out)
act=P.solweig_globe_load_uncorrected_Wm2
Tg=(act/(0.957*S))**0.25-273.15
clipped=Tg>79.99
ok=(P.period=='day')&~clipped
print('estimator check on unclipped day: mean diff %.1f sd %.1f W/m2'%((P.est-act)[ok].mean(),(P.est-act)[ok].std()))
print('clipped samples',clipped.sum(),'of day',(P.period=='day').sum(),'; post-spinup day clipped',(clipped&(P.period=='day')&~P.globe_spinup_affected.astype(bool)).sum())
lost=(P.est-act)[clipped]
print('estimated load lost on clipped samples: mean %.1f W/m2, median %.1f'%(lost.mean(),lost.median()))
print(P[clipped].groupby('case').size().to_dict())
P.to_pickle('/tmp/claude-1000/-home-harshin-files-fastUTEC-paper/655b6985-02aa-42c7-89b6-13a1d0cf8da5/scratchpad/clipP.pkl')
