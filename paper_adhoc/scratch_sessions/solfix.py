import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np, black_globe as bg
S=5.670374419e-8
allP=[]
for i in range(1,7):
    P=pd.read_csv(f'run_output/lisbon{i}/solweig/solweig_comparison_points.csv')
    t=pd.read_csv(f'run_output/lisbon{i}/mrt_facet_out/times.csv'); tt=pd.to_datetime(t.time); th=(tt.dt.hour+tt.dt.minute/60).to_numpy(float)
    h=P.arrival_hour_local.to_numpy(float)
    az=np.radians(np.interp(h,th,t.azimuth_deg)); el=np.interp(h,th,t.elevation_deg); dni=np.interp(h,th,t.DNI_Wm2)
    hh=np.radians(np.clip(el,0,90))
    box=(np.sin(hh)+np.cos(hh)*(np.abs(np.sin(az))+np.abs(np.cos(az))))/6
    corr=np.where(el>0,0.95*dni*P.solweig_shadow*(0.25-box),0)
    load=P.solweig_globe_load_uncorrected_Wm2+corr
    # check old
    cube=0.20*np.cos(hh)+np.sin(hh)/6
    old=P.solweig_globe_load_uncorrected_Wm2+np.where(el>0,0.95*dni*P.solweig_shadow*(0.25-cube),0)
    assert np.allclose(old,P.solweig_globe_load_Wm2,atol=1e-6)
    P['fix']=np.nan
    for _,g in P.groupby(['route_id','period']):
        g=g.sort_values('seq'); el_s=(g.arrival_hour_local-g.arrival_hour_local.iloc[0]).to_numpy()*3600
        T,_=bg.integrate_globe_temperature_C(el_s, load[g.index].to_numpy(), g.trec_route_air_temperature_c.to_numpy(), g.globe_ventilation_ms.to_numpy(), bg.DEFAULT_GLOBE)
        P.loc[g.index,'fix']=T
    allP.append(P)
P=pd.concat(allP)
for per in ('day','night'):
    G=P[(P.period==per)&~P.globe_spinup_affected.astype(bool)]
    for c in ('solweig_globe_C','fix','globe_transient_temperature_C'):
        d=G[c]-G.measured_black_globe_temperature_c
        print(per,c,len(d),'MBE %.2f RMSE %.2f cRMSE %.2f r %.3f'%(d.mean(),np.sqrt((d**2).mean()),d.std(ddof=0),np.corrcoef(G[c],G.measured_black_globe_temperature_c)[0,1]))
    if per=='day':
        for cs,g in G.groupby('case_id'):
            print('  ',cs,'old %.2f/%.2f'%((g.solweig_globe_C-g.measured_black_globe_temperature_c).mean(),np.sqrt(((g.solweig_globe_C-g.measured_black_globe_temperature_c)**2).mean())),
                  'fix %.2f/%.2f'%((g.fix-g.measured_black_globe_temperature_c).mean(),np.sqrt(((g.fix-g.measured_black_globe_temperature_c)**2).mean())),
                  'trec %.2f/%.2f'%((g.globe_transient_temperature_C-g.measured_black_globe_temperature_c).mean(),np.sqrt(((g.globe_transient_temperature_C-g.measured_black_globe_temperature_c)**2).mean())))
G=P[(P.period=='day')&~P.globe_spinup_affected.astype(bool)]
ms=G.measured_swin_wm2>0.5*G.trec_atmospheric_ghi_wm2
mo=G.sensor_shortwave_down_Wm2>0.5*G.trec_atmospheric_ghi_wm2
for lab,m in (('meas sun',ms),('meas shade',~ms),('both sun',ms&mo),('both shade',~ms&~mo)):
    g=G[m]; print(lab,len(g),'SOLWEIG-TREC old %.2f fix %.2f'%((g.solweig_globe_C-g.globe_transient_temperature_C).mean(),(g.fix-g.globe_transient_temperature_C).mean()))
