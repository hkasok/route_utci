import pandas as pd, numpy as np
P=pd.concat([pd.read_csv(f'run_output/lisbon{i}/solweig/solweig_comparison_points.csv') for i in range(1,7)],ignore_index=True)
D=P[(P.period=='day')&(P.trec_atmospheric_ghi_wm2>50)].dropna(subset=['measured_swin_wm2','sensor_shortwave_down_Wm2','solweig_kdown'])
g=D.trec_atmospheric_ghi_wm2.values
ms=D.measured_swin_wm2.values>0.5*g; ts=D.sensor_shortwave_down_Wm2.values>0.5*g; ss=D.solweig_kdown.values>0.5*g
print('agreement TREC %.1f%%  SOLWEIG %.1f%%'%(100*(ms==ts).mean(),100*(ms==ss).mean()))
for nm,mod,st in (('TREC',D.sensor_shortwave_down_Wm2.values,ts),('SOLWEIG',D.solweig_kdown.values,ss)):
    b=ms&st; print(nm,'sunlit(both) K-down bias %.0f'%(mod[b]-D.measured_swin_wm2.values[b]).mean())
# albedo, both sunlit with measured
for nm,ku,kd,st in (('TREC','sensor_shortwave_up_Wm2','sensor_shortwave_down_Wm2',ts),('SOLWEIG','solweig_kup','solweig_kdown',ss)):
    b=ms&st; a=D[ku].values[b]/D[kd].values[b]; am=D.measured_swout_wm2.values[b]/D.measured_swin_wm2.values[b]
    print(nm,'albedo mean %.3f meas %.3f r %.2f'%(a.mean(),am.mean(),np.corrcoef(a,am)[0,1]))
G=P[(P.period=='day')&~P.globe_spinup_affected.astype(bool)]
gg=G.trec_atmospheric_ghi_wm2.values; gm=G.measured_swin_wm2.values>0.5*gg
d=G.solweig_globe_C-G.globe_transient_temperature_C
print('SOLWEIG-TREC globe: in measured sun %.2f, in measured shade %.2f, all %.2f'%(d[gm].mean(),d[~gm].mean(),d.mean()))
for k in ('solweig_globe_C','globe_transient_temperature_C'):
    e=G[k]-G.measured_black_globe_temperature_c; print(k,'cRMSE %.2f r %.2f'%(np.sqrt(((e-e.mean())**2).mean()), np.corrcoef(G[k],G.measured_black_globe_temperature_c)[0,1]))
