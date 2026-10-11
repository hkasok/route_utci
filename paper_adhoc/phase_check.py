import pandas as pd, numpy as np, sys
def walks(root):
    for c in range(1,7):
        P=pd.read_csv(f'{root(c)}/validation/mrt_lisbon/radiant_flux_comparison_points.csv')
        for (rid,per),g in P.groupby(['route_id','period']):
            g=g.sort_values('seq'); g=g[~g.globe_spinup_affected.astype(bool)]
            yield c,per,g
def analyse(root):
    out=[]
    for c,per,g in walks(root):
        t=(g.arrival_hour_local.to_numpy()-g.arrival_hour_local.iloc[0])*3600
        m=g.globe_transient_temperature_C.to_numpy(); o=g.measured_black_globe_temperature_c.to_numpy()
        ok=np.isfinite(m)&np.isfinite(o); t,m,o=t[ok],m[ok],o[ok]
        tu=np.arange(t[0],t[-1],5.0); mu=np.interp(tu,t,m); ou=np.interp(tu,t,o)
        best=None
        for lag in range(-60,61):   # lag in 5 s steps; positive: model ahead
            if lag>=0: a,b=mu[:len(mu)-lag],ou[lag:]
            else: a,b=mu[-lag:],ou[:len(ou)+lag]
            r=np.corrcoef(a,b)[0,1]
            if best is None or r>best[1]: best=(lag*5,r)
        area=0.5*np.sum(o[:-1]*m[1:]-o[1:]*m[:-1])   # signed area, x=measured, y=model
        out.append((c,per,best[0],round(area,1)))
    return out
for tag,root in (('old',lambda c: f'backups_cloudfix_20261011/run_output/lisbon{c}' if c in (4,6) else f'run_output/lisbon{c}'),('new',lambda c:f'run_output/lisbon{c}')):
    o=analyse(root); print(tag,o)
    day=[x for x in o if x[1]=='day']
    print(tag,'model ahead by xcorr:',sum(x[2]>0 for x in o),'/12; day leads',sorted(x[2] for x in day))
