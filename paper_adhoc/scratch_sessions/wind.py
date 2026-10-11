import pandas as pd, numpy as np, json, sys
sys.path.insert(0,'.')
import black_globe as bg
for i in range(1,7):
    m=pd.read_csv(f'input/lisbon{i}/measurements/experimental_measurements_all_routes.csv')
    d=pd.read_csv(f'run_output/lisbon{i}/validation/mrt_lisbon/radiant_flux_comparison_points.csv')
    for rid,g in m.groupby('route_id'):
        g=g.sort_values('seq')
        t=pd.to_datetime(g.timestamp_utc_refined,utc=True,format="ISO8601")
        ts=(t-t.iloc[0]).dt.total_seconds().to_numpy()
        u,v=bg.receptor_velocity(g.x_local_m.to_numpy(),g.y_local_m.to_numpy(),ts)
        sp=np.hypot(u,v); hd=np.degrees(np.arctan2(u,v))%360
        wd=g.WDir.to_numpy(float); ws=g.WS.to_numpy(float)
        rel=np.abs((wd-hd+180)%360-180)
        moving=sp>0.8
        stopped=sp<0.2
        dd=d[d.route_id==rid]
        print(i,rid,'n',len(g),'WS mean',ws.mean().round(2),'min',ws.min(),'p05',np.percentile(ws,5),
              'speed mean',sp.mean().round(2),
              'frac WDir within45 of heading(moving)',np.mean(rel[moving]<45).round(2),
              'WS stopped',ws[stopped].mean().round(2) if stopped.any() else None, stopped.sum(),
              'corr(WS,speed)',np.corrcoef(ws,sp)[0,1].round(2),
              'model local wind mean', dd.trec_route_wind_ms.mean().round(2), 'vent mean', dd.globe_ventilation_ms.mean().round(2))
