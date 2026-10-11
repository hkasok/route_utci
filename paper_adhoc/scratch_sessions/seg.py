import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np, pandas as pd
import paper_globe_attribution as pa, paper_validation_figures as pv, black_globe as bg
pts=pv.load_points(Path('.')); pts,_=pv.apply_measurement_qc(pts)
walks=pa.walk_frames(pts); all_day=pa.add_sample_quantities(walks, Path('.'))
walks=[g.sort_values('seq').reset_index(drop=True) for _,g in all_day.groupby(['case_id','route_id'])]
C=bg.CAMPBELL_BLACKGLOBE_L.areal_heat_capacity_J_m2K
for g in walks:
    tg=g.measured_black_globe_temperature_c.to_numpy(float); t=g.elapsed_s.to_numpy(float)
    # smoothed derivative over +-30 s
    g['storage']=C*np.gradient(pd.Series(tg).rolling(7,center=True,min_periods=1).mean().to_numpy(), t)
orig=pa.implied_excess
def ie(d): return orig(d)-d['storage'].to_numpy(float)
pa.implied_excess=ie
for lab,f in (('meas',lambda g:g['measured_sunlit']),('both',lambda g:g['measured_sunlit'].where(g['measured_sunlit']==g['model_sunlit']))):
    s=pa.sustained_segments(walks,f,lab)
    print(lab,{k:(s[k]['n'],round(s[k]['residual_mean_K'],2),round(s[k]['excess_Wm2'],1)) for k in ('shade','sun')})
pa.implied_excess=orig
for lab,f in (('meas',lambda g:g['measured_sunlit']),('both',lambda g:g['measured_sunlit'].where(g['measured_sunlit']==g['model_sunlit']))):
    s=pa.sustained_segments(walks,f,lab)
    print('orig',lab,{k:(s[k]['n'],round(s[k]['residual_mean_K'],2),round(s[k]['excess_Wm2'],1)) for k in ('shade','sun')})
d=pa.usable_day(all_day)
print('whole-day storage mean W/m2', np.mean(pd.concat(walks).loc[lambda x:~x.globe_spinup_affected.astype(bool),'storage']))
print('--- shade residual by time since entering measured shade (state run start), both-agree')
rows=[]
for g in walks:
    st=g['measured_sunlit'].where(g['measured_sunlit']==g['model_sunlit']).to_numpy()
    t=g.elapsed_s.to_numpy(float)
    start=0
    for i in range(1,len(g)+1):
        if i==len(g) or st[i]!=st[start] or (isinstance(st[i],float) and np.isnan(st[i])):
            if st[start] is False or st[start]==0:
                for k in range(start,i):
                    rows.append((t[k]-t[start], g.residual_K.iloc[k], bool(g.globe_spinup_affected.iloc[k]), t[i-1]-t[start]))
            start=i
r=pd.DataFrame(rows,columns=['since','res','spin','dur'])
r=r[~r.spin]
for lo,hi in [(0,60),(60,120),(120,300),(300,600),(600,1e9)]:
    m=(r.since>=lo)&(r.since<hi); print(lo,hi,m.sum(),round(r.res[m].mean(),2))
print('segments with dur>=300:', (r.dur>=300).sum(), r.res[r.dur>=300].mean())
print('--- run durations (measured state)')
for lab,f in (('meas',lambda g:g['measured_sunlit'].astype(float)),('both',lambda g:g['measured_sunlit'].where(g['measured_sunlit']==g['model_sunlit']).astype(float))):
    durs={0.0:[],1.0:[]}
    for g in walks:
        st=f(g).to_numpy(); t=g.elapsed_s.to_numpy(float); dtm=np.median(np.diff(t))
        start=0
        for i in range(1,len(g)+1):
            if i==len(g) or st[i]!=st[start] or np.isnan(st[i]):
                if not np.isnan(st[start]):
                    du=t[i-1]-t[start]+dtm
                    if du>=60: durs[st[start]].append(du)
                start=i
    print(lab,'shade durs',sorted(np.round(durs[0.0]))[-8:],'n',len(durs[0.0]),'sun max',max(durs[1.0]), 'sun median', np.median(durs[1.0]))
