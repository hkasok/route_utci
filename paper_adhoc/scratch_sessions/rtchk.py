import trimesh, numpy as np, pandas as pd, sys
case=sys.argv[1]; base=f'/media/harshin/data_drive/route based UTCI/input/{case}/'
b=trimesh.load(base+'geometry/building_final.stl'); g=trimesh.load(base+'geometry/ground_and_water_final.stl')
for rid in sys.argv[2:]:
    r=pd.read_csv(base+f'routes/route_{rid}.csv')
    xy=r[['x_local_m','y_local_m']].to_numpy()[::5]
    o=np.column_stack([xy,np.full(len(xy),500.0)]); d=np.tile([0,0,-1.0],(len(o),1))
    ib=b.ray.intersects_any(o,d)
    loc,ri,_=g.ray.intersects_location(o,d,multiple_hits=False)
    zg=np.full(len(o),np.nan); zg[ri]=loc[:,2]
    print(case,rid,'n',len(xy),'under building %.2f%%'%(100*ib.mean()),'ground hit %.3f'%np.isfinite(zg).mean(),'ground z range %.1f..%.1f'%(np.nanmin(zg),np.nanmax(zg)), 'xy range',xy.min(0).round(1),xy.max(0).round(1),'mesh xy',g.bounds[:,:2].round(1).tolist())
