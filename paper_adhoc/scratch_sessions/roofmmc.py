import numpy as np, pandas as pd, trimesh
base='/media/harshin/data_drive/route based UTCI/'
B=np.load(base+'out_mmc_metric/00_split/building_points.npy',mmap_mode='r')[::3]
B=np.asarray(B)
res=2.0
ix=((B[:,0]-B[:,0].min())//res).astype(int); iy=((B[:,1]-B[:,1].min())//res).astype(int)
df=pd.DataFrame({'ix':ix,'iy':iy,'z':B[:,2]}).groupby(['ix','iy']).z.max().reset_index()
X=B[:,0].min()+(df.ix+0.5)*res; Y=B[:,1].min()+(df.iy+0.5)*res
b=trimesh.load(base+'input/MMC/geometry/building_final.stl'); g=trimesh.load(base+'input/MMC/geometry/ground_and_water_final.stl')
og=np.column_stack([X,Y,np.full(len(X),2000.0)]); d=np.tile([0,0,-1.0],(len(X),1))
loc,ri,_=b.ray.intersects_location(og,d,multiple_hits=False); zl=np.full(len(X),np.nan); zl[ri]=loc[:,2]
loc,ri,_=g.ray.intersects_location(og,d,multiple_hits=False); zg=np.full(len(X),np.nan); zg[ri]=loc[:,2]
ok=np.isfinite(zl); e=zl[ok]-df.z.to_numpy()[ok]; h=df.z.to_numpy()[ok]-zg[ok]
print('MMC cells',ok.sum(),'of',len(X),'median err %.1f MAE %.1f p10 %.1f p90 %.1f frac>5m %.2f real h median %.1f'%(np.median(e),np.abs(e).mean(),np.percentile(e,10),np.percentile(e,90),(np.abs(e)>5).mean(),np.median(h)))
