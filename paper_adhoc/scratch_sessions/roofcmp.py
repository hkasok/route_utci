import laspy, numpy as np, glob, trimesh, pandas as pd, json, sys
case=sys.argv[1]; half=float(sys.argv[2])
base=f'/media/harshin/data_drive/route based UTCI/input/{case}/'
o=json.load(open(base+'case.json'))['coordinates']; ox,oy=o['local_origin_x'],o['local_origin_y']
r=pd.read_csv(base+'routes/route_1.csv'); mid=r.iloc[len(r)//2]
cx,cy=mid.x_local_m+ox, mid.y_local_m+oy
L='/media/harshin/data_drive/route based UTCI/validation data/Liseben/LAZ/'
B=[]
for f in sorted(glob.glob(L+'*.laz')):
    if '(' in f: continue
    with laspy.open(f) as rd:
        h=rd.header
        if h.maxs[0]<cx-half or h.mins[0]>cx+half or h.maxs[1]<cy-half or h.mins[1]>cy+half: continue
        for ch in rd.chunk_iterator(3_000_000):
            x=np.asarray(ch.x);y=np.asarray(ch.y);c=np.asarray(ch.classification)
            s=(np.abs(x-cx)<half)&(np.abs(y-cy)<half)&(c==6)
            B.append(np.column_stack([x[s]-ox,y[s]-oy,np.asarray(ch.z)[s]]))
B=np.concatenate(B)
res=2.0
ix=((B[:,0]-B[:,0].min())//res).astype(int); iy=((B[:,1]-B[:,1].min())//res).astype(int)
df=pd.DataFrame({'ix':ix,'iy':iy,'z':B[:,2]}).groupby(['ix','iy']).z.max().reset_index()
X=B[:,0].min()+(df.ix+0.5)*res; Y=B[:,1].min()+(df.iy+0.5)*res
b=trimesh.load(base+'geometry/building_final.stl'); g=trimesh.load(base+'geometry/ground_and_water_final.stl')
og=np.column_stack([X,Y,np.full(len(X),2000.0)]); d=np.tile([0,0,-1.0],(len(X),1))
loc,ri,_=b.ray.intersects_location(og,d,multiple_hits=False); zl=np.full(len(X),np.nan); zl[ri]=loc[:,2]
loc,ri,_=g.ray.intersects_location(og,d,multiple_hits=False); zg=np.full(len(X),np.nan); zg[ri]=loc[:,2]
ok=np.isfinite(zl)
e=zl[ok]-df.z.to_numpy()[ok]
hreal=df.z.to_numpy()[ok]-zg[ok]
print(case,'cells',ok.sum(),'LoD1 roof minus LiDAR roof (p90 per 2m cell): median %.1f, MAE %.1f, p10 %.1f p90 %.1f; frac |err|>5m %.2f; real height above ground median %.1f'%(np.median(e),np.abs(e).mean(),np.percentile(e,10),np.percentile(e,90),(np.abs(e)>5).mean(),np.median(hreal)))
