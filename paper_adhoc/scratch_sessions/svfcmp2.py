import laspy, numpy as np, glob, trimesh, pandas as pd, json, sys
from scipy import ndimage
case=sys.argv[1]; half=float(sys.argv[2])
base=f'/media/harshin/data_drive/route based UTCI/input/{case}/'
o=json.load(open(base+'case.json'))['coordinates']; ox,oy=o['local_origin_x'],o['local_origin_y']
r=pd.read_csv(base+'routes/route_1.csv'); mid=r.iloc[len(r)//2]
cxl,cyl=mid.x_local_m, mid.y_local_m; cx,cy=cxl+ox,cyl+oy
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
res=1.0; x0,y0=cxl-half,cyl-half; n=int(2*half/res)
gx=x0+(np.arange(n)+0.5)*res; GX,GY=np.meshgrid(gx,y0+(np.arange(n)+0.5)*res)
b=trimesh.load(base+'geometry/building_final.stl'); g=trimesh.load(base+'geometry/ground_and_water_final.stl')
og=np.column_stack([GX.ravel(),GY.ravel(),np.full(GX.size,3000.0)]); d=np.tile([0,0,-1.0],(GX.size,1))
loc,ri,_=g.ray.intersects_location(og,d,multiple_hits=False); zg=np.full(GX.size,np.nan); zg[ri]=loc[:,2]
loc,ri,_=b.ray.intersects_location(og,d,multiple_hits=False); zl=np.full(GX.size,np.nan); zl[ri]=loc[:,2]
zg=zg.reshape(n,n); zl=zl.reshape(n,n)
dsm_lod=np.where(np.isfinite(zl),zl,zg)
ix=((B[:,0]-x0)//res).astype(int); iy=((B[:,1]-y0)//res).astype(int); ok=(ix>=0)&(ix<n)&(iy>=0)&(iy<n)
true=np.full((n,n),-np.inf); np.maximum.at(true,(iy[ok],ix[ok]),B[ok,2])
true=ndimage.grey_closing(np.where(np.isfinite(true),true,-1e9),size=3)  # fill pinholes
dsm_true=np.maximum(np.where(true>-1e8,true,zg),zg)
# route points in the inner region, outside both building footprints
PX=np.load(f'/media/harshin/data_drive/route based UTCI/run_output/{case}/mrt_facet_out/path_xyz.npy')[:,:2]
P=PX[(np.abs(PX[:,0]-cxl)<half-150)&(np.abs(PX[:,1]-cyl)<half-150)][::25]
def svf(dsm,p):
    i=int((p[1]-y0)//res); j=int((p[0]-x0)//res); z0=zg[i,j]+1.1
    out=[]
    for az in np.linspace(0,2*np.pi,72,endpoint=False):
        dd=np.arange(1.0,150.0,0.5); xs=p[0]+dd*np.sin(az); ys=p[1]+dd*np.cos(az)
        jj=((xs-x0)//res).astype(int); ii=((ys-y0)//res).astype(int)
        m=(ii>=0)&(ii<n)&(jj>=0)&(jj<n)
        el=np.max(np.arctan2(dsm[ii[m],jj[m]]-z0,dd[m])); el=max(el,0)
        out.append(np.cos(el)**2)
    return np.mean(out)
inl=[np.isfinite(zl.reshape(n,n)[int((p[1]-y0)//res),int((p[0]-x0)//res)]) for p in P]
s1=[];s2=[]
for p,ib in zip(P,inl):
    if ib: continue
    s1.append(svf(dsm_lod,p)); s2.append(svf(dsm_true,p))
s1=np.array(s1);s2=np.array(s2);k=np.isfinite(s1)&np.isfinite(s2);s1=s1[k];s2=s2[k]
print(case,'route pts',len(s1),'planar SVF LoD1 mean %.3f  LiDAR-DSM mean %.3f  diff %.3f (median %.3f)'%(s1.mean(),s2.mean(),(s1-s2).mean(),np.median(s1-s2)))
