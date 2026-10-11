import numpy as np, trimesh, sys
p=sys.argv[1]; half=float(sys.argv[2]) if len(sys.argv)>2 else 80
m=trimesh.load(p,process=False)
lo=m.bounds[0]; hi=m.bounds[1]
cx=(lo[0]+hi[0])/2; cy=(lo[1]+hi[1])/2
tri=m.triangles
sel=(np.abs(tri[:,:,0].mean(1)-cx)<half)&(np.abs(tri[:,:,1].mean(1)-cy)<half)
sub=m.submesh([np.nonzero(sel)[0]],append=True); sub.merge_vertices()
bodies=[b for b in sub.split(only_watertight=True) if len(b.faces)>=100]
res=[]
for b in bodies:
    bb=b.bounds; h=bb[1,2]-bb[0,2]
    w=max(bb[1,0]-bb[0,0],bb[1,1]-bb[0,1])
    xs=np.linspace(bb[0,0],bb[1,0],7)[2:-2]; ys=np.linspace(bb[0,1],bb[1,1],7)[2:-2]
    X,Y=np.meshgrid(xs,ys); o=np.column_stack([X.ravel(),Y.ravel(),np.full(X.size,bb[1,2]+1)])
    d=np.tile([0,0,-1.0],(len(o),1))
    loc,ri,_=b.ray.intersects_location(o,d,multiple_hits=True)
    Ls=[]
    for k in range(len(o)):
        z=np.sort(loc[ri==k,2])[::-1]
        if len(z)>=2: Ls.append(sum(z[i]-z[i+1] for i in range(0,len(z)-1,2)))
    if Ls: res.append((h,w,np.max(Ls),b.volume/(h*w*w+1e-9)))
r=np.array(res)
print(p,'watertight crowns',len(r))
print(' median height %.2f width %.2f max central vertical chord %.2f'%tuple(np.median(r[:,:3],0)), ' median chord/height %.2f'%np.median(r[:,2]/r[:,0]))
big=r[r[:,0]>5]
print(' crowns >5 m tall: n=%d median height %.2f median chord %.2f chord/height %.2f'%(len(big),np.median(big[:,0]),np.median(big[:,2]),np.median(big[:,2]/big[:,0])))
