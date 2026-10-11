import numpy as np, trimesh, sys
p=sys.argv[1]
m=trimesh.load(p,process=False)
print(p,len(m.faces))
# group faces into crowns by 800-face-ish connectivity: use split on a subset region
rng=np.random.default_rng(0)
# pick random region 60x60 m
c=m.vertices.mean(0)
lo=m.bounds[0]; hi=m.bounds[1]
res=[]
for trial in range(6):
    cx=rng.uniform(lo[0]+50,hi[0]-50); cy=rng.uniform(lo[1]+50,hi[1]-50)
    tri=m.triangles
    sel=(np.abs(tri[:,:,0].mean(1)-cx)<30)&(np.abs(tri[:,:,1].mean(1)-cy)<30)
    if sel.sum()<100: continue
    sub=m.submesh([np.nonzero(sel)[0]],append=True); sub.merge_vertices()
    bodies=sub.split(only_watertight=False)
    for b in bodies:
        if len(b.faces)<200: continue
        bb=b.bounds; h=bb[1,2]-bb[0,2]
        try: vol=b.volume
        except: continue
        # vertical rays on grid inside footprint
        xs=np.linspace(bb[0,0],bb[1,0],9)[1:-1]; ys=np.linspace(bb[0,1],bb[1,1],9)[1:-1]
        X,Y=np.meshgrid(xs,ys); o=np.column_stack([X.ravel(),Y.ravel(),np.full(X.size,bb[1,2]+1)])
        d=np.tile([0,0,-1.0],(len(o),1))
        loc,ri,_=b.ray.intersects_location(o,d,multiple_hits=True)
        chords=[]
        for k in range(len(o)):
            z=np.sort(loc[ri==k,2])[::-1]
            if len(z)>=2:
                L=sum(z[i]-z[i+1] for i in range(0,len(z)-1,2))
                chords.append((L, z[0]-z[-1]))
        if chords:
            ch=np.array(chords)
            res.append((h, b.is_watertight, vol, np.median(ch[:,0]), np.median(ch[:,1]), np.median(ch[:,0]/np.maximum(ch[:,1],1e-6))))
    if len(res)>40: break
r=np.array(res,dtype=float)
print('n crowns',len(r),'watertight frac',r[:,1].mean())
print('median bbox height',np.median(r[:,0]),'median vertical in-crown chord',np.median(r[:,3]),'median first-to-last extent',np.median(r[:,4]),'median chord/extent',np.median(r[:,5]))
for row in r[:12]: print(np.round(row,2))
