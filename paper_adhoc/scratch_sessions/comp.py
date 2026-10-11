import trimesh,numpy as np
for v in ['full','thin']:
    m=trimesh.load(f'veg_{v}.stl'); 
    bodies=m.split(only_watertight=False)
    vol=sum(abs(b.volume) for b in bodies)
    # vertical rays over grid
    xs=np.arange(-55,55,1.0); X,Y=np.meshgrid(xs,xs)
    o=np.column_stack([X.ravel(),Y.ravel(),np.full(X.size,200.0)]); d=np.tile([0,0,-1.0],(len(o),1))
    loc,ri,_=m.ray.intersects_location(o,d,multiple_hits=True)
    L=np.zeros(len(o))
    for k in np.unique(ri):
        z=np.sort(loc[ri==k,2])[::-1]
        L[k]=sum(z[i]-z[i+1] for i in range(0,len(z)-1,2))
    cov=(L>0)
    # oblique rays sun elev 60 deg
    el=np.radians(60); az=np.radians(200)
    dd=-np.array([np.cos(el)*np.sin(az),np.cos(el)*np.cos(az),np.sin(el)])
    o2=np.column_stack([X.ravel(),Y.ravel(),np.full(X.size,2.0)])-dd*300
    loc,ri,_=m.ray.intersects_location(o2,np.tile(dd,(len(o2),1)),multiple_hits=True)
    L2=np.zeros(len(o2))
    for k in np.unique(ri):
        t=np.sort(np.linalg.norm(loc[ri==k]-o2[k],axis=1))
        L2[k]=sum(t[i+1]-t[i] for i in range(0,len(t)-1,2))
    print(v,'faces',len(m.faces),'bodies',len(bodies),'vol %.0f'%vol,'canopy cover %.3f'%cov.mean(),'mean vertical L in canopy %.2f'%L[cov].mean(), 'sun60 shaded frac %.3f mean L %.2f'%((L2>0).mean(),L2[L2>0].mean()))
    for k in [0.3,0.5,0.7]:
        print('   k=%.1f mean tau over sun-blocked pts %.3f'%(k,np.exp(-k*L2[L2>0]).mean()))
