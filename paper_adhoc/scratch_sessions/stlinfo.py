import numpy as np, sys
def read(p):
    with open(p,'rb') as f:
        h=f.read(80); 
        if h[:5]==b'solid':
            # might be ascii; check size
            pass
        n=np.frombuffer(f.read(4),'<u4')[0]
        dt=np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')])
        d=np.fromfile(f,dt,count=n)
    return d
for p in sys.argv[1:]:
    d=read(p); v=d['v'].reshape(-1,3)
    print(p, len(d), 'min',v.min(0).round(2),'max',v.max(0).round(2))
    # normal check: compare stored normal with computed
    e1=d['v'][:,1]-d['v'][:,0]; e2=d['v'][:,2]-d['v'][:,0]; c=np.cross(e1,e2)
    nn=np.linalg.norm(c,axis=1); ok=nn>0
    cn=c[ok]/nn[ok,None]
    print('  frac up-facing (nz>0.5):',(cn[:,2]>0.5).mean().round(3),'down (nz<-0.5):',(cn[:,2]<-0.5).mean().round(3), 'degenerate',(~ok).sum())
    sn=d['n'][ok]; dot=(sn*cn).sum(1)
    print('  stored-normal agreement: mean dot',dot.mean().round(3), 'frac zero stored', (np.abs(sn).sum(1)==0).mean().round(3))
