# Median crown solidity (|volume| / convex-hull volume) per scene, same rule as
# paper_adhoc/crown_solidity.py (components with >=20 faces), without trimesh.split.
import sys, numpy as np
from scipy.spatial import ConvexHull
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
P="/media/harshin/data_drive/route based UTCI"
def readstl(p):
    with open(p,'rb') as f:
        f.seek(80); n=np.frombuffer(f.read(4),'<u4')[0]
        dt=np.dtype([('n','<f4',3),('v','<f4',(3,3)),('a','<u2')])
        return np.frombuffer(f.read(n*50),dt)['v'].astype(np.float64)
for c in sys.argv[1:]:
    V=readstl(f"{P}/input/{c}/geometry/vegetation_final.stl")
    flat=V.reshape(-1,3)
    u,inv=np.unique(flat,axis=0,return_inverse=True); inv=inv.ravel()
    F=inv.reshape(-1,3); nf=len(F)
    # face-vertex graph components
    rows=np.repeat(np.arange(nf),3); g=coo_matrix((np.ones(3*nf),(rows,F.ravel()+0)),shape=(nf,len(u)))
    A=coo_matrix((np.ones(3*nf),(rows,nf+F.ravel())),shape=(nf+len(u),nf+len(u)))
    nc,lab=connected_components(A,directed=False)
    fl=lab[:nf]; order=np.argsort(fl,kind='stable'); fl_s=fl[order]
    starts=np.r_[0,np.flatnonzero(np.diff(fl_s))+1,nf]
    sol=[]
    for i in range(len(starts)-1):
        idx=order[starts[i]:starts[i+1]]
        if len(idx)<20: continue
        tri=V[idx]
        vol=abs(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6.0)
        try: hv=ConvexHull(u[np.unique(F[idx])]).volume
        except Exception: continue
        if hv>1e-6: sol.append(vol/hv)
    sol=np.array(sol)
    print(f"{c}: {len(sol)} crowns, median solidity {np.median(sol):.4f}",flush=True)
