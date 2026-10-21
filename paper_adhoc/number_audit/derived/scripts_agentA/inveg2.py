import numpy as np, trimesh, json, sys
P="/media/harshin/data_drive/route based UTCI"
sys.path.insert(0,P)
from thermal_common import get_intersector
for case in sys.argv[1:]:
    cj=json.load(open(f"{P}/input/{case}/case.json"))
    v=trimesh.load(f"{P}/input/{case}/"+cj["files"]["vegetation_stl"],force="mesh")
    xyz=np.load(f"{P}/run_output/{case}/mrt_facet_out/path_xyz.npy")
    it=get_intersector(v,quiet=True)
    cnts=[]
    for d in ([0,0,1],[0.6,0.0,0.8],[-0.3,0.5,0.81]):
        d=np.array(d,float); d/=np.linalg.norm(d)
        loc,ir,_=it.intersects_location(xyz,np.tile(d,(len(xyz),1)),multiple_hits=True)
        dist=np.einsum('ij,ij->i',loc-xyz[ir],np.tile(d,(len(ir),1)))
        ir=ir[dist>1e-6]
        cnts.append(np.bincount(ir,minlength=len(xyz))%2)
    iv=(np.sum(cnts,axis=0)>=2)
    print(case,"pts",len(xyz),"inside veg crown %.3f%%"%(100*iv.mean()),flush=True)
