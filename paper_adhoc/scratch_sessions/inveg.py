import numpy as np, trimesh, json, sys
sys.path.insert(0,'.')
from thermal_common import get_intersector
for case in sys.argv[1:]:
    cj=json.load(open(f"input/{case}/case.json"))
    v=trimesh.load(f"input/{case}/"+cj["files"]["vegetation_stl"],force="mesh")
    b=trimesh.load(f"input/{case}/"+cj["files"]["buildings_stl"],force="mesh")
    xyz=np.load(f"run_output/{case}/mrt_facet_out/path_xyz.npy")
    res={}
    for name,m in (("veg",v),("bldg",b)):
        it=get_intersector(m,quiet=True)
        # parity along 3 directions to be robust
        cnts=[]
        for d in ([0,0,1],[0.6,0.0,0.8],[-0.3,0.5,0.81]):
            d=np.array(d,float); d/=np.linalg.norm(d)
            loc,ir,_=it.intersects_location(xyz,np.tile(d,(len(xyz),1)),multiple_hits=True)
            dist=np.einsum('ij,ij->i',loc-xyz[ir],np.tile(d,(len(ir),1)))
            ir=ir[dist>1e-6]
            c=np.bincount(ir,minlength=len(xyz)); cnts.append(c%2)
        inside=(np.sum(cnts,axis=0)>=2)
        res[name]=inside
    tau=np.load(f"run_output/{case}/mrt_facet_out/direct_transmission_matrix.npy")[78]
    iv=res['veg']
    print(case,"pts",len(xyz),"inside veg crown %.2f%%"%(100*iv.mean()),"inside building %.2f%%"%(100*res['bldg'].mean()),
          "| 13:00 tau_dir inside-veg mean %.2f vs outside %.2f"%(tau[iv].mean() if iv.any() else np.nan, tau[~iv].mean()))
