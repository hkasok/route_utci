import numpy as np, pandas as pd, trimesh, json, sys
sys.path.insert(0,'.')
from thermal_common import get_intersector, sun_vector_enu
for case in sys.argv[1:]:
    cj=json.load(open(f"input/{case}/case.json"))
    g=trimesh.load(f"input/{case}/"+cj["files"]["ground_stl"],force="mesh")
    gi=get_intersector(g,quiet=True)
    m=f"run_output/{case}/mrt_facet_out"
    xyz=np.load(m+"/path_xyz.npy")[::4]
    tau=np.load(m+"/direct_transmission_matrix.npy")[:, ::4]
    t=pd.read_csv(m+"/times.csv")
    out=[]
    for it in range(len(t)):
        el=t.elevation_deg[it]
        if el<=2: continue
        if it%3: continue
        sv=sun_vector_enu(t.azimuth_deg[it],el)
        o=xyz+1e-3*sv
        hit=gi.intersects_any(o,np.tile(sv,(len(o),1)))
        lit=tau[it]>0
        out.append((t.time[it][11:16],el,hit.mean(),(hit&lit).mean()))
    df=pd.DataFrame(out,columns=["t","el","terrain_blocked","blocked_but_model_lit"])
    print(case); print(df[(df.blocked_but_model_lit>0.0)].to_string(index=False))
