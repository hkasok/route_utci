import numpy as np, trimesh, sys
for c in sys.argv[1:]:
    R='/media/harshin/data_drive/route based UTCI/'
    p=np.load(R+f'run_output/{c}/mrt_facet_out/path_xyz.npy')[::20]
    g=trimesh.load(R+f'input/{c}/geometry/ground_and_water_final.stl')
    o=np.column_stack([p[:,:2],np.full(len(p),3000.0)])
    loc,ri,_=g.ray.intersects_location(o,np.tile([0,0,-1.0],(len(o),1)),multiple_hits=False)
    zg=np.full(len(p),np.nan); zg[ri]=loc[:,2]
    dz=p[:,2]-zg
    print(c,'path z - ground: median %.3f min %.3f max %.3f'%(np.nanmedian(dz),np.nanmin(dz),np.nanmax(dz)))
