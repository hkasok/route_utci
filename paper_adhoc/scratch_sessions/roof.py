import trimesh, numpy as np, sys
case=sys.argv[1]; base=f'/media/harshin/data_drive/route based UTCI/input/{case}/geometry/'
b=trimesh.load(base+'building_final.stl'); g=trimesh.load(base+'ground_and_water_final.stl')
bodies=b.split(only_watertight=False)
rows=[]
for bd in bodies:
    zb=bd.vertices[:,2].min(); zt=bd.vertices[:,2].max()
    v=bd.vertices[np.abs(bd.vertices[:,2]-zb)<1e-3]
    o=np.column_stack([v[:,0],v[:,1],np.full(len(v),1000.0)])
    loc,ri,_=g.ray.intersects_location(o,np.tile([0,0,-1.0],(len(o),1)),multiple_hits=False)
    zg=loc[:,2]
    area=bd.area_faces[bd.face_normals[:,2]>0.9].sum()
    rows.append((area, zg.max()-zg.min(), zt-zg.max(), zt-zg.min(), zt-zb))
r=np.array(rows); w=r[:,0]
print(case,'n',len(r),'total roof area %.0f'%w.sum())
print('  ground range under footprint (area-weighted median %.1f m, p90 %.1f); frac roof area on buildings with range>5m %.2f, >10m %.2f'%(np.median(np.repeat(r[:,1],np.maximum(1,(w/50).astype(int)))), np.percentile(np.repeat(r[:,1],np.maximum(1,(w/50).astype(int))),90),w[r[:,1]>5].sum()/w.sum(),w[r[:,1]>10].sum()/w.sum()))
big=np.argsort(-w)[:5]
for i in big: print('   area %.0f ground range %.1f  roof above highest ground %.1f, above lowest ground %.1f, extrusion %.1f'%tuple(r[i]))
