import trimesh, numpy as np, sys
case=sys.argv[1]; base=f'/media/harshin/data_drive/route based UTCI/input/{case}/geometry/'
b=trimesh.load(base+'building_final.stl'); g=trimesh.load(base+'ground_and_water_final.stl')
bodies=b.split(only_watertight=False)
gaps=[];perim_frac=[];maxgap=[]
allv=[]
for bd in bodies:
    zb=bd.vertices[:,2].min()
    v=bd.vertices[np.abs(bd.vertices[:,2]-zb)<1e-3]
    allv.append(np.column_stack([v,np.full(len(v),zb)]))
V=np.vstack(allv)
o=np.column_stack([V[:,0],V[:,1],np.full(len(V),1000.0)])
# offset slightly outward is hard; sample ground at the vertex location (ground continues under buildings by NN fill)
loc,ri,_=g.ray.intersects_location(o,np.tile([0,0,-1.0],(len(o),1)),multiple_hits=False)
zg=np.full(len(o),np.nan); zg[ri]=loc[:,2]
gap=V[:,3]-zg   # >0: floor above ground at this footprint corner -> open gap under wall
print(case,'buildings',len(bodies),'footprint corners',len(V))
print('  floor-above-ground gap at corners: median %.2f p90 %.2f p99 %.2f max %.2f; frac corners gap>0.5m %.3f, >1m %.3f, >2m %.3f'%(np.nanmedian(gap),np.nanpercentile(gap,90),np.nanpercentile(gap,99),np.nanmax(gap),(gap>0.5).mean(),(gap>1).mean(),(gap>2).mean()))
