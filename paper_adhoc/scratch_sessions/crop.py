import laspy, numpy as np, glob, sys
cx,cy,half=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3])
L='/media/harshin/data_drive/route based UTCI/validation data/Liseben/LAZ/'
veg=[];gnd=[]
for f in sorted(glob.glob(L+'*.laz')):
    if '(' in f: continue
    with laspy.open(f) as r:
        h=r.header
        if h.maxs[0]<cx-half or h.mins[0]>cx+half or h.maxs[1]<cy-half or h.mins[1]>cy+half: continue
        print('tile',f.split('/')[-1])
        for ch in r.chunk_iterator(2_000_000):
            x=np.asarray(ch.x);y=np.asarray(ch.y);z=np.asarray(ch.z);c=np.asarray(ch.classification)
            s=(np.abs(x-cx)<half)&(np.abs(y-cy)<half)
            v=s&np.isin(c,[3,4,5]); g=s&np.isin(c,[2,9])
            veg.append(np.column_stack([x[v]-cx,y[v]-cy,z[v]])); gnd.append(np.column_stack([x[g]-cx,y[g]-cy,z[g]]))
veg=np.concatenate(veg);gnd=np.concatenate(gnd)
np.save('veg_full.npy',veg); np.save('gnd_full.npy',gnd)
import importlib.util
spec=importlib.util.spec_from_file_location('clg','/media/harshin/data_drive/route based UTCI/create_lisbon_laz_geometry.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
np.save('veg_thin.npy',m.reduce_xy_grid(veg,0.5,True)); np.save('gnd_thin.npy',m.reduce_xy_grid(gnd,1.5,False))
print('veg',len(veg),'thin',len(np.load('veg_thin.npy')),'gnd',len(gnd))
