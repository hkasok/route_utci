import numpy as np, solweig, rasterio, glob, os
from datetime import datetime
n=40
dsm=np.full((n,n),10.0,np.float32); dem=dsm.copy()
lc=np.full((n,n),1,np.int32)  # paved?
loc=solweig.Location(latitude=38.7,longitude=-9.2,utc_offset=1)
res={}
for post in ('standing','sitting'):
  for az in (180.0,225.0,250.0):
    surface=solweig.SurfaceData.prepare(dsm=dsm,dem=dem,land_cover=lc,pixel_size=1.0,working_dir=f'sw/cache',dsm_relative=False)
    w=[solweig.Weather(datetime=datetime(2023,6,24,15,0),ta=30.0,rh=50.0,ws=1.5,global_rad=800.0,timestep_minutes=10,
        measured_direct_rad=850.0,measured_diffuse_rad=100.0,precomputed_sun_altitude=50.0,precomputed_sun_azimuth=az,precomputed_altmax=75.0)]
    od=f'sw/out_{post}_{int(az)}'; os.makedirs(od,exist_ok=True)
    solweig.calculate(surface=surface,weather=w,location=loc,output_dir=od,human=solweig.HumanParams(posture=post,abs_k=0.95,abs_l=0.957),outputs=['tmrt','kdown','shadow'])
    f=sorted(glob.glob(od+'/tmrt/*.tif'))[0]
    T=rasterio.open(f).read(1)[n//2,n//2]
    res[(post,az)]=0.957*5.67e-8*(T+273.15)**4
    print(post,az,T,res[(post,az)],flush=True)
