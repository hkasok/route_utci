import numpy as np, solweig, rasterio, glob, os
from datetime import datetime
n=40
dsm=np.full((n,n),10.0,np.float32); dem=dsm.copy(); lc=np.full((n,n),1,np.int32)
loc=solweig.Location(latitude=38.7,longitude=-9.2,utc_offset=1)
for post in ('sitting',):
  for I in (0.0,850.0):
    surface=solweig.SurfaceData.prepare(dsm=dsm,dem=dem,land_cover=lc,pixel_size=1.0,working_dir='sw/cache',dsm_relative=False)
    w=[solweig.Weather(datetime=datetime(2023,6,24,15,0),ta=30.0,rh=50.0,ws=1.5,global_rad=800.0,timestep_minutes=10,
        measured_direct_rad=I,measured_diffuse_rad=100.0,precomputed_sun_altitude=50.0,precomputed_sun_azimuth=180.0,precomputed_altmax=75.0)]
    od=f'sw/o2_{post}_{int(I)}'; os.makedirs(od,exist_ok=True)
    solweig.calculate(surface=surface,weather=w,location=loc,output_dir=od,human=solweig.HumanParams(posture=post,abs_k=0.95,abs_l=0.957),outputs=['tmrt','kdown','kup','shadow'])
    g=lambda v: rasterio.open(sorted(glob.glob(f'{od}/{v}/*.tif'))[0]).read(1)[n//2,n//2]
    T=g('tmrt'); print(post,I,'S',0.957*5.67e-8*(T+273.15)**4,'kdown',g('kdown'),'kup',g('kup'),'shadow',g('shadow'))
