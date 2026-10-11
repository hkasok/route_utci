import numpy as np, solweig, rasterio, glob, os, shutil, logging
logging.disable(logging.WARNING)
from datetime import datetime
n=30
dsm=np.full((n,n),10.0,np.float32); dem=dsm.copy(); lc=np.full((n,n),1,np.int32)
loc=solweig.Location(latitude=38.7,longitude=-9.2,utc_offset=1)
def S(alt,az,I,abs_k=0.95):
    surface=solweig.SurfaceData.prepare(dsm=dsm,dem=dem,land_cover=lc,pixel_size=1.0,working_dir='cache',dsm_relative=False)
    w=[solweig.Weather(datetime=datetime(2023,6,24,15,0),ta=30.0,rh=50.0,ws=1.5,global_rad=I*np.sin(np.radians(alt))+100,timestep_minutes=10,
        measured_direct_rad=I,measured_diffuse_rad=100.0,precomputed_sun_altitude=alt,precomputed_sun_azimuth=az,precomputed_altmax=75.0)]
    od=f'o_{alt}_{az}_{I}_{abs_k}'; shutil.rmtree(od,ignore_errors=True); os.makedirs(od)
    solweig.calculate(surface=surface,weather=w,location=loc,output_dir=od,human=solweig.HumanParams(posture='sitting',abs_k=abs_k,abs_l=0.957),outputs=['tmrt','kup'])
    g=lambda v: float(rasterio.open(sorted(glob.glob(f'{od}/{v}/*.tif'))[0]).read(1)[n//2,n//2])
    T=g('tmrt'); return 0.957*5.670374419e-8*(T+273.15)**4, g('kup'), T
for alt in (30,50,70):
    base={}
    for az in (180,225,250):
        s1,k1,_=S(alt,az,850.); s0,k0,_=S(alt,az,0.)
        h=np.radians(alt); a=np.radians(az)
        cube=(np.sin(h)+np.cos(h)*(abs(np.sin(a))+abs(np.cos(a))))/6
        old=0.20*np.cos(h)+np.sin(h)/6
        print(f"alt {alt} az {az}: dS/(0.95 I) = {(s1-s0)/(0.95*850):.4f}  dKup {k1-k0:.1f}  cube {cube:.4f} old {old:.4f}")
# linearity in abs_k (for the unclipping trick)
sa,_,Ta=S(60,240,950.,0.475); sb,_,Tb=S(60,240,950.,0.2375); sf,_,Tf=S(60,240,950.,0.95)
print('linearity: full', sf, ' 3Sa-2Sb', 3*sa-2*sb, Tf, Ta, Tb)
