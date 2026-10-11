import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np
import paper_globe_attribution as pa, paper_validation_figures as pv
pts=pv.load_points(Path('.')); pts,_=pv.apply_measurement_qc(pts)
walks=pa.walk_frames(pts); all_day=pa.add_sample_quantities(walks, Path('.'))
d=pa.usable_day(all_day)
for R in (int(sys.argv[1]),):
    rng=np.random.default_rng(20261003)
    boots=pa.bootstrap(d, lambda s: np.asarray(pa.fit_regression(s)["coef_per_sd_K"]), R, rng)
    spin=pts[~pts.globe_spinup_affected.astype(bool)]
    s=spin[spin.period=='day'].dropna(subset=['globe_transient_temperature_C','measured_black_globe_temperature_c']).reset_index(drop=True)
    b=pa.bootstrap(s, lambda f: np.array([pv.stats(f.globe_transient_temperature_C.to_numpy(float), f.measured_black_globe_temperature_c.to_numpy(float))['mbe']]), R, rng)
    print(R, pa.ci(b[:,0]), [pa.ci(boots[:,j]) for j in range(5)][:2])
