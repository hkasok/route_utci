import sys; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np
import paper_globe_attribution as pa, paper_validation_figures as pv
pts=pv.load_points(Path('.')); pts,_=pv.apply_measurement_qc(pts)
walks=pa.walk_frames(pts); all_day=pa.add_sample_quantities(walks, Path('.'))
d=pa.usable_day(all_day)
base=pa.fit_regression(d)
print('orig coef/SD', np.round(base['coef_per_sd_K'],3), 'R2 %.3f'%base['r2'])
d['measured_tg_minus_ta_K']=d.globe_transient_temperature_C-d.trec_route_air_temperature_c
alt=pa.fit_regression(d)
print('model-side dT coef/SD', np.round(alt['coef_per_sd_K'],3), 'R2 %.3f'%alt['r2'])
