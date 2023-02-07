"""Analyse the results, output the data required for the article (CSV for SI, table for figures)"""

import numpy as np
import os
import pandas as pd
from sbp_modelling.single_source_low_r.model import SingleSourceModel
from sbp_modelling.single_source_low_r.forward_model import source_xz, a_x_s, a_z_s, seeds
from sbp_modelling.analysis import envelope

results_path = '../../../results/single-source-low-r/traces.npz'
results_path = os.path.realpath(os.path.join(os.path.dirname(__file__), results_path))
csv_path = '../../../results/single-source-low-r/rms-amplitudes.csv'
csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), csv_path))

loaded = np.load(results_path)
params = loaded['params']
data_raw = loaded['data'].T
data = envelope(data_raw)
t = loaded['t']

# Calculate the time offset (because source is not a spike at time zero) based on the peak amplitude
peaks = np.argmax(data, axis=0)
assert (peaks == peaks[0]).all()
t_offset = t[peaks[0]]

model = SingleSourceModel()
waterbottom, mtd_top, mtd_base = model.horizons()

def depth_to_time(z, shift=0., offset=True):
    v = model.elastic_model()[0]  # P-wave velocity vertical profile
    ind = (model.z <= z) & (model.z >= source_xz[1])
    owtt = np.sum(model.d_z / v[0][ind])
    if offset is True:
        offset = t_offset
    else:
        offset = 0
    return 2 * owtt + shift + offset

print(depth_to_time(waterbottom))

wb = depth_to_time(waterbottom)
t_start = depth_to_time(mtd_top)
t_end = depth_to_time(mtd_base)

if __name__ == '__main__':
    t_ind = (t > t_start) & (t < t_end)
    df = pd.DataFrame(columns=['a_x', 'a_z', 'rms_amplitude'])

    _, mtd_top_i, mtd_base_i = model.horizons(return_gridpoints=True)
    for a_x in a_x_s:
        for a_z in a_z_s:
            p_ind = (params[:, 0] == a_x) & (params[:, 1] == a_z)
            mtd_data = data[t_ind, :][:, p_ind]
            rms = np.sqrt(np.mean(np.square(mtd_data)))
            R_concat = np.array([])
            for seed in seeds:
                model = SingleSourceModel(a_x, a_z, seed)
                R = model.acoustic_reflectivity_model()
                R_concat = np.concatenate((R_concat, R[:, mtd_top_i:mtd_base_i].ravel()))

            print(R_concat.shape)

            rms_reflectivity = np.sqrt(np.mean(np.square(R_concat)))

            df = df.append({
                'a_x': a_x,
                'a_z': a_z,
                'rms_amplitude': rms,
                'rms_reflectivity': rms_reflectivity}, ignore_index=True)

    df.to_csv(csv_path, index=False)