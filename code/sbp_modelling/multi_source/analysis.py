"""Analyse the results, output the data required for the article (CSV for SI, table for figures)"""

import numpy as np
import pandas as pd
from sbp_modelling.multi_source.model import MultiSourceModel
from sbp_modelling.multi_source.forward_model import a_x_s, a_z_s, seeds
from sbp_modelling.analysis import envelope
import os

results_path = '../../../results/multi-source'
csv_path = '../../../results/multi-source/rms-amplitudes.csv'

def path(a_x, a_z, seed, results_path):
    return os.path.join(results_path, f'a_x_{a_x}_a_z_{a_z}_seed_{seed}.npz')

def load_run(a_x, a_z, seed, base_path=results_path):
    loaded = np.load(path(a_x, a_z, seed, base_path))
    return loaded['t'], loaded['data'], loaded['auv_sources']

def depth_to_time(z):
    return 0.5 + 2 * (z - model.z[0]) / 1.5

model = MultiSourceModel(a_x_s[0], a_z_s[0], seeds[0])
waterbottom, mtd_x, mtd_z, mtd_top, mtd_base = model.horizons()
mtd = model.x[mtd_x][0], model.x[mtd_x][-1], model.z[mtd_z][0], model.z[mtd_z][-1]

def top_horizon(x, shift=5, depth=False):
    if depth is True:
        return np.interp(x, model.x[mtd_x], mtd_top[mtd_x])
    else:
        return np.interp(x, model.x[mtd_x], shift + depth_to_time(mtd_top[mtd_x]))

def base_horizon(x, shift=-2, depth=False):
    if depth is True:
        return np.interp(x, model.x[mtd_x], mtd_base[mtd_x])
    else:
        return np.interp(x, model.x[mtd_x], shift + depth_to_time(mtd_base[mtd_x]))

def lower_top_horizon(x, shift=0, depth=False):
    # The second window, below MTD
    if depth is True:
        return np.full(np.array(x).size, 942.)
    else:
        return np.full(np.array(x).size, shift + 190.)

def lower_base_horizon(x, shift=0, depth=False):
    # The second window, below MTD
    if depth is True:
        return np.full(np.array(x).size, 958.)
    else:
        return np.full(np.array(x).size, shift + 212.)

if __name__ == '__main__':
    twtt, data, sources = load_run(a_x_s[0], a_z_s[0], seeds[0])

    trace_x = sources[:, 0]
    trace_x = trace_x[trace_x >= mtd[0]]
    trace_x = trace_x[trace_x <= mtd[1]]

    top = top_horizon(trace_x)
    base = base_horizon(trace_x)
    top_z = top_horizon(trace_x, depth=True)
    base_z = base_horizon(trace_x, depth=True)

    lower_top = lower_top_horizon(trace_x)
    lower_base = lower_base_horizon(trace_x)
    lower_top_z = lower_top_horizon(trace_x, depth=True)
    lower_base_z = lower_base_horizon(trace_x, depth=True)

    df = pd.DataFrame(columns=['a_x', 'a_z', 'seed', 'rms_amplitude', 'lower_rms', 'rms_reflectivity', 'lower_reflectivity'])

    i = 0
    for a_x in a_x_s:
        for a_z in a_z_s:
            for seed in seeds:
                twtt, data, _ = load_run(a_x, a_z, seed)
                print(a_x, a_z, seed, twtt.shape, data.shape, _.shape)
                data = envelope(data)

                means = []
                lower = []

                model2 = MultiSourceModel(a_x, a_z, seed,
                                          left=mtd[0], right=mtd[1], top=top_z.min(), base=lower_base_z.max(),
                                          subsampling=(1, 1))
                R = model2.acoustic_reflectivity_model()
                R_concat = np.array([])
                R_concat_lower = np.array([])

                for j, x in enumerate(trace_x):
                    t_ind = (twtt >= top[j]) & (twtt <= base[j])
                    z_ind = (model2.z >= top_z[j]) & (model2.z <= base_z[j])
                    z_ind_lower = (model2.z >= lower_top_z[j]) & (model2.z <= lower_base_z[j])
                    z_ind = z_ind[:-1] # Reflectivity field is one shorter than depths...
                    z_ind_lower = z_ind_lower[:-1]

                    means.append(np.sqrt(np.mean(np.square(data[:, sources[:, 0] == x][t_ind].ravel()))))

                    t_ind = (twtt >= lower_top[j]) & (twtt <= lower_base[j])
                    lower.append(np.sqrt(np.mean(np.square(data[:, sources[:, 0] == x][t_ind].ravel()))))

                    # Get acoustic reflectivity
                    #print(z_ind.shape, R.shape)
                    R_concat = np.concatenate((R_concat, R[j, z_ind].ravel()))
                    R_concat_lower = np.concatenate((R_concat_lower, R[j, z_ind_lower].ravel()))

                rms_reflectivity = np.sqrt(np.mean(np.square(R_concat)))
                upper_reflectivity = np.sqrt(np.mean(np.square(R_concat_lower)))

                df.loc[i] = (a_x, a_z, seed, np.array(means).mean(), np.array(lower).mean(), rms_reflectivity, upper_reflectivity)
                i += 1
    df.to_csv(csv_path, index=False)
    print(df)