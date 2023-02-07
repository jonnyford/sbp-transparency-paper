from examples.seismic import SeismicModel, TimeAxis, RickerSource, Receiver, plot_velocity, demo_model
import numpy as np
from sbp_modelling.model import run_single_channel_modelling, v_p_water, v_p_0, v_p_1, v_p_v_s_ratio
from sbp_modelling.multi_source.model import MultiSourceModel, pad, x_0, x_1, z_0, z_1, get_horizon, df
from scipy.ndimage import gaussian_filter1d
import os
import pickle
from pathlib import Path

try:
    results_path = Path(os.environ['SBP_PATH']) / 'results/multi-source'
except KeyError:
    results_path = Path(__file__).parent.resolve() / '../../../results/multi-source'

scratch = MultiSourceModel()

a_z_s = 0.05,
a_x_s = scratch.no_aniso, 1000., 100., 10., 1., 0.5, 0.1, 0.05, #scratch.no_aniso, 10000., 1000., 100., 10., 1., 0.1, 0.05,
seeds = 3021, 3022, 3023, 3024, 3025

f_0 = 1.5 #3.5 # Source dominant frequency (kHz)
auv_d_x = 2. # 2.
temp_x = np.arange(x_0, x_1, 1.0)
waterbottom = np.interp(temp_x, *get_horizon(df, 'Seabed').to_numpy().T)
smoothed_waterbottom = gaussian_filter1d(waterbottom, 200)
flight_height = 40
auv_x = np.arange(1000., 4001., auv_d_x)
auv_sources = np.vstack((auv_x, np.interp(auv_x, temp_x, smoothed_waterbottom) - flight_height)).T

def run_run(a_x, a_z, seed, source_x, source_z):
    left = source_x - pad
    right = source_x + pad
    top = source_z
    base = z_1 # Effective base of model for ms figures

    if left < x_0:
        left = x_0
    if right > x_1:
        right = x_1
    if top < z_0:
        top = z_0
    if base > z_1:
        base = z_1

    j = int(np.where(np.all(auv_sources == (source_x, source_z), axis=1))[0])

    realisation = MultiSourceModel(a_x, a_z, seed, left=left, right=right, top=top, base=base)

    assert isinstance(realisation, MultiSourceModel)
    realisation.realize_model()

    source_xz = np.array([[source_x - left, source_z - top]])
    max_t = 2 * (base - source_z) / v_p_water
    assert f_0 <= v_p_water / (realisation.d_x * 6)

    model = realisation.devito_model(demo=False, x_0=0, z_0=0, nbl=int(15 / realisation.d_x)) # nbl=125
    print(f'{source_xz} ({model.grid.shape}) for {max_t} ms (source {f_0} kHz)')

    C = 0.5
    d_t = C * realisation.d_x / max(v_p_0, v_p_1)
    time_range = TimeAxis(start=realisation.t_0, stop=max_t, step=d_t)

    return run_single_channel_modelling(model, f_0, source_xz, time_range, time_order=realisation.to)

if __name__ == '__main__':
    params = []
    for a_x in a_x_s:
        for a_z in a_z_s:
            for seed in seeds:
                for source_x, source_z in auv_sources:
                    params.append([a_x, a_z, seed, source_x, source_z])

    try:
        task = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
        #n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        with open(Path(__file__).parent.resolve() / 'multi_source.job') as f:
            for line in f.readlines():
                if '#SBATCH --array=' in line:
                    n_tasks = int(line.split('=')[1].split('-')[1])
    except:
        task = 0
        n_tasks = 1

    params_chunks = np.array_split(params, n_tasks)
    print(f'=== Processing chunk {task + 1}/{n_tasks} ===')

    results = []
    twtts = []

    for i, (a_x, a_z, seed, source_x, source_z) in enumerate(params_chunks[task]):
        print(f'Running a_x={a_x} a_z={a_z} seed={seed} shot={i+1}/{len(params)} x={source_x} z={source_z}')

        twtt, result = run_run(a_x, a_z, int(seed), source_x, source_z)
        twtts.append(twtt)
        results.append(np.array(result))

    out = {
        'params': params_chunks[task],
        'twtts': twtts,
        'results': results
    }

    with open(os.path.join(results_path, f'traces-tasks-{n_tasks:d}-task-{task:d}.p'), 'wb') as f:
        pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
