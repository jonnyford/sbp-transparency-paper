from devito import *
from examples.seismic import TimeAxis
import numpy as np
import os
from sbp_modelling.model import run_single_channel_modelling
from sbp_modelling.single_source_p_low.model import SingleSourceModel
from multiprocessing import Pool
from pathlib import Path

try:
    results_path = Path(os.environ['SBP_PATH']) / 'results/single-source-p-low'
except KeyError:
    results_path = Path(__file__).parent.resolve() / '../../../results/single-source-p-low'

assert results_path.exists()

# Ensure consistent timestep for all models
scratch_model = SingleSourceModel()
d_t = scratch_model.devito_model().critical_dt

f_0 = 1.5
source_xz = scratch_model.n_x * scratch_model.d_x / 2, scratch_model.h * scratch_model.d_z

t_max = 2 * ((scratch_model.n_z * scratch_model.d_z) - (scratch_model.h * scratch_model.d_z)) / 1.5
delay = 1.5 / f_0

a_x_s = np.logspace(-3, 4, 8)
a_z_s = 0.01, 0.05, 0.1, 0.5, 1.
seeds = 1000 + np.arange(10)

print(len(a_x_s) * len(a_z_s) * len(seeds))

def run_run(a_x, a_z, seed):
    a_x = float(a_x)
    a_z = float(a_z)
    seed = int(seed)
    model = SingleSourceModel(a_x, a_z, seed)
    time_range = TimeAxis(start=scratch_model.t_0, stop=(t_max + delay), step=d_t)
    print(time_range)

    assert f_0 <= 1.5 / (model.d_x * 14)

    print(f'Running a_x={a_x} a_z={a_z} seed={seed}')
    model.realize_binary_field()
    return run_single_channel_modelling(
        model.devito_model(nbl=500), f_0, source_xz, time_range, time_order=model.to)

if __name__ == '__main__':
    try:
        n_threads = int(os.environ['SBP_NUM_THREADS'])
    except KeyError:
        n_threads = 1
    print(f'n_threads={n_threads}')

    results = []
    params = []

    for a_x in a_x_s:
        for a_z in a_z_s:
            for seed in seeds:
                params.append([a_x, a_z, seed])

    #p = Pool(n_threads)

    try:
        task = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
        n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        params_chunks = np.array_split(params, n_tasks)
        print(f'=== Processing chunk {task + 1}/{n_tasks} ===')

        run_return = []

        results = []
        t = None

        for a_x, a_z, seed in params_chunks[task]:
            twtt, result = run_run(a_x, a_z, seed)
            t = twtt
            results.append(np.array(result))

        np.savez_compressed(os.path.join(results_path, f'traces-tasks-{n_tasks:d}-task-{task:d}.npz'),
                            params=np.array(params_chunks[task]),
                            t=np.array(t),
                            data=np.array(results))
    except KeyError:
        #run_return = p.starmap(run_run, params)
        print('Halp')
