import numpy as np
import pickle
from sbp_modelling.multi_source.forward_model import results_path, a_x_s, a_z_s, seeds, auv_sources
from sbp_modelling.multi_source.model import z_0, v_p_water
from pathlib import Path
import os

n_tasks = None

try:
    slurm_id = os.environ['SLURM_JOB_ID']
    with open(Path(__file__).parent.resolve() / 'multi_source.job') as f:
        for line in f.readlines():
            if '#SBATCH --array=' in line:
                n_tasks = int(line.split('=')[1].split('-')[1])
except (FileNotFoundError, KeyError):
    n_tasks = 1

if __name__ == '__main__':
    results = []
    params = []
    twtts = []

    # Load and merge
    for task in range(n_tasks):
        print(f'=== Merging chunk {task + 1}/{n_tasks} ===')
        path = results_path / f'traces-tasks-{n_tasks:d}-task-{task:d}.p'
        with open(path, 'rb') as f:
            out = pickle.load(f)

        twtts.extend(out['twtts'])
        params.extend(out['params'])
        results.extend(out['results'])

    params = np.array(params)
    print(params.shape)

    shifts = 2 * (z_0 - auv_sources[:, 1]) / v_p_water

    for a_x in a_x_s:
        for a_z in a_z_s:
            for seed in seeds:
                ind = np.argwhere((params[:, 0] == a_x) & (params[:, 1] == a_z) & (params[:, 2] == seed)).ravel()

                _twtts = []
                _results = []

                for i in ind:
                    _twtts.append(twtts[i])
                    _results.append(results[i])

                min_twtt = np.inf
                max_twtt = -np.inf

                for times in _twtts:
                    if min(times) < min_twtt:
                        min_twtt = min(times)
                    if max(times) > max_twtt:
                        max_twtt = max(times)
                max_twtt -= max(shifts)
                twtt_d_t = twtts[0][1] - twtts[0][0]

                print(min_twtt, max_twtt, twtt_d_t)
                t = np.arange(min_twtt, max_twtt + twtt_d_t, twtt_d_t)
                a = np.zeros((t.size, len(_results)))

                for i in range(len(_results)):
                    ind = _twtts[i] > 40
                    a[:, i] = np.interp(t, _twtts[i][ind] - shifts[i], np.array(_results[i])[ind])

                np.savez_compressed(results_path / f'a_x_{a_x}_a_z_{a_z}_seed_{seed}.npz',
                                    params=np.array((a_x, a_z, seed)),
                                    auv_sources=auv_sources,
                                    t=np.array(t),
                                    data=a)