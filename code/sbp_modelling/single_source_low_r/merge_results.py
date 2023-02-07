import numpy as np
from sbp_modelling.single_source_low_r.forward_model import results_path, a_x_s, a_z_s, seeds
from pathlib import Path

n_tasks = None

with open(Path(__file__).parent.resolve() / 'single_source_low_r.job') as f:
    for line in f.readlines():
        if '#SBATCH --array=' in line:
            n_tasks = int(line.split('=')[1].split('-')[1])

if n_tasks is None:
    raise Exception('Cannot determine number of batch jobs')

if __name__ == '__main__':
    results = []
    params = []

    for task in range(n_tasks):
        print(f'=== Merging chunk {task + 1}/{n_tasks} ===')
        path = results_path / f'traces-tasks-{n_tasks:d}-task-{task:d}.npz'
        a = np.load(path)
        t = a['t']
        params.extend(a['params'])
        results.extend(a['data'])

    t = np.array(t)
    params = np.array(params)
    results = np.array(results)

    print(t.shape, params.shape, results.shape)
    np.savez_compressed(results_path / 'traces.npz', params=params, t=t, data=results)
