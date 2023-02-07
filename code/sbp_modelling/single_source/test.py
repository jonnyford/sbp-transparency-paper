import os
import time
import numpy as np

slurm_env_vars = 'SLURM_CPUS_ON_NODE', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_JOB_NODELIST',\
    'SLURM_JOB_NUM_NODES', 'SLURM_LOCALID', 'SLURM_NODEID', 'SLURM_NTASKS', 'SLURM_PROCID', 'SLURM_SUBMIT_DIR',\
    'SLURM_SUBMIT_HOST', 'SLURM_TASKS_PER_NODE', 'SLURM_ARRAY_TASK_ID', 'SLURM_ARRAY_TASK_COUNT', 'SLURM_ARRAY_JOB_ID'\
    'OMP_NUM_THREADS', 'SCRATCH', 'HOSTNAME', 'SBP_CACHE_PATH', 'SBP_PATH', 'CONDA_PREFIX'

a = range(10)
b = range(10)

jobs = []
for m in a:
    for n in b:
        jobs.append((m, n))

def run(a, b):
    print(f'{a} x {b} = {a * b}')

if __name__ == '__main__':
    time.sleep(0.1)
    for key in slurm_env_vars:
        try:
            value = os.environ[key]
        except KeyError:
            value = '[not set]'

        print(f'${key} = {value}')

    task = int(os.environ['SLURM_ARRAY_TASK_ID']) - 1
    n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    job_tasks = np.array_split(jobs, n_tasks)
    for job in job_tasks[task]:
        run(*job)

    time.sleep(0.1)