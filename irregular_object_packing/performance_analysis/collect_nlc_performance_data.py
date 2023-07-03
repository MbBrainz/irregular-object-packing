
import os
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from itertools import product
from time import sleep, time

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from irregular_object_packing.packing.optimizer import default_optimizer_config

# Define the parameters
n_objects_list = [8, 16, 32, 64, 128, 256]
# n_objects_list.reverse()
stage_list = [0, 8]
stage_list.reverse()
n_threads_list = [1, 2, 4, 8, 16, 32, 64, 128]
iterations = 20
NO_NUMBA = bool(os.getenv("NUMBA_DISABLE_JIT")) or False

def generate_seeds(n):
    np.random.seed(0)
    return np.random.randint(0, 1000000, size=n)
# # Initialize the dataframe

# Function to run the commands and parse the output
def prepare_optimizer(n_objects, seed, data_dir):
    optimizer = default_optimizer_config(n_objects, mesh_dir=data_dir, seed=seed)
    optimizer.setup()
    return optimizer

@click.command()
@click.option('--output-dir', default='results/', help='Output file')
@click.option('--input-dir', default='data/mesh/', help='Input file')
def run(output_dir, input_dir):
    output_file = output_dir + f'collect_nlc_perf_data_numba{NO_NUMBA}.csv'
    df = pd.DataFrame(columns=['iteration', 'n_objects', 'stage', 'n_threads', 'runtime', 'numba'])
    df.to_csv(output_file, index=False)

    seeds = generate_seeds(iterations)
    for iteration in tqdm(range(iterations), total=iterations, position=0):
        for n_objects, stage in tqdm(product(n_objects_list, stage_list), total=len(n_objects_list)*len(stage_list), position=1):
            success = False
            while not success:
                try:
                    optimizer = prepare_optimizer(n_objects, seed=seeds[iteration], data_dir=input_dir)
                    optimizer.i_b = stage
                    optimizer.resample_meshes()
                    optimizer.executor = PoolExecutor(max_workers=16)
                    optimizer.perform_optimisation_iteration()
                    optimizer.tf_arrays = optimizer._tf_arrays(-1)
                    success = True
                except Exception as e:
                    print(e)

            for n_threads in tqdm(n_threads_list, total=len(n_threads_list), position=2):
                optimizer.config.n_threads = n_threads
                optimizer.executor = PoolExecutor(max_workers=n_threads, initializer=lambda: np.sqrt(2))
                sleep(0.1)

                runtime = time()
                # perform growth based optimisation
                optimizer.optimize_positions()
                runtime = time() - runtime

                # reset state
                optimizer.tf_arrays = optimizer._tf_arrays(-1)
                data = {'iteration':iteration, 'n_objects':n_objects, 'stage':stage, 'n_threads':n_threads, 'runtime':runtime, 'numba':NO_NUMBA}
                df = pd.DataFrame([data])
                df.to_csv(output_file, index=False, mode='a', header=False)

    pass

if __name__ == '__main__':
    run()
