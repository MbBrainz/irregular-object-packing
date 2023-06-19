import subprocess
from itertools import product
from os import remove

import click
import pandas as pd
from tqdm import tqdm

# Define the parameters
# n_objects_list = [64] #, 128, 256]
n_objects_list = [8, 16, 32, 64, 128, 256]
# n_objects_list.reverse()
stage_list = [0, 8]
# stage_list.reverse()
n_threads_list = [1, 2, 4, 8, 16, 32, 64, 128]
iterations = 10

# Test Parameters
# n_objects_list = [4 ]#, 8, 16, 32, 64, 128, 256]
# stage_list = [0,  ]#1, 2, 3, 4, 5, 6, 7, 8]
# n_threads_list = [1]
# iterations = 3

output_file = 'results/profile_optimizer_funcs_large0-8.csv'
# # Initialize the dataframe
output_file = 'results/profile_optimizer_funcs_large0-8.csv'
# df = pd.DataFrame(columns=['iteration', 'n_objects', 'stage', 'n_threads', 'current_meshes', 'compute_cdt', 'compute_cat_cells', 'optimize_positions', 'process_iteration'])
# df.to_csv(output_file, index=False)

# Function to run the commands and parse the output
def run_and_parse_functional_profiling(n_objects, stage, n_threads):
    command1 = f"kernprof -l irregular_object_packing/packing/optimizer.py --n_objects {n_objects} --stage {stage} --n_threads {n_threads}"
    command2 = "python -m line_profiler optimizer.py.lprof"

    _ = subprocess.run(command1, shell=True, check=True, capture_output=True, text=True).stdout
    output = subprocess.run(command2, shell=True, check=True, capture_output=True, text=True).stdout

    # Split output into lines
    lines = output.split('\n')

    # Dictionary to store times
    times = {}

    # Iterate over the lines
    for line in lines:
        # Split the line into columns (assuming they are separated by spaces)
        columns = line.split(maxsplit=5)
        if len(columns) < 6:
            continue

        # Check if the line number corresponds to one of the desired lines
        time = columns[2]
        line_of_code = columns[5]

        if 'self.current_meshes' in line_of_code:
            times['current_meshes'] = float(time)
        elif 'cat.compute_cdt' in line_of_code:
            times['compute_cdt'] = float(time)
        elif 'self.compute_cat_cells' in line_of_code:
            times['compute_cat_cells'] = float(time)
        elif 'self.optimize_positions' in line_of_code:
            times['optimize_positions'] = float(time)
        elif 'self.compute_violations' in line_of_code:
            times['process_iteration'] = float(time)

    return times

# Loop over the parameter combinations
def collect_profile_data(n_objects_list, stage_list, n_threads_list, iterations, output_file, input_dir):
    df = pd.DataFrame(columns=['iteration', 'n_objects', 'stage', 'n_threads', 'current_meshes', 'compute_cdt', 'compute_cat_cells', 'optimize_positions', 'process_iteration'])
    df.to_csv(output_file, index=False)

    for iteration in tqdm(range(iterations), total=iterations, desc='iterations'):
        for (n_objects, stage, n_threads) in tqdm(product(n_objects_list, stage_list, n_threads_list), total = len(n_objects_list) * len(stage_list) * len(n_threads_list), desc='profile data'):
            rows = []
            # Run the commands and parse the output
            times = run_and_parse_functional_profiling(n_objects, stage, n_threads)
            print(f"collected_times: {times}")

            # Append the data to the dataframe
            rows.append({'iteration': iteration, 'n_objects': n_objects, 'stage': stage, 'n_threads': n_threads, **times},)

            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False, mode='a', header=False)

    remove('optimizer.py.lprof')
# Run the function

@click.command()
@click.option('--output-dir', type=click.Path(exists=True), default='results/')
@click.option('--input-dir', default='data/mesh/', help='Input file')
def run(output_dir, input_dir):
    output_file = f'{output_dir}profile_optimizer_funcs_large-0-8-snellius.csv'
    collect_profile_data(n_objects_list, stage_list, n_threads_list, iterations, output_file, input_dir)

if __name__ == '__main__':
    run()


# Save the dataframe to a CSV file
