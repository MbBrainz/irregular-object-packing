from dataclasses import dataclass

import pandas as pd
from dataclass_csv import DataclassWriter
from numpy import array, ndarray
from pandas import DataFrame

CONFIG = {
    "number_of_iterations": 1,
    "title": "test",
    "result_dir": "results",
    "temp_output_dir": "temp_output",
    "data_dir": "../data",
}

CASE_PARAMETER_SEARCH = {
    'n_objects': [5],
    'container': ['sphere', 'cube',],
    # 'n_objects': [5, 6, 7, 8, 9,],
    # 'container': ["cube", 'cylinder', 'sphere'],
    'shape': [ "normal_red_blood_cell"],
    # 'shape': ["cube", 'sphere', "normal_red_blood_cell", "sickle_blood_cell", "bunny"],
    # 'padding': [0, 0.0005, 0.0001, 0.00005, 0.00001],
    'padding': [0, 0.00005, 0.00001],
    'alpha': [0.1],
    'beta': [0.5],
    'n_threads': [128],
    # 'alpha': [0.01, 0.05, 0.1],
    # 'beta': [0.01, 0.05, 0.1],
}

# could do a extra field per parameter that defines if you want combinations or iterations
CASE_TRIVIAL_SHAPES = {
    'n_objects': 1,
    'container': ['cube','sphere','cylinder', 'tetrahedron', 'cone'],
    'shape': ["cube", 'sphere', "cylinder", 'tetrahedron', 'cone'],
    'padding': 0,
    'alpha': 0.1,
    'beta': 0.5,
    'n_threads': 8,
}

CASE_BLOODCELL_MAX = {
    'n_objects': [2, 4, 8, 16, 32, 64],
    'container': ['cylinder', 'cube'],
    'shape': ["normal_red_blood_cell"],
    'padding': [0],
    'alpha': [0.1],
    'beta': [0.5],
    'n_threads': [128],
}

RESULTS = {
    'i': int,
    'run_time': float,
    'setup_time': float,
    'n_total_steps': int,
    'time_per_step': ndarray[int],
    'its_per_step': ndarray[float],
    'fails_per_step': ndarray[int],
    'errors_per_step': ndarray[float],
}

@dataclass
class ResultData:
    # PARAMETERS
    n_objects: int
    container: str
    shape: str
    padding: float
    alpha: float
    beta: float
    n_threads: int

    # COLLECTED_DATA
    i: int = 0
    run_time: float = 0
    setup_time: float = 0
    n_total_steps: int = 0
    time_per_step: ndarray[float] = array([])
    iterations_per_step: ndarray[float] = array([])
    fails_per_step: ndarray[float] = array([])
    errors_per_step: ndarray[float] = array([])

    # Custom param for irop or cellpack
    implementation: str = "irop"

    @staticmethod
    def write_csv(file_path):
        with open(file_path, 'w') as f:
            writer = DataclassWriter(f, [], ResultData)
            writer.write()

    @staticmethod
    def read_csv(file_path):
        df = pd.read_csv(file_path)
        return df

    @staticmethod
    def read_csv_to_df(file_path):
        return DataFrame(ResultData.read_csv(file_path))


    def update_csv(self, file_path):
        with open(file_path, 'a') as f:
            writer = DataclassWriter(f, [self], ResultData)
            writer.write(skip_header=True)

    @staticmethod
    def create_result(params, i, run_time, setup_time, n_total_steps, time_per_step, its_per_step,
                      fails_per_step, errors_per_step,):
        return ResultData(
            **params,
            i=i,
            run_time=run_time,
            setup_time=setup_time,
            n_total_steps=n_total_steps,
            time_per_step=time_per_step,
            iterations_per_step=its_per_step,
            fails_per_step=fails_per_step,
            errors_per_step=errors_per_step,
        )


