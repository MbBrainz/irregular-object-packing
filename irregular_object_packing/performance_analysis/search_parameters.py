from dataclasses import dataclass

import pandas as pd
from dataclass_csv import DataclassWriter
from numpy import array, ndarray
from pandas import DataFrame

CONFIG = {
    "title": "test",
    "temp_output_dir": "temp_output",
    "data_dir": "data",
}

CASE_PARAMETER_SEARCH_ALPHA_BETA = {
    'n_objects': [8],
    'coverage_rate': [0.3],
    'container': ['sphere'],
    'shape': [ "rbc_normal"],
    # 'shape': ["cube", 'sphere', "rbc_normal", "sickle_blood_cell", "bunny"],
    # 'padding': [0, 0.0005, 0.0001, 0.00005, 0.00001],
    'padding': [0],
    'alpha': [0.02, 0.04, 0.06, 0.08, 0.1],
    'beta': [0.1, 0.2, 0.3, 0.4, 0.5],
    'n_threads': [32],
}

CASE_PARAMETER_SEARCH_PADDING = {
    'n_objects': [8],
    'coverage_rate': [0.3],
    'container': ['sphere'],
    'shape': [ "rbc_normal"],
    # 'shape': ["cube", 'sphere', "rbc_normal", "sickle_blood_cell", "bunny"],
    # 'padding': [0, 0.0005, 0.0001, 0.00005, 0.00001],
    'padding': [0, 0.0000001, 0.000001, 0.00001],
    'alpha': [0.06],
    'beta': [0.1],
    'n_threads': [128],
}

CASE_TEST = {
    'n_objects': [4],
    'coverage_rate': [0.3],
    'container': ['cube'],
    'shape': ["cube"],
    'padding': [0],
    'alpha': [0.1],
    'beta': [0.5],
    'n_threads': [128],
}


# could do a extra field per parameter that defines if you want combinations or iterations
CASE_TRIVIAL_SHAPES = {
    'n_objects': 1,
    'coverage_rate': 0.8,
    'container': ['cube','sphere', 'tetrahedron', 'cone'],
    'shape': ["cube", 'sphere', 'tetrahedron', 'cone'],
    'padding': 0,
    'alpha': 0.1,
    'beta': 0.5,
    'n_threads': 8,
}

CASE_BLOODCELL_MAX = {
    'n_objects': [4, 8, 16, 32, 64, 128],
    'coverage_rate': 0.3,
    'container': ['cube'],
    'shape': ["rbc_normal"],
    'padding': 0,
    'alpha': 0.1,
    'beta': 0.2,
    'n_threads': 128,
}

CASE_BLOODCELL_MAX_CICKLE = {
    'n_objects': [4, 8, 16, 32, 64, 128, 256, 512],
    'coverage_rate': 0.3,
    'container': ['cylinder', 'cube'],
    'shape': [ "sickle_red_blood_cell"],
    'padding': 0,
    'alpha': 0.1,
    'beta': 0.5,
    'n_threads': 128,
}

CASES = ["alpha_beta","padding", "trivial_shapes", "bloodcell_max_irop", "bloodcell_max_cellpack"]

RESULTS = {
    'i': int,
    'run_time': float,
    'setup_time': float,
    'n_total_steps': int,
    'object_scales': ndarray[float],
    'time_per_step': ndarray[int],
    'its_per_step': ndarray[float],
    'fails_per_step': ndarray[int],
    'errors_per_step': ndarray[float],
}

@dataclass
class ResultData:
    # PARAMETERS
    n_objects: int
    coverage_rate: float
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
    object_scales: ndarray[float] = array([])
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
    def create_result(params, i, run_time, setup_time, n_total_steps, object_scales, time_per_step, its_per_step,
                      fails_per_step, errors_per_step,) -> 'ResultData':
        return ResultData(
            **params,
            i=i,
            run_time=run_time,
            setup_time=setup_time,
            n_total_steps=n_total_steps,
            object_scales=object_scales,
            time_per_step=time_per_step,
            iterations_per_step=its_per_step,
            fails_per_step=fails_per_step,
            errors_per_step=errors_per_step,
        )
