from dataclasses import dataclass

from dataclass_csv import DataclassReader, DataclassWriter
from numpy import ndarray

CONFIG = {
    "number_of_iterations": 1,
    "title": "test",
    "result_dir": "results",
    "temp_output_dir": "temp_output",
    "data_dir": "../data",
}

PARAMETERS = {
    'n_objects': [2, 3, 4, 6, 8],
    'container': ['sphere'],
    # 'container': ["cube", 'cylinder', 'sphere'],
    'shape': [ "normal_red_blood_cell"],
    # 'shape': ["cube", 'sphere', "normal_red_blood_cell", "sickle_blood_cell", "bunny"],
    # 'padding': [0, 0.0005, 0.0001, 0.00005, 0.00001],
    'padding': [0, 0.00005, 0.00001],
    'alpha': [0.1],
    'beta': [0.5],
    # 'alpha': [0.01, 0.05, 0.1],
    # 'beta': [0.01, 0.05, 0.1],
}

RESULTS = {
    'i': int,
    'run_time': float,
    'setup_time': float,
    'scale_step_time': ndarray[float],
    'n_total_steps': int,
}

@dataclass
class ResultData:
    n_objects: int
    container: str
    shape: str
    padding: float
    alpha: float
    beta: float

    # COLLECTED_DATA
    i: int
    run_time: float
    setup_time: float
    scale_step_time: ndarray[float]
    n_total_steps: int


    def write_csv(self, file_path):
        with open(file_path, 'w') as f:
            writer = DataclassWriter(f, [self], ResultData)
            writer.write()

    def read_csv(self, file_path):
        with open(file_path, 'r') as f:
            reader = DataclassReader(f, ResultData)
            reader.write()

    def update_csv(self, file_path):
        with open(file_path, 'rw') as f:
            writer = DataclassWriter(f, [self], ResultData)
            writer.write(skip_header=True)

    @staticmethod
    def create_result(params, i, elapsed_time, setup_time, scale_step_time, n_total_steps):
        return ResultData(
            **params,
            i=i,
            run_time=elapsed_time,
            setup_time=setup_time,
            scale_step_time=scale_step_time,
            n_total_steps=n_total_steps,
        )


