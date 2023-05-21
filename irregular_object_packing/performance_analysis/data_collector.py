"""
The data collector runs all the combination of the parameters and saves the results.
This file is assumed to be called from the root of the directory. as follows:
```python3 graphproc/performance/data_collector.py```

The datacollector will run from terminal without parameters and will then:
- read all the parameters we want to test from parameters.json
- (if non existent) create results directory for specific datamodel version
- create a temporary output dir for all the compute nodes
- aggregates the values from nodes
- save it to the right data csv
- If this process is interupted by any external factor on the DAS, we can turn it on later and it will continue from where it stopped
"""

import itertools
import time
from os import path
from pathlib import Path
from time import gmtime, strftime
from typing import List

import click
import pyvista as pv
from tqdm import tqdm

from irregular_object_packing.mesh.transform import (
    scale_and_center_mesh,
    scale_to_volume,
)
from irregular_object_packing.packing.growth_based_optimisation import Optimizer
from irregular_object_packing.packing.optimizer_data import SimConfig
from irregular_object_packing.performance_analysis.search_parameters import (
    CONFIG,
    PARAMETERS,
    ResultData,
)


def get_pv_container(name):
    """Returns the container from a string"""
    match name:
        case "cube": return pv.Cube().triangulate().extract_surface()
        case "sphere": return pv.Sphere().triangulate().extract_surface()
        case "cylinder": return pv.Cylinder().triangulate().extract_surface()


def get_shape(name):
    """Returns the shape from a string"""
    match name:
        case "cube": return pv.Cube().triangulate().extract_surface()
        case "sphere": return pv.Sphere().triangulate().extract_surface()
        case "normal_red_blood_cell": return pv.read("../../data/mesh/RBC_normal.stl")
        case "sickle_red_blood_cell": return pv.read("../../data/mesh/sikleCell.stl")

# TODO: Abstract away all the read and write logic to a separate class
class DataCollector:
    start_time: str
    """Time this collector started collecting data"""
    description: str
    """Description of the data collection"""
    number_of_iterations = 3
    """number of iterations to run per parameter set"""


    #############################
    # INITIALIZATION
    #############################
    def __init__(self, number_of_iterations, description):
        self.number_of_iterations = number_of_iterations
        self.description = description
        self._set_start_time()

    def setup_directories(self):
        Path(CONFIG["result_dir"]).mkdir(parents=True, exist_ok=True)
        Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)

    def _set_start_time(self):
        """sets tje start time of the data collection to the moste recent version of the data file"""
        self.start_time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())


    #############################
    # Convenient stuff
    #############################
    @property
    def path(self):
        """Return the path to the data file"""
        return path.join(CONFIG["result_dir"], self._data_file_name())

    def _data_file_name(self):
        """Return the path to the registry file"""
        return "results_{}_{}_{}.csv".format(CONFIG["title"],self.start_time,self.description)


    def parameter_combinations(self) -> List[dict]:
        """Return the parameters used for the data collection"""
        keys, values = zip(*PARAMETERS.items(), strict=True)
        return [dict(zip(keys, v, strict=True)) for v in itertools.product(*values)]
            # key should match the parameter name in the data model

    def check_initialisation(self, scenarios):
        for scenario in scenarios:
            opt = self.setup_optimizer(scenario)
            opt.setup()
        del(opt)

    #############################
    # Data collection
    #############################

    def setup_optimizer(self,params) -> Optimizer:
        """Setup the optimizer"""

        container_volume = 10
        shape_volume = container_volume/params["n_objects"]

        container = get_pv_container(params["container"])
        container = scale_to_volume(container, 10)
        shape = get_shape(params["shape"])
        shape = scale_and_center_mesh(shape, shape_volume)


        config = SimConfig(
            padding=params["padding"],
            alpha=params["alpha"],
            beta=params["beta"],
            max_t=shape_volume**(1 / 3) * 2,
        )
        optimizer = Optimizer(shape, container, config, "performance_tests")
        return optimizer

    def collect(self,scenarios):
        """Collect the data"""
        tqdm_bar = tqdm(scenarios, desc="collect", total=len(scenarios), postfix={"i": 0})
        for scenario in tqdm_bar:
            tqdm_bar.set_postfix(i=0)
            for _i in range(CONFIG["number_of_iterations"]):

                setup_time = time.time()
                optimizer = self.setup_optimizer(scenario)
                setup_time = time.time() - setup_time

                runtime = time.time()
                optimizer.run()
                runtime = time.time() - runtime

                ResultData.create_result(
                    scenario,
                    _i,
                    runtime,
                    setup_time,
                    scale_step_time=optimizer.time_array,
                    n_total_steps=optimizer.idx,
                ).update_csv(self.path)

    def run(self):
        """Run the data collection"""
        print("Starting data collection...")
        test_scenarios = self.parameter_combinations()
        self.check_initialisation(test_scenarios)
        self.collect(test_scenarios)
        print("Data collection finished.")

@click.command()
@click.option("-i", "iterations", default=1, help="Number of iterations to run per parameter set")
@click.option("--new", "-n", "description", prompt=True, prompt_required=False, default="", help="Description of the data collection")
def main(iterations, description):
    """CLI for the data collection"""
    collector = DataCollector(iterations, description)
    collector.run()

if __name__ == "__main__":
    main()
    # idea: add one cli flag to option that specifies a new type of data collection

