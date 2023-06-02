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
#%%
import itertools
import time
from os import path
from pathlib import Path
from time import gmtime, strftime

import click
from tqdm import tqdm

from irregular_object_packing.mesh.transform import (
    scale_and_center_mesh,
    scale_to_volume,
)
from irregular_object_packing.packing.optimizer import Optimizer
from irregular_object_packing.packing.optimizer_data import SimConfig
from irregular_object_packing.performance_analysis.search_parameters import (
    CASE_PARAMETER_SEARCH,
    CASE_TRIVIAL_SHAPES,
    CONFIG,
    ResultData,
)
from irregular_object_packing.performance_analysis.utils import (
    get_pv_container,
    get_pv_shape,
)


# TODO: Abstract away all the read and write logic to a separate class
class DataCollector:
    start_time: str
    """Time this collector started collecting data"""
    description: str
    """Description of the data collection"""
    number_of_iterations = 1
    """number of iterations to run per parameter set"""
    Ni: int = -1
    test: bool = False

    #############################
    # INITIALIZATION
    #############################
    def __init__(self, number_of_iterations, description, test):
        self.number_of_iterations = number_of_iterations
        self.description = description
        self._set_start_time()
        self.test = test

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
        return f"results_{CONFIG['title']}_{self.start_time}_{self.description}.csv"


    def parameters_parameter_search(self) -> list[dict]:
        """Return the parameters used for the data collection"""
        keys, values = zip(*CASE_PARAMETER_SEARCH.items(), strict=True)
        return [dict(zip(keys, v, strict=True)) for v in itertools.product(*values)]
            # key should match the parameter name in the data model

    def parameters_trivial_shapes(self) -> list[dict]:

        CASE_TRIVIAL_SHAPES["container"], CASE_TRIVIAL_SHAPES["shape"]
        parameters = []
        container_shape_tuple = list(zip(CASE_TRIVIAL_SHAPES["container"], CASE_TRIVIAL_SHAPES["shape"], strict=True))
        for cs_tuple in container_shape_tuple:
            parameters.append({
                "container": cs_tuple[0],
                "shape": cs_tuple[1],
                "n_objects": CASE_TRIVIAL_SHAPES["n_objects"],
                "padding": CASE_TRIVIAL_SHAPES["padding"],
                "alpha": CASE_TRIVIAL_SHAPES["alpha"],
                "beta": CASE_TRIVIAL_SHAPES["beta"],
                "n_threads": CASE_TRIVIAL_SHAPES["n_threads"],
            })

        return parameters




    def check_initialisation(self, scenarios):
        for scenario in scenarios:
            opt = self.setup_optimizer(scenario)
            opt.setup()
            del(opt)

    #############################
    # Data collection
    #############################

    def setup_optimizer(self, params) -> Optimizer:
        """Setup the optimizer with the given parameters"""

        container_volume = 10
        shape_volume = container_volume/params["n_objects"]

        container = get_pv_container(params["container"])
        container = scale_to_volume(container, 10)
        shape = get_pv_shape(params["shape"])
        shape = scale_and_center_mesh(shape, shape_volume)

        config = SimConfig(
            padding=params["padding"],
            alpha=params["alpha"],
            beta=params["beta"],
            max_t=shape_volume**(1 / 3) * 2,
            n_threads=params["n_threads"],
        )
        optimizer = Optimizer(shape, container, config, "performance_tests")
        return optimizer

    def collect_irop_data(self,scenarios):
        """Collect the data"""
        tqdm_bar = tqdm(scenarios, desc="collect", total=len(scenarios), postfix={"i": 0}, position=0, leave=True)
        for scenario in tqdm_bar:
            tqdm_bar.set_postfix(i=0)
            for i in range(CONFIG["number_of_iterations"]):
                self.run_irop_optimizer_scenario(scenario, i)

    def run_irop_optimizer_scenario(self, scenario, i):
        setup_time = time.time()
        optimizer = self.setup_optimizer(scenario)
        optimizer.setup()
        setup_time = time.time() - setup_time

        run_time = time.time()

        optimizer.run(Ni=1 if self.test else -1)
        run_time = time.time() - run_time

                # Add an image of the result

        ResultData.create_result(
                    scenario,
                    i=i,
                    run_time=run_time,
                    setup_time=setup_time,
                    n_total_steps=optimizer.idx,
                    time_per_step=optimizer.time_per_step,
                    its_per_step=optimizer.its_per_step,
                    fails_per_step=optimizer.fails_per_step,
                    errors_per_step=optimizer.errors_per_step,
                ).update_csv(self.path)

    def collect_cellpack_data(self, scenarios):
        tqdm_bar = tqdm(scenarios, desc="collect", total=len(scenarios), postfix={"i": 0}, position=0, leave=True)
        for scenario in tqdm_bar:
            tqdm_bar.set_postfix(i=0)
            for i in range(CONFIG["number_of_iterations"]):
                self.run_cellpack_scenarios(scenario, i)


    def run_cellpack_scenario(self, scenario, i):
        setup_time = time.time()
        # Setup celpack (if needed)
        # --------------- Cellpack setup code here ---------------------
        # Building and Running Cellpack is not that difficult check the use guide here : https://hemocell.eu/user_guide/QuickStart.html#packcells
        # But you can only run it for cubes. For example the following command places RBCs in a 25x25x25 cube for a given hematocrit.
        # ./packCells  25 25 25 --plt_ratio 0 --hematocrit 0.3 -r 
        # So the first argument is the complex gemotries in which you can place RBCs and PLTs
        # Second you run a few benchmarks for various hematocrit types and various sizes to see the scaling capabilities of each algorithm
        # ------------------------------------------------------------
        setup_time = time.time() - setup_time

        run_time = time.time()
        # Run cellpack
        # --------------- cellpack run code here ----------------------


        # ------------------------------------------------------------
        run_time = time.time() - run_time

        ResultData.create_result(
                    scenario,
                    i=i,
                    run_time=run_time,
                    setup_time=setup_time,
                    n_total_steps=0,
                    # Any other parameters you can fill in for cellpack that are comparable to irop...,
                    implementation="cellpack",
                ).update_csv(self.path)

    def run(self):
        """Run the data collection"""

        if self.test:
            print("------TEST MODE------")
        print("Start DataCollector.")
        print(f"Results will be stored here: {self.path}")
        ResultData.write_csv(self.path)

        test_scenarios = self.parameters_parameter_search()
        print("Checking initialisation of all scenarios...")
        self.check_initialisation(test_scenarios)
        print("start collecting...")
        self.collect_irop_data(test_scenarios)
        print("Data collection finished.")

@click.command()
@click.option("-i", "iterations", default=1, help="Number of iterations to run per parameter set")
@click.option("--new", "-n", "description", prompt=True, prompt_required=False, default="", help="Description of the data collection")
@click.option("--test", "-t", "test", is_flag=True,default=False, help="Test to run")
def main(iterations, description, test):
    """CLI for the data collection"""
    collector = DataCollector(iterations, description, test)
    collector.run()

if __name__ == "__main__":
    # main()
    pass
    # idea: add one cli flag to option that specifies a new type of data collection

# %%
dc = DataCollector(1, "test", True)
dc.parameters_trivial_shapes()

# %%
