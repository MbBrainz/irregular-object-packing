"""
The data collector runs all the combination of the parameters and saves the results.
    os.chdir(get_project_root().parent)his file is assumed to be called from the root of the directory. as follows:
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
import os
import shutil
import time
from os import path
from pathlib import Path
from time import gmtime, strftime

import click
from tqdm import tqdm

from irregular_object_packing.mesh.shapes import get_pv_manifold_shape
from irregular_object_packing.mesh.transform import (
    scale_and_center_mesh,
)
from irregular_object_packing.packing.optimizer import Optimizer
from irregular_object_packing.packing.optimizer_data import SimConfig
from irregular_object_packing.performance_analysis.search_parameters import (
    CASE_BLOODCELL_MAX,
    CASE_PARAMETER_SEARCH_ALPHA_BETA,
    CASE_PARAMETER_SEARCH_PADDING,
    CASE_TEST,
    CASE_TRIVIAL_SHAPES,
    CASES,
    CONFIG,
    ResultData,
)
from irregular_object_packing.performance_analysis.utils import (
    HiddenPrints,
)


# TODO: Abstract away all the read and write logic to a separate class
class DataCollector:
    start_time: str
    """Time this collector started collecting data"""
    description: str
    """Description of the data collection"""
    data_dir: str
    output_dir: str
    number_of_iterations = 1
    """number of iterations to run per parameter set"""
    Ni: int = -1
    test: bool = False
    cellpack: bool = False

    #############################
    # INITIALIZATION
    #############################
    def __init__(self, number_of_iterations, description, data_dir, output_dir, test):
        self.number_of_iterations = number_of_iterations
        self.description = description
        self._set_start_time()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.test = test
        self.setup_directories()

    def setup_directories(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(CONFIG["data_dir"]).mkdir(parents=True, exist_ok=True)
        # Path(f"{self.output_dir}/{self._file_name}").mkdir(parents=True, exist_ok=True)

    def _set_start_time(self):
        """sets tje start time of the data collection to the moste recent version of the data file"""
        self.start_time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())


    #############################
    # Convenient stuff
    #############################
    @property
    def csv_path(self):
        """Return the path to the data file"""
        return path.join(self.output_dir, self._file_name + ".csv")

    @property
    def results_path(self):
        """Return the path to the registry file"""
        return path.join(self.output_dir, self._file_name)

    def img_path(self, i):
        """Return the path to the image file"""
        return path.join(self.output_dir, self._file_name, self._img_file_name(i))

    @property
    def _file_name(self):
        """Return the path to the registry file"""
        return f"results_{CONFIG['title']}_{self.start_time}_{self.description}"

    def _img_file_name(self, i):
        """Return the path to the registry file"""
        return f"final_{i}.png"


    def parameters_alpha_beta(self) -> list[dict]:
        """Return the parameters used for the data collection"""
        case = CASE_PARAMETER_SEARCH_ALPHA_BETA if self.test is not True else CASE_TEST
        keys, values = zip(*case.items(), strict=True)
        return [dict(zip(keys, v, strict=True)) for v in itertools.product(*values)]
            # key should match the parameter name in the data model

    def parameters_padding(self) -> list[dict]:
        """Return the parameters used for the data collection"""
        case = CASE_PARAMETER_SEARCH_PADDING if self.test is not True else CASE_TEST
        keys, values = zip(*case.items(), strict=True)
        return [dict(zip(keys, v, strict=True)) for v in itertools.product(*values)]
            # key should match the parameter name in the data model

    def parameters_trivial_shapes(self) -> list[dict]:

        CASE_TRIVIAL_SHAPES["container"], CASE_TRIVIAL_SHAPES["shape"]
        parameters = []
        # create a list of tuples of container and shape with corresponding shapes
        container_shape_tuple = list(zip(CASE_TRIVIAL_SHAPES["container"], CASE_TRIVIAL_SHAPES["shape"], strict=True))
        for cs_tuple in container_shape_tuple:
            parameters.append({
                "container": cs_tuple[0],
                "shape": cs_tuple[1],
                "coverage_rate": CASE_TRIVIAL_SHAPES["coverage_rate"],
                "n_objects": CASE_TRIVIAL_SHAPES["n_objects"],
                "padding": CASE_TRIVIAL_SHAPES["padding"],
                "alpha": CASE_TRIVIAL_SHAPES["alpha"],
                "beta": CASE_TRIVIAL_SHAPES["beta"],
                "n_threads": CASE_TRIVIAL_SHAPES["n_threads"],
            })

        return parameters

    def parameters_bloodcell_max(self) -> list[dict]:
        parameters = []
        for container in CASE_BLOODCELL_MAX["container"]:
            for shape in  CASE_BLOODCELL_MAX["shape"]:
                for n_objects in CASE_BLOODCELL_MAX['n_objects']:
                    parameters.append({
                        "container": container,
                        "shape": shape,
                        "coverage_rate": CASE_BLOODCELL_MAX["coverage_rate"],
                        "n_objects": n_objects,
                        "padding": CASE_BLOODCELL_MAX["padding"],
                        "alpha": CASE_BLOODCELL_MAX["alpha"],
                        "beta": CASE_BLOODCELL_MAX["beta"],
                        "n_threads": CASE_BLOODCELL_MAX["n_threads"],
                    })
        return parameters


    def check_initialisation(self, scenarios):
        print(f"Checking {len(scenarios)} scenarios..")
        for scenario in tqdm(scenarios):
            opt = self.setup_optimizer(scenario)
            opt.setup()
            del(opt)

    #############################
    # Data collection
    #############################

    def setup_optimizer(self, params) -> Optimizer:
        """Setup the optimizer with the given parameters"""


        shape_volume = 97 # see https://github.com/UvaCsl/HemoCell/blob/d91698721f7a06166cb6ac8d22fa2f2f4baa4ed7/tools/packCells/packCells.cpp#L203
        container_volume = shape_volume * params["n_objects"] / params["coverage_rate"]

        container = get_pv_manifold_shape(params["container"], self.data_dir)
        container = scale_and_center_mesh(container, container_volume)

        shape = get_pv_manifold_shape(params["shape"], self.data_dir)
        shape = scale_and_center_mesh(shape, shape_volume)

        config = SimConfig(
            padding=params["padding"],
            alpha=params["alpha"],
            r=params["coverage_rate"],
            beta=params["beta"],
            max_t=shape_volume**(1 / 3) * 2,
            n_threads=params["n_threads"],
            itn_max=300,
        )
        optimizer = Optimizer(shape, container, config, "performance_tests")
        return optimizer

    def collect_irop_data(self,scenarios):
        """Collect the data"""
        print(f"Collecting {len(scenarios)} scenarios..")
        tqdm_bar = tqdm(scenarios, desc="collect", total=len(scenarios), postfix={"i": 0}, position=0)
        for s, scenario in enumerate(tqdm_bar):
            tqdm_bar.write(f"Collecting data for {scenario} ({s+1}/{len(scenarios)})")
            tqdm_bar.set_postfix(i=0)
            for _i in range(self.number_of_iterations):
                self.run_irop_optimizer_scenario(scenario, s)

    def run_irop_optimizer_scenario(self, scenario, i):
        setup_time = time.time()
        optimizer = self.setup_optimizer(scenario)

        # re route the text output to a temporary file. I dont want to see it
        with HiddenPrints():
            optimizer.setup()
            setup_time = time.time() - setup_time

            run_time = time.time()

            optimizer.run(Ni=1 if self.test else -1)
            run_time = time.time() - run_time

            # optimizer.plotter.plot_step(save_path=self.img_path(i))

        ResultData.create_result(
                    scenario,
                    i=i,
                    run_time=run_time,
                    setup_time=setup_time,
                    n_total_steps=optimizer.idx,
                    object_scales=optimizer.object_scales,
                    time_per_step=optimizer.time_per_step,
                    its_per_step=optimizer.its_per_step,
                    fails_per_step=optimizer.fails_per_step,
                    errors_per_step=optimizer.errors_per_step,
                ).update_csv(self.csv_path)

    def collect_cellpack_data(self, scenarios):
        tqdm_bar = tqdm(scenarios, desc="collect", total=len(scenarios), postfix={"i": 0}, position=0, leave=True)
        for scenario in tqdm_bar:
            tqdm_bar.set_postfix(i=0)
            for i in range(self.number_of_iterations):
                self.run_cellpack_scenario(scenario, i)


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

        # compute the corresponding size of the container per nobjects and hematocrit
        rbc_volume = 97
        container_volume = rbc_volume * scenario["n_objects"] / scenario["coverage_rate"]
        sX = container_volume**(1 / 3)
        sY, sZ = sX, sX
        setup_time = time.time() - setup_time


        run_time = time.time()
        # Run cellpack
        # --------------- cellpack run code here ----------------------
        output_lines = os.popen(f"packCells  {sX} {sY} {sZ} --plt_ratio 0 --hematocrit {scenario['coverage_rate']}").read().split("\n")

        with open("RBC.pos", "r") as f:
            output_lines = f.readlines()
        # ------------------------------------------------------------
        run_time = time.time() - run_time

        resultdata = ResultData(
                    **scenario,
                    i=i,
                    run_time=run_time,
                    setup_time=setup_time,
                    object_scales=[1.0] * int(output_lines[0]),
                    n_total_steps=0,
                    implementation="cellpack")
        resultdata.update_csv(self.csv_path)
                # Any other parameters you can fill in for cellpack that are comparable to irop...,

    def run(self, scenarios):
        """Run the data collection"""

        if self.test:
            print("------TEST MODE------")
        print("Start DataCollector.")
        print(f"Results will be stored here: {self.csv_path}")
        ResultData.write_csv(self.csv_path)

        if self.cellpack is False:
            self.check_initialisation(scenarios)
            self.collect_irop_data(scenarios)
        else:
            self.collect_cellpack_data(scenarios)

@click.command()
@click.option("--case", "-c", "case", type=click.Choice(CASES), help="Case to run", required=True)
@click.option("-N", "iterations", default=1, help="Number of iterations to run per parameter set")
@click.option("--description", "-d", "description", prompt=True, prompt_required=False, default="", help="Description of the data collection")
@click.option("--input-dir", "-i", "input_dir", default="./data", help="Directory where the input data like mesh files is stored")
@click.option("--output-dir", "-o", "output_dir", default="./results", help="Output directory for the data")
@click.option("--skip-check", "-s", "skip_check", is_flag=True, default=False, help="Skip the initialisation check")
@click.option("--test", "-t", "test", is_flag=True,default=False, help="Test to run")
def main(case, iterations, description, input_dir, output_dir ,skip_check, test):
    """CLI for the data collection"""
    collector = DataCollector(iterations, description, input_dir, output_dir, test=test)

    if case == "alpha_beta":
        scenarios = collector.parameters_alpha_beta()
    elif case == "padding":
        scenarios = collector.parameters_padding()
    elif case == "trivial_shapes":
        scenarios = collector.parameters_trivial_shapes()
    elif case == "bloodcell_max_irop":
        scenarios = collector.parameters_bloodcell_max()
    elif case == "bloodcell_max_cellpack":
        scenarios = collector.parameters_bloodcell_max()
        collector.cellpack = True
    else:
        raise ValueError(f"Case {case} not found")


    if test:
        print("------TEST MODE------")
    print("Start DataCollector.")
    print(f"Results will be stored here: {collector.csv_path}")
    ResultData.write_csv(collector.csv_path)

    if collector.cellpack is False:
        if  not skip_check:
            collector.check_initialisation(scenarios)

        collector.collect_irop_data(scenarios)
    else:
        # Check if cellpack is installed using os.system("packCells -h") and return code
        res = shutil.which("packCells") # Check if packCells is in path
        if res is None:
            raise ValueError("Cellpack is not installed. Please install it and add it to your path")

        collector.collect_cellpack_data(scenarios)

if __name__ == "__main__":
    main()
    pass

# %%
