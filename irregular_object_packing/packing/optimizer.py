# %%
import logging
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from time import time

import numpy as np
import pyvista as pv
from pyvista import PolyData, UnstructuredGrid
from tqdm.auto import tqdm

from irregular_object_packing.cat import chordal_axis_transform as cat
from irregular_object_packing.mesh.collision import (
    compute_all_collisions,
)
from irregular_object_packing.mesh.sampling import (
    mesh_simplification_condition,
    resample_mesh_by_triangle_area,
    resample_pyvista_mesh,
)
from irregular_object_packing.mesh.transform import (
    scale_and_center_mesh,
    scale_to_volume,
)
from irregular_object_packing.mesh.utils import (
    convert_faces_to_polydata_input,
    print_mesh_info,
)
from irregular_object_packing.packing import initialize as init
from irregular_object_packing.packing import nlc_optimisation as nlc
from irregular_object_packing.packing.optimizer_data import (
    IterationData,
    OptimizerData,
    SimConfig,
)
from irregular_object_packing.packing.optimizer_plotter import ScenePlotter
from irregular_object_packing.packing.utils import (
    check_cat_cells_quality,
    log_violations,
)


class Optimizer(OptimizerData):

    def __init__(
        self, shape: PolyData, container: PolyData, config: SimConfig, description="default"
    ):
        super().__init__()
        self.shape0 = shape
        self.shape = shape
        self.container0 = container
        self.container = container
        self.config = config
        self.description = description

        self.scale_steps = np.linspace(
            self.config.init_f,
            self.config.final_scale,
            num=self.config.n_scale_steps + 1,
        )[1:]

        self.plotter = ScenePlotter(self)
        self.log = logging.getLogger(__name__)
        self.log.setLevel(config.log_lvl)

        # intermediate results
        self.time_per_step = np.zeros(self.config.n_scale_steps)
        self.its_per_step = np.zeros(self.config.n_scale_steps)
        self.fails_per_step = np.zeros(self.config.n_scale_steps)
        self.errors_per_step = np.zeros(self.config.n_scale_steps)

    @property
    def curr_max_scale(self):
        return self.scale_steps[self.i_b]

    @ property
    def start_scale(self):
        return self.scale_steps[self.i_b - 1] if self.i_b > 0 else self.config.init_f

    # ----------------------------------------------------------------------------------------------
    # SETUP functions
    # ----------------------------------------------------------------------------------------------
    def setup(self):
        # init current state
        self.curr_sample_rate = self.shape0.n_faces
        self.tf_arrays = init.initialize_state(
            mesh=self.shape0,
            container=self.container0,
            coverage_rate=self.config.r,
            f_init=self.config.init_f,
        )

        self.log.info(f"Setup with settings: \n{self.config}")
        self.log.info(f"Number of objects: {len(self.tf_arrays)}")

        self.update_data(-1, -1)
        objects = self.current_meshes(shape=self.shape0)
        init.check_initial_state(self.container0, objects)

    def setup_pbars(self):
        self.pbar1 = tqdm(
            range(self.config.n_scale_steps), desc="scale\t", position=1, leave=True, initial=self.i_b
        )
        self.pbar2 = tqdm(range(self.config.itn_max), desc="Iteration\t", position=2, leave=True, initial=self.i)
        self.pbar3 = tqdm(range(self.n_objs), desc="Object\t", position=3, leave=True, initial=0)

    # ----------------------------------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------------------------------
    def update_data(self, i_b, i, viol_data=()):
        self.log.info(f"Updating data for {i_b=}, {i=}")
        iterdata = IterationData(
            i,
            i_b,
            self.scale_steps[i_b - 1] if i_b > 0 else self.config.init_f,
            self.scale_steps[i_b],
            np.count_nonzero(self.tf_arrays[:, 0] >= self.scale_steps[i_b]),
            self.curr_sample_rate,
            *viol_data
        )
        self.add(self.tf_arrays, self.normals, self.cat_cells, iterdata)

    def sample_rate_mesh(self, scale_factor):
        if self.config.dynamic_simplification:
            return int(mesh_simplification_condition(scale_factor, self.config.alpha, self.config.beta) * self.shape0.n_faces * self.config.upscale_factor)
        return self.shape0.n_faces  # currently simple

    def resample_meshes(self, scale_factor=None):
        self.log.info("resampling meshes")
        if scale_factor is None:
            scale_factor = self.curr_max_scale

        self.curr_sample_rate = self.sample_rate_mesh(scale_factor)
        if not self.config.sampling_disabled:
            self.shape = resample_pyvista_mesh(self.shape0, self.curr_sample_rate)
        self.container = resample_mesh_by_triangle_area(self.shape, self.container0)

        self.log.info(f"container: n_faces: {self.container.n_faces}[sampled]/{self.container0.n_faces}[original]")
        self.log.info(f"mesh: n_faces: {self.curr_sample_rate}[sampled]/{self.shape0.n_faces}[original]")

    def run(self, start_idx=None, end_idx=None, Ni=-1):
        self.check_setup()
        try:
            if self.config.n_threads == 1:
                self._run(start_idx, end_idx, Ni)
            else:
                self.executor = PoolExecutor(thread_name_prefix="optimizer", max_workers=self.config.n_threads)
                self._run(start_idx, end_idx, Ni)
                self.executor.shutdown(wait=False, cancel_futures=False)
        except KeyboardInterrupt:
            self.executor.shutdown(wait=False, cancel_futures=False)
            self.write_state()
        self.log.info("Exiting optimizer.run()")

    def _run(self, start_idx=None, end_idx=None, Ni=-1):
        self.setup_pbars()
        if start_idx is None:
            start_idx = self.i_b
        if end_idx is None:
            end_idx = self.config.n_scale_steps

        for i_b in range(start_idx, end_idx):
            self.log.info(f"Starting scaling step {i_b}")
            self.i_b = i_b
            self.resample_meshes(self.curr_max_scale)
            self.pbar1.set_postfix(ƒ_max=f"{self.curr_max_scale:.3f}")
            self.pbar2.reset()
            iteration_times = []
            for i in range(self.config.itn_max):
                self.log.info(f"Starting iteration [{i}, scale_step:{i_b}] total: {self.idx}")
                self.pbar3.reset()
                self.pbar3.set_postfix(total=self.idx)
                self.i = i
                start_time = time()
                if self.perform_optimisation_iteration() is False:
                    continue
                end_time = time()
                iteration_times.append(end_time - start_time)

                # administrative stuff
                self.process_iteration()
                self.pbar2.update()

                if Ni != -1 and self.idx >= Ni:
                    return

                if self.step_should_terminate():
                    self.time_per_step[i_b] = np.mean(iteration_times)
                    self.its_per_step[i_b] = i
                    break

            self.pbar1.update()

    # ----------------------------------------------------------------------------------------------
    # Optimisation
    # ----------------------------------------------------------------------------------------------
    def perform_optimisation_iteration(self):
        """Computes cat cells and scales the objects accordingly"""
        try:
            self.normals, self.cat_cells, self.normals_pp, _ = self.compute_cat_cells()
        except RuntimeError as e:
            self.log.error(f"RuntimeError: {e}, Scaling down and trying again...")
            self.reduce_all_scales()
            self.errors_per_step[self.i_b] += 1
            return False

        check_cat_cells_quality(self.log,self.normals)

        self.optimize_positions()
        return True

    def optimize_positions(self):
        self.log.debug("optimizing cells...")

        if self.config.n_threads is None or self.config.n_threads == 1:
            self.executor.map(self.parallel_local_optimisation, range(self.n_objs), self.tf_arrays)
        else:
            for obj_id, previous_tf_array in enumerate(self.tf_arrays):
                self.tf_arrays[obj_id] = self.local_optimisation(obj_id, previous_tf_array)

    def parallel_local_optimisation(self,obj_id, previous_tf_array):
        """workaround for setting the tf_arrays in parallel"""
        self.tf_arrays[obj_id] = self.local_optimisation(obj_id, previous_tf_array)

    def local_optimisation(self, obj_id, previous_tf_array, max_scale=None):
        max_scale = max_scale or self.curr_max_scale

        vertex_fpoint_fnormal_arr = np.array(self.normals[obj_id], dtype=np.float64)
        assert np.shape(vertex_fpoint_fnormal_arr)[1:] == (3,3)

        new_tf = nlc.compute_optimal_transform(
            previous_tf_array=previous_tf_array,
            obj_coord=self.object_coords[obj_id],
            vertex_fpoint_normal_arr=vertex_fpoint_fnormal_arr,
            max_scale=max_scale,
            scale_bound=(self.config.init_f, None),
            max_angle=self.config.max_a,
            max_t=self.config.max_t * max_scale if self.config.max_t is not None else None,
            padding=self.config.padding,
        )
        self.pbar3.update()
        return new_tf

    def compute_cat_cells(self, kwargs=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, UnstructuredGrid]:
        self.log.info("Computing CAT cells")

        self.objects = self.current_meshes()

        # Compute the CDT
        meshes = self.objects + [self.container]
        tetmesh = cat.compute_cdt(meshes, kwargs)

        steiner_points = tetmesh.points[range(tetmesh.n_points - sum([mesh.n_points for mesh in meshes]))]
        # COMPUTE CAT CELLS
        n_points_per_object = [obj.n_points for obj in self.objects] + [self.container.n_points+ len(steiner_points)]
        assert sum(n_points_per_object) == tetmesh.n_points, "Number of points in tetmesh does not match the sum of points in the objects"

        normals, cat_cells, normals_pp = cat.compute_cat_faces(
                tetmesh, n_points_per_object, self.tf_arrays[:, 4:]
            )
        return normals, cat_cells, normals_pp, tetmesh


    def step_should_terminate(self):
        """Returns true if all objects are scaled to the current max scale."""

        are_scaled = [arr[0] >= self.curr_max_scale for arr in self.tf_arrays]
        count = 0
        for i in range(self.n_objs):
            if are_scaled[i]:
                count += 1

        self.log.info(f"{count}/{self.n_objs} objects have reached the scaling barrier")
        self.log.info(f"scales: {[f'{f[0]:.2f}' for f in self.tf_arrays]}")
        if count == self.n_objs:
            return True
        return False

    # ----------------------------------------------------------------------------------------------
    # VALIDITY CHECKS
    # ----------------------------------------------------------------------------------------------
    def process_iteration(self):
        i, ib = self.i, self.i_b
        is_correct = False
        failed = False
        while is_correct is False:
            violations, violating_ids = self.compute_violations()

            is_correct = len(violating_ids) == 0
            if len(violating_ids) != 0:
                failed = True
                self.log.info("reducing scale for violating objects: " + str(violating_ids))
                for id in violating_ids:
                    self.reduce_scale(id, scale=0.93)

        if failed:
            self.fails_per_step[ib] += 1
        self.update_data(ib, i, violations)

    def compute_violations(self):
        p_meshes = self.current_meshes()
        cat_meshes = self.final_cat_meshes()
        cat_viols, con_viols, collisions = compute_all_collisions(p_meshes, cat_meshes, self.container, set_contacts=False)
        log_violations(self.log, self.idx+1, (cat_viols, con_viols, collisions))
        violating_ids = set()
        for ((obj_ida, obj_idb), _) in collisions:
            violating_ids.add(obj_ida)
            violating_ids.add(obj_idb)

        for (obj_id, _) in con_viols:
            violating_ids.add(obj_id)
        return (cat_viols,con_viols,collisions),violating_ids

    def reduce_all_scales(self, scale=0.99):
        for i in self.n_objs:
            self.reduce_scale(i, scale)

    def reduce_scale(self, id, scale=0.95):
        self.tf_arrays[id][0] *= scale

    def check_closed_cells(self):
        cat_cells = [
            PolyData(*convert_faces_to_polydata_input(self.cat_cells[obj_id]))
            for obj_id in range(self.n_objs)
        ]
        for _i, cell in enumerate(cat_cells):
            if not cell.is_manifold:
                self.log.error(
                )

def default_optimizer_config(N=5, mesh_dir ="./../../data/mesh/") -> "Optimizer":

    coverage_rate = 0.3
    mesh_volume = 0.2
    container_volume = 10
    mesh_volume = container_volume * coverage_rate  / N

    loaded_mesh = pv.read(mesh_dir + "RBC_normal.stl")
    container = pv.Sphere()

    # Scale the mesh and container to the desired volume
    container = scale_to_volume(container, container_volume)
    original_mesh = scale_and_center_mesh(loaded_mesh, mesh_volume)
    print_mesh_info(original_mesh, "original mesh")

    config = SimConfig(
        itn_max=100,
        n_scale_steps=9,
        r=coverage_rate,
        final_scale=1.0,
        log_lvl=logging.ERROR,
        init_f=0.1,
        max_t=mesh_volume**(1 / 3) * 2,
        padding=1E-4 * mesh_volume**(1 / 3),
        dynamic_simplification=True,
        alpha=0.1,
        beta=0.5,
        upscale_factor=1,
    )
    optimizer = Optimizer(original_mesh, container, config, description="cells_in_sphere")
    return optimizer

def simple_shapes_optimizer_config() -> "Optimizer":
    mesh_volume = 0.2
    container_volume = 10

    original_mesh = pv.Cube().triangulate().extract_surface()
    container = pv.Cube().triangulate().extract_surface()


    container = scale_to_volume(container, container_volume)
    container = resample_pyvista_mesh(container, 15*4*2)
    original_mesh = scale_and_center_mesh(original_mesh, mesh_volume)
    print_mesh_info(original_mesh, "original mesh")

    settings = SimConfig(
        itn_max=100,
        n_scale_steps=9,
        r=0.3,
        final_scale=1,
        log_lvl=logging.WARNING,
        sampling_disabled=True,
        init_f=0.1,
        # padding=0,
    )
    plotter = None
    optimizer = Optimizer(original_mesh, container, settings, plotter)
    return optimizer

def load_optimizer_from_state(statefile: str) -> 'Optimizer':
        state = OptimizerData.load_state(statefile)
        optimizer = Optimizer(
            shape=state.shape0,
            container=state.container0,
            config=state.settings,
        )
        optimizer.description = state.description
        optimizer.tf_arrays = state.tf_arrays
        optimizer.i = 0
        optimizer.i_b = state.iteration_data.i_b
        optimizer.curr_sample_rate = state.iteration_data.sample_rate
        return optimizer

if __name__ == "__main__":
    print("This is an example run of the optimizer.\n\
       the optimizer will run for 10 iterations and then plot the final state.")
    # optimizer = default_optimizer_config()
    # optimizer.setup()
    # optimizer.run(Ni=10)
    # optimizer.plotter.plot_step()


# %%

optimizer = default_optimizer_config()
optimizer.setup()
optimizer.run(Ni=1)
optimizer.plotter.plot_step()

#%%
# optimizer.resample_meshes()
# cat_cells = optimizer.compute_cat_cells()
# cat_meshes = [PolyData(*convert_faces_to_polydata_input(cat_cell)) for cat_cell in cat_cells[1]]

# plotter = pv.Plotter()
# for cat_mesh in cat_meshes:
#     plotter.add_mesh(cat_mesh, color="yellow", opacity=0.8)

# meshes = optimizer.current_meshes()
# for o in meshes:
#     plotter.add_mesh(o, color="red", opacity=0.8)

# plotter.add_mesh(optimizer.container, color="blue", opacity=0.4, show_edges=True)
# plotter.show()
# #%%
# obj_id = 4
# plotter = pv.Plotter()
# plotter.add_mesh(meshes[obj_id], color="red", opacity=0.8)
# plotter.add_mesh(cat_meshes[obj_id], color="yellow", opacity=1 , show_edges=True)
# plotter.show()



# %%
# %%

