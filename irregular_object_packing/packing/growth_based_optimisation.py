# %%
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from importlib import reload
from os import mkdir
from time import sleep, time

import numpy as np
import pandas as pd
import pyvista as pv
from pyvista import PolyData
from tqdm.auto import tqdm

from irregular_object_packing.mesh.collision import (
    compute_all_collisions,
    compute_container_violations,
    compute_object_collisions,
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
from irregular_object_packing.mesh.utils import print_mesh_info
from irregular_object_packing.packing import chordal_axis_transform as cat
from irregular_object_packing.packing import initialize as init
from irregular_object_packing.packing import nlc_optimisation as nlc
from irregular_object_packing.packing import plots
from irregular_object_packing.packing.cat import CatData
from irregular_object_packing.packing.optimizer_data import (
    IterationData,
    OptimizerData,
    SimConfig,
)
from irregular_object_packing.tools.profile import cprofile

# pv.set_jupyter_backend("panel")
LOG_LVL_ERROR = 0
LOG_LVL_WARNING = 1
LOG_LVL_INFO = 2
LOG_LVL_DEBUG = 3
LOG_LVL_NO_LOG = -1
LOG_PREFIX = ["[ERROR]: ", "[WARNING]: ", "[INFO]: ", "[DEBUG]: "]


violations, viol_meshes, n = [], [], 0


class Optimizer(OptimizerData):

    def __init__(
        self, shape: PolyData, container: PolyData, config: SimConfig, plotter=None, description="default"
    ):
        super().__init__()
        self.shape0 = shape
        self.shape = shape
        self.container0 = container
        self.container = container
        self.config = config
        self.cat_data = CatData.default()
        self.tf_arrays = np.empty(0)
        self.plotter: pv.Plotter = plotter
        self.scaling_barrier_list = np.linspace(
            self.config.init_f,
            self.config.final_scale,
            num=self.config.n_scaling_steps + 1,
        )[1:]
        self.description = description
        self.executor = PoolExecutor()
        self.pbar1 = None
        self.pbar2 = None
        self.pbar3 = None

    @property
    def curr_max_scale(self):
        return self.scaling_barrier_list[self.i_b]

    @ property
    def start_scale(self):
        return self.scaling_barrier_list[self.i_b - 1] if self.i_b > 0 else self.config.init_f

    # ----------------------------------------------------------------------------------------------
    # SETUP functions
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def from_state(statefile: str) -> 'Optimizer':
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

    def setup(self):
        # init current state
        self.curr_sample_rate = self.shape0.n_faces
        object_coords, skipped = init.init_coordinates(
            container=self.container0,
            mesh=self.shape0,
            coverage_rate=self.config.r,
            f_init=self.config.init_f,
        )
        n_objects = len(object_coords)
        # self.resample_meshes(self.settings.init_f)
        self.log(
            f"Skipped {skipped} points to avoid overlap with container", LOG_LVL_DEBUG
        )
        self.log(f"Setup with settings: \n{self.config}", LOG_LVL_INFO)
        self.log(f"Number of objects: {n_objects}", LOG_LVL_INFO)

        object_rotations = np.random.uniform(-np.pi, np.pi, (n_objects, 3))

        # SET TRANSFORM DATA
        self.tf_arrays = np.empty((n_objects, 7))

        for i in range(n_objects):
            tf_arr_i = np.array(
                [self.config.init_f, *object_rotations[i], *object_coords[i]]
            )
            self.tf_arrays[i] = tf_arr_i

        self.update_data(-1, -1)
        objects = self.current_meshes(shape=self.shape0)
        overlaps = compute_object_collisions(objects)
        if len(overlaps) > 0:
            raise ValueError(
                f"Initial object placements show overlaps for {overlaps}"
            )

        overlaps = compute_container_violations(objects, self.container0)
        if len(overlaps) > 0:
            raise ValueError(
                "Initial object placements show container violations for"
                f" {overlaps} objects."
            )

        if self.plotter is not None:
            self.plotter.show(interactive=True, interactive_update=True)

    def setup_pbars(self):
        self.pbar1 = tqdm(
            range(self.config.n_scaling_steps), desc="scaling\t", position=0, leave=True, initial=self.i_b
        )
        self.pbar2 = tqdm(range(self.config.itn_max), desc="Iteration\t", position=1, leave=True, initial=self.i)
        self.pbar3 = tqdm(range(self.n_objs), desc="Object\t", position=2, leave=True, initial=0)

    # ----------------------------------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------------------------------
    def update_data(self, i_b, i, viol_data=()):
        self.log(f"Updating data for {i_b=}, {i=}")
        iterdata = IterationData(
            i,
            i_b,
            self.scaling_barrier_list[i_b - 1] if i_b > 0 else self.config.init_f,
            self.scaling_barrier_list[i_b],
            np.count_nonzero(self.tf_arrays[:, 0] >= self.scaling_barrier_list[i_b]),
            self.curr_sample_rate,
            *viol_data
        )
        self.add(self.tf_arrays, self.cat_data, iterdata)

    def log(self, msg, log_lvl=LOG_LVL_INFO):
        if log_lvl > self.config.log_lvl:
            return

        msg = LOG_PREFIX[log_lvl] + msg + f"[i={self.idx}]"
        if self.pbar1 is None:
            print(msg)
        else:
            self.pbar1.write(msg)

    def report(self):
        df = pd.DataFrame(
            data=self.tf_arrays, columns=["scale", "r_x", "ry", "rz", "t_x", "t_y", "t_z"]
        )
        return df

    def sample_rate_mesh(self, scale_factor):
        if self.config.dynamic_simplification:
            return int(mesh_simplification_condition(scale_factor, self.config.alpha, self.config.beta) * self.shape0.n_faces * self.config.upscale_factor)
        return self.config.sample_rate  # currently simple

    def resample_meshes(self, scale_factor=None):
        self.log("resampling meshes", LOG_LVL_DEBUG)
        if scale_factor is None:
            scale_factor = self.curr_max_scale

        if self.config.sample_rate is not None:
            self.curr_sample_rate = self.sample_rate_mesh(scale_factor)
            self.shape = resample_pyvista_mesh(self.shape0, self.curr_sample_rate)
            self.container = resample_mesh_by_triangle_area(self.shape, self.container0)
        assert self.shape.is_manifold
        assert self.container.is_manifold

        self.log(f"container: n_faces: {self.container.n_faces}[sampled]/{self.container0.n_faces}[original]", LOG_LVL_INFO)
        self.log(f"mesh: n_faces: {self.curr_sample_rate}[sampled]/{self.shape0.n_faces}[original]", LOG_LVL_INFO)
        # self.padding = avg_mesh_area**(0.5) / 4 # This doenst help
        # self.log(f"new_padding: {self.padding}", LOG_LVL_INFO)

    def run(self, start_idx=None, end_idx=None, Ni=-1):
        try:
            self._run(start_idx, end_idx, Ni)
        except KeyboardInterrupt:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.write_state()

    def _run(self, start_idx=None, end_idx=None, Ni=-1):
        self.setup_pbars()
        if start_idx is None:
            start_idx = self.i_b
        if end_idx is None:
            end_idx = self.config.n_scaling_steps

        for i_b in range(start_idx, end_idx):
            self.i_b = i_b
            self.resample_meshes(self.curr_max_scale)
            self.log(f"Starting scaling step {i_b}")
            self.pbar1.set_postfix(ƒ_max=f"{self.curr_max_scale:.3f}")
            self.pbar2.reset()

            for i in range(self.config.itn_max):
                self.i = i
                self.log(f"Starting iteration [{i}, scale_step:{i_b}] total: {self.idx}")
                self.pbar3.reset()
                self.pbar3.set_postfix(total=self.idx)
                if self.iteration() is False:
                    continue

                # administrative stuff
                self.process_iteration()
                self.pbar2.update()
                if self.step_should_terminate():
                    break

                if Ni != -1 and self.idx >= Ni:
                    break

            self.pbar1.update()

    # ----------------------------------------------------------------------------------------------
    # Optimisation
    # ----------------------------------------------------------------------------------------------
    def iteration(self):
        """Perform a single iteration of the optimisation."""

        # DOWN SAMPLE MESHES
        try:
            self.compute_cat_cells()
        except RuntimeError as e:
            self.log(f"RuntimeError: {e}")
            self.log("Scaling down and trying again...")
            # self.store_state(self.current_meshes() + [self.container], f"issues/{self.description}{self.idx}")
            for i in range(self.n_objs):
                self.reduce_scale(i, scale=0.99)
            return False

        # GROWTH-BASED OPTIMISATION
        self.log("optimizing cells...", LOG_LVL_INFO)
        tasks = []
        for obj_id, previous_tf_array in enumerate(self.tf_arrays):
            task = self.executor.submit(self.local_optimisation, obj_id, previous_tf_array, self.curr_max_scale)
            tasks.append(task)

        for task in tasks:
            task.result()  # Wait for the tasks to complete
        return True

    def local_optimisation(self, obj_id, previous_tf_array, max_scale):

        new_tf = nlc.compute_optimal_growth(
            obj_id=obj_id,
            previous_tf_array=previous_tf_array,
            max_scale=max_scale,
            scale_bound = (self.config.init_f, None),
            max_angle = self.config.max_a,
            max_t = self.config.max_t * max_scale if self.config.max_t is not None else None,
            padding = self.config.padding,
            cat_data = self.cat_data,
        )

        self.tf_arrays[obj_id] = new_tf
        self.pbar3.update()

    def compute_cat_cells(self, kwargs=None):
        self.log("Computing CAT cells")
        if kwargs is None:
            kwargs = {
                "steinerleft": 0,
                "minratio": 3.0,
                "switches": "D1",
                # "verbose": 2,
            }

        # TRANSFORM MESHES TO OBJECT COORDINATES, SCALE, ROTATION
        object_meshes = self.current_meshes()

        # Compute the CDT
        tetmesh = cat.compute_cdt(object_meshes + [self.container], kwargs)

        # The point sets are sets(uniques) of tuples (x,y,z) for each object, for quick lookup
        obj_point_sets = [set(map(tuple, obj.points)) for obj in object_meshes] + [
            set(map(tuple, self.container.points))
        ]

        # COMPUTE CAT CELLS
        self.cat_data = cat.compute_cat_faces(
            tetmesh, obj_point_sets, self.tf_arrays[:, 4:]
        )

    def step_should_terminate(self):
        """Returns true if all objects are scaled to the current max scale."""

        are_scaled = [arr[0] >= self.curr_max_scale for arr in self.tf_arrays]
        count = 0
        for i in range(self.n_objs):
            if are_scaled[i]:
                count += 1
                self.log(f"Object {i} has reached the scaling barrier", LOG_LVL_DEBUG)

        self.log(f"{count}/{self.n_objs} objects have reached the scaling barrier", LOG_LVL_INFO)
        self.log(f"scales: {[f'{f[0]:.2f}' for f in self.tf_arrays]}", LOG_LVL_INFO)
        if count == self.n_objs:
            return True
        return False

    # ----------------------------------------------------------------------------------------------
    # VALIDITY CHECKS
    # ----------------------------------------------------------------------------------------------
    def process_iteration(self):
        i, ib = self.i, self.i_b
        p_meshes = self.current_meshes()
        cat_meshes = self.final_cat_meshes()
        cat_viols, con_viols, collisions = compute_all_collisions(p_meshes, cat_meshes, self.container, set_contacts=False)
        self.log_violations((cat_viols, con_viols, collisions))

        violating_ids = set()
        for ((obj_ida, obj_idb), _) in collisions:
            violating_ids.add(obj_ida)
            violating_ids.add(obj_idb)

        for (obj_id, _) in con_viols:
            violating_ids.add(obj_id)

        if len(violating_ids) != 0:
            self.log("reducing scale for violating objects: " + str(violating_ids), LOG_LVL_INFO)
            for id in violating_ids:
                self.reduce_scale(id, scale=0.98)

        self.update_data(ib, i, (cat_viols, con_viols, collisions))

    def reduce_scale(self, id, scale=0.95):
        self.tf_arrays[id][0] *= scale

    def check_closed_cells(self):
        cat_cells = [
            PolyData(*cat.face_coord_to_points_and_faces(self.cat_data, obj_id))
            for obj_id in range(self.n_objs)
        ]
        for i, cell in enumerate(cat_cells):
            if not cell.is_manifold:
                self.log(
                    f"CAT cell of object {i} is not manifold", log_lvl=LOG_LVL_WARNING
                )

    def log_violations(self, violations):
        if len(violations[0]) > 0:
            self.log(f"! cat violation found {violations[0]}", LOG_LVL_WARNING,)
        if len(violations[1]) > 0:
            self.log(f"! container violation found {violations[1]}", LOG_LVL_WARNING)
        if len(violations[2]) > 0:
            self.log(f"! collisiond found {violations[2]}", LOG_LVL_WARNING)
        sleep(0.5)  # for easier spotting in the terminal

    def store_state(self, meshes, name=""):
        sum = pv.PolyData()
        for mesh in meshes:
            sum = sum + mesh

        sum.save(f"../dump/{name}error-{time():.0f}.stl", sum)

    @staticmethod
    def default_setup() -> "Optimizer":
        DATA_FOLDER = "./../../data/mesh/"

        mesh_volume = 0.5
        container_volume = 10

        loaded_mesh = pv.read(DATA_FOLDER + "RBC_normal.stl")
        container = pv.Sphere()

        # Scale the mesh and container to the desired volume
        container = scale_to_volume(container, container_volume)
        original_mesh = scale_and_center_mesh(loaded_mesh, mesh_volume)
        print_mesh_info(original_mesh, "original mesh")

        config = SimConfig(
            itn_max=100,
            n_scaling_steps=9,
            r=0.3,
            final_scale=1.0,
            log_lvl=LOG_LVL_INFO,
            init_f=0.1,
            max_t=mesh_volume**(1 / 3) * 2,
            # padding=1E-2 * mesh_volume**(1 / 3),
            dynamic_simplification=True,
            alpha=0.1,
            beta=0.1,
            upscale_factor=1,
        )
        plotter = None
        optimizer = Optimizer(original_mesh, container, config, plotter, description="cells_in_sphere")
        return optimizer

    @staticmethod
    def simple_shapes_setup() -> "Optimizer":
        mesh_volume = 0.8

        container_volume = 10

        original_mesh = pv.Cube().triangulate().extract_surface()
        container = pv.Cube().triangulate().extract_surface()

        container = scale_to_volume(container, container_volume)
        original_mesh = scale_and_center_mesh(original_mesh, mesh_volume)
        print_mesh_info(original_mesh, "original mesh")


        settings = SimConfig(
            itn_max=100,
            n_scaling_steps=5,
            r=0.3,
            final_scale=1,
            sample_rate=None,
            log_lvl=LOG_LVL_ERROR,
            init_f=0.1,
            padding=0,
        )
        plotter = None
        optimizer = Optimizer(original_mesh, container, settings, plotter)
        return optimizer


# %%

optimizer = Optimizer.default_setup()
# optimizer = Optimizer.simple_shapes_setup()
optimizer.setup()

# optimizer.run(Ni=1)
# %%
# state_file = "state-cells_in_sphere-n15_cv10.0_f0.7000000000000001.pickle"
# state_file = "state-cells_in_sphere-n15_cv10.0_f1.0-t1683810762.pickle"
# optimizer = Optimizer.from_state(state_file)
# %%
# %load_ext pyinstrument
# %%
# optimizer.run(Ni=2)
optimizer.run()


# %%

# reload(plots)
# save_path = f"../dump/cdt_fix_{time()}"
# plots.generate_gif(optimizer , save_path + ".gif")

reload(plots)


def plot_step(optimizer: Optimizer, step, meshes, cat_meshes, container):
    plotter = pv.Plotter()
    plots.plot_simulation_scene(plotter, meshes, cat_meshes, container, c_kwargs={"show_edges": False, "edge_color": "purple"})
    plotter.add_text(optimizer.status(step).table_str, position="upper_left")
    plotter.show()
    return plotter


step = 49
meshes_before, meshes_after, cat_meshes, container = optimizer.recreate_scene(step)
plot_step(optimizer, step, meshes_after, cat_meshes, container )

# %%
obj_i = 0
plotter = plots.plot_step_single(
    meshes_before[obj_i], cat_meshes[obj_i],  # container=container,
    # meshes_after[obj_i], cat_meshes[obj_i],  # container=container,
    cat_opacity=0.7, mesh_opacity=1 , clipped=True, title="Non overlapping cells",
    other_meshs=[meshes_after[2], ],
    # tetmesh=tetmesh,
    # c_kwargs={"show_edges": True, "edge_color": "purple", "show_vertices": True, "point_size": 10},
    m_kwargs={"show_edges": True, "show_vertices": True, "point_size": 10, },
    cat_kwargs={"show_edges": True, "show_vertices": True, "point_size": 5, },
    oms_kwargs=[
        {"show_edges": True, "color": "w", "edge_color": "red", "show_vertices": True, "point_size": 1, }
    ],
)
#%%
plotter = plots.plot_step_single(
    # meshes_before[obj_i], cat_meshes[obj_i],  # container=container,
    meshes_after[obj_i], cat_meshes[obj_i],  # container=container,
    cat_opacity=0.7, mesh_opacity=1 , clipped=True, title="cat overlap and collision",
    other_meshs=[meshes_after[2], ],
    # tetmesh=tetmesh,
    # c_kwargs={"show_edges": True, "edge_color": "purple", "show_vertices": True, "point_size": 10},
    m_kwargs={"show_edges": True, "show_vertices": True, "point_size": 10, },
    cat_kwargs={"show_edges": True, "show_vertices": True, "point_size": 5, },
    oms_kwargs=[
        {"show_edges": True, "color": "w", "edge_color": "red", "show_vertices": True, "point_size": 1, }
    ],
)

# %%
# # store cat mesh in file

title = "NLC_opt_too_much"
obj_ids, step = [10, 13], 89


def store_issue_files(optimizer, step, title, obj_ids):
    issue_name = f"issue{title}_{int(time())}"
    folder_dir = f"../dump/issue_reports/{issue_name}/"
    mkdir(folder_dir)

    meshes_before, meshes_after, cat_meshes, container = optimizer.recreate_scene(step)
    tetmesh, filtered_tetmesh, _ = optimizer.reconstruct_delaunay(step)
    for obj_i in obj_ids:
        cat_meshes[obj_i].save(folder_dir + f"cat[o{obj_i}i{step}].stl")
        meshes_before[obj_i].save(folder_dir + f"obj_before[o{obj_i}i{step}].stl")
        meshes_after[obj_i].save(folder_dir + f"obj_after[o{obj_i}i{step}].stl")

    tetmesh, filtered_tetmesh, _ = optimizer.reconstruct_delaunay(step)
    tetmesh.save(folder_dir + f"tetmesh[i{step}].vtk")
    filtered_tetmesh.save(folder_dir + f"filtered_tetmesh[i{step}].vtk")


store_issue_files(optimizer, step, title, obj_ids)

# %%


@cprofile
def profile_optimizer():
    optimizer.run(start_idx=optimizer.i_b, Ni=1)


profile_optimizer()
# # %%


# # %%


# fig, ax = plt.subplots()

# a = [0.05, 0.15, 0.25]
# b = [0.1, 0.2, 0.3, 0.5]

# x = np.linspace(0, 1, 100)

# for ai in a:
#     for bi in b:
#         print(f"{ai} {bi}`")
#         ax.plot(mesh_simplification_condition(x, ai, bi), label=f"a:{ai:.2f},  b:{bi:.2f}")
# ax.legend()

# # %%

# %%
