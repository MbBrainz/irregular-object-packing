# %%
from dataclasses import dataclass
from importlib import reload
from itertools import combinations
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from pyvista import PolyData
from scipy.optimize import minimize
from tqdm.auto import tqdm

import irregular_object_packing.packing.plots as plots
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
from irregular_object_packing.packing.optimizer_data import IterationData, OptimizerData
from irregular_object_packing.packing.utils import get_max_bounds
from irregular_object_packing.tools.profile import pprofile

# pv.set_jupyter_backend("panel")
LOG_LVL_ERROR = 0
LOG_LVL_WARNING = 1
LOG_LVL_INFO = 2
LOG_LVL_DEBUG = 3
LOG_LVL_NO_LOG = -1
LOG_PREFIX = ["[ERROR]: ", "[WARNING]: ", "[INFO]: ", "[DEBUG]: "]


violations, viol_meshes, n = [], [], 0


def compute_collisions(p_meshes: list[PolyData]):
    i, colls, coll_meshes = 0, [], []

    for mesh_1, mesh_2 in combinations(p_meshes, 2):
        col, n_contacts = mesh_1.collision(mesh_2, 1)
        if col:
            coll_meshes.append(col)
        if n_contacts > 0:
            i += 1
            colls.append(n_contacts)
    return i, colls, coll_meshes


def compute_boundary_violation(mesh: PolyData, container: PolyData, mode=1):
    return mesh.collision(container, mode, cell_tolerance=1e-6)


def compute_container_violations(p_meshes, container):
    n, violations, viol_meshes = 0, [], []

    for i, mesh in enumerate(p_meshes):
        violation = compute_boundary_violation(mesh, container)
        if violation[1] > 0:
            violations.append(i)
            viol_meshes.append(violation[0])
            n += 1
    return n, violations, viol_meshes


def compute_cat_violations(p_meshes, cat_meshes):
    n, violations, viol_meshes = 0, [], []

    for i, (mesh, cat_mesh) in enumerate(zip(p_meshes, cat_meshes, strict=True)):
        violation = compute_boundary_violation(mesh, cat_mesh)
        if violation[1] > 0:
            violations.append(i)
            viol_meshes.append(violation[0])
            n += 1
    return n, violations, viol_meshes


def optimal_local_transform(
    obj_id,
    cat_data,
    scale_bound=(0.1, None),
    max_angle=1 / 12 * np.pi,
    max_t=None,
    padding=0.0,
):
    """Computes the optimal local transform for a given object id.

    This will return the transformation parameters that maximises scale with
    respect to a local coordinate system of the object. This is possible due to
    the `obj_coords`.
    """

    r_bound = (-max_angle, max_angle)
    t_bound = (-max_t, max_t)
    bounds = [scale_bound, r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]
    x0 = np.array([scale_bound[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    constraint_dict = {
        "type": "ineq",
        "fun": nlc.local_constraints_from_cat,
        "args": (
            obj_id,
            cat_data,
            padding,
        ),
    }
    res = minimize(
        nlc.objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict
    )
    return res.x


# SimSettings as dataclass:
@dataclass
class SimSettings:
    sample_rate: int = 50
    """The sample rate of the object surface mesh."""
    max_a: float = 1 / 12 * np.pi
    """The maximum rotation angle per growth step."""
    max_t: float = None
    """The maximum translation per growth step."""
    init_f: float = 0.1
    """Final scale."""
    final_scale: float = 1.0
    """The initial scale factor."""
    itn_max: int = 1
    """The maximum number of iterations per scaling step."""
    n_scaling_steps: int = 1
    """The number of scaling steps."""
    r: float = 0.3
    """The coverage rate."""
    plot_intermediate: bool = False
    """Whether to plot intermediate results."""
    log_lvl: int = -1
    """The log level maximum level is 3."""
    decimate: bool = True
    """Whether to decimate the object mesh."""
    sample_rate_ratio: float = 1.0
    """The ratio between the sample rate of the object and the container."""
    padding: float = 0.0
    """The padding which is added to the inside of the cat cells."""
    dynamic_simplification: bool = False
    """Whether to use dynamic simplification."""
    alpha: float = 0.05
    beta: float = 0.1


class Optimizer(OptimizerData):
    settings: SimSettings
    tf_arrs: np.ndarray[np.ndarray]
    object_coords: np.ndarray
    prev_tf_arrs: np.ndarray[np.ndarray]

    def __init__(
        self, shape: PolyData, container: PolyData, settings: SimSettings, plotter=None
    ):
        self.shape0 = shape
        self.shape = shape
        self.container0 = container
        self.container = container
        self.settings = settings
        self.cat_data = None
        self.tf_arrs = np.empty(0)
        self.object_coords = np.empty(0)
        self.prev_tf_arrs = np.empty(0)
        self.plotter: pv.Plotter = plotter
        self.objects = None
        self.pbar1 = self.pbar2 = None
        self.padding = 0.0
        self.scaling_barrier_list = np.linspace(
            self.settings.init_f,
            self.settings.final_scale,
            num=self.settings.n_scaling_steps + 1,
        )[1:]

    # ----------------------------------------------------------------------------------------------
    # SETUP functions
    # ----------------------------------------------------------------------------------------------
    def setup(self):
        self.curr_sample_rate = self.shape0.n_faces
        self.object_coords, skipped = init.init_coordinates(
            container=self.container,
            mesh=self.shape,
            coverage_rate=self.settings.r,
            f_init=self.settings.init_f,
        )
        # self.resample_meshes(self.settings.init_f)
        self.log(
            f"Skipped {skipped} points to avoid overlap with container", LOG_LVL_DEBUG
        )
        self.log(f"Number of objects: {self.n_objs}", LOG_LVL_INFO)

        object_rotations = np.random.uniform(-np.pi, np.pi, (self.n_objs, 3))
        self.padding = 0.01

        # SET TRANSFORM DATA
        self.tf_arrs = np.empty((self.n_objs, 7))
        self.prev_tf_arrs = np.empty((self.n_objs, 7))

        for i in range(self.n_objs):
            tf_arr_i = np.array(
                [self.settings.init_f, *object_rotations[i], *self.object_coords[i]]
            )
            self.tf_arrs[i] = tf_arr_i

        self.update_data(-1, -1)
        has_overlap = self.has_object_overlap()
        if has_overlap:
            raise ValueError(
                f"Initial object placements show overlaps for {has_overlap}"
            )

        has_c_violations = self.has_container_violations()
        if has_c_violations:
            raise ValueError(
                "Initial object placements show container violations for"
                f" {len(has_c_violations)} objects."
            )

        if self.plotter is not None:
            self.plotter.show(interactive=True, interactive_update=True)

    def setup_pbars(self):
        self.pbar1 = tqdm(
            range(self.settings.n_scaling_steps), desc="scaling \t", position=0
        )
        self.pbar2 = tqdm(range(self.settings.itn_max), desc="Iteration\t", position=1)
        self.pbar3 = tqdm(range(self.n_objs), desc="Object\t", position=2)

    # ----------------------------------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------------------------------
    def update_data(self, i_b, i):
        self.log(f"Updating data for {i_b=}, {i=}")
        iterdata = IterationData(
            i,
            i_b,
            self.scaling_barrier_list[i_b - 1] if i_b > 0 else self.settings.init_f,
            self.scaling_barrier_list[i_b],
            np.count_nonzero(self.tf_arrs[:, 0] > self.scaling_barrier_list[i_b]),
            self.curr_sample_rate,
        )
        self.add(self.tf_arrs, self.cat_data, iterdata)

    def log(self, msg, log_lvl=LOG_LVL_INFO):
        if log_lvl > self.settings.log_lvl:
            return

        msg = LOG_PREFIX[log_lvl] + msg + f"[i={self.idx}]"
        if self.pbar1 is None:
            print(msg)
        else:
            self.pbar1.write(msg)

    def report(self):
        df = pd.DataFrame(
            data=self.tf_arrs, columns=["scale", "r_x", "ry", "rz", "t_x", "t_y", "t_z"]
        )
        return df

    def sample_rate_mesh(self, scale_factor):
        if self.settings.dynamic_simplification:
            return mesh_simplification_condition(scale_factor, self.settings.alpha, self.settings.beta) * self.shape0.n_faces
        return self.settings.sample_rate  # currently simple

    # def container_sample_rate(self, scale_factor):
    #     return self.sample_rate_mesh(scale_factor) * self.n_objs

    def resample_meshes(self, scale_factor):
        self.log("resampling meshes", LOG_LVL_DEBUG)

        # self.container = resample_pyvista_mesh(
        #     self.container0, container_sample_rate
        # )
        self.curr_sample_rate = self.sample_rate_mesh(scale_factor)
        self.shape = resample_pyvista_mesh(self.shape0, self.curr_sample_rate)
        self.container = resample_mesh_by_triangle_area(self.shape, self.container0)
        self.log(f"container: n_faces: {self.container.n_faces}[sampled]/{self.container0.n_faces}[original]", LOG_LVL_INFO)
        self.log(f"mesh: n_faces: {self.curr_sample_rate}[sampled]/{self.shape0.n_faces}[original]", LOG_LVL_INFO)

    @property
    def n_objs(self):
        return len(self.object_coords)

    def add_meshes_to_plot(self, coll_meshes, obj_id=None):
        if self.plotter is None:
            return

        for mesh in coll_meshes:
            self.plotter.add_mesh(mesh)

        self.plotter.render()

    def add_mesh_to_plot(self, obj_id):
        pass

    def update_plot(self):
        if self.plotter is None:
            return

        self.plotter.clear()
        self.plotter.add_mesh(self.container, color="white", opacity=0.2)
        colors = plots.generate_tinted_colors(self.n_objs)
        for i, mesh in enumerate(self.final_meshes_after(self.shape)):
            self.plotter.add_mesh(mesh, color=colors[1][i], opacity=0.7)

        cat_meshes = [
            PolyData(*cat.face_coord_to_points_and_faces(self.cat_data, obj_id))
            for obj_id in range(self.n_objs)
        ]

        for i, mesh in enumerate(cat_meshes):
            self.plotter.add_mesh(mesh, color=colors[0][i], opacity=0.5)

        self.plotter.update()
        sleep(5)

    def run(self):
        # self.setup()
        self.setup_pbars()

        for i_b in range(0, self.settings.n_scaling_steps):
            self.resample_meshes(self.scaling_barrier_list[i_b])
            self.log(f"Starting scaling step {i_b}")
            self.pbar1.set_postfix(ƒ_max=f"{self.scaling_barrier_list[i_b]:.2f}")
            self.pbar2.reset()

            for i in range(self.settings.itn_max):
                self.log(f"Starting iteration [{i}, scale_step:{i_b}] total: {self.idx}")
                self.pbar3.reset()
                self.pbar3.set_postfix(total=self.idx)
                self.iteration(self.scaling_barrier_list[i_b])

                # administrative stuff
                self.update_data(i_b, i)
                self.check_validity()
                self.pbar2.update()
                if self.terminate_step(i_b):
                    break

            self.pbar1.update()

    # ----------------------------------------------------------------------------------------------
    # Optimisation
    # ----------------------------------------------------------------------------------------------
    def iteration(self, scale_bound):
        """Perform a single iteration of the optimisation."""
        # DOWN SAMPLE MESHES
        self.compute_cat_cells()
        self.update_plot()

        # GROWTH-BASED OPTIMISATION
        for obj_id, previous_tf_array in enumerate(self.tf_arrs):
            self.pbar2.set_postfix(obj_id=obj_id)
            self.local_optimisation(obj_id, previous_tf_array, scale_bound)

        self.update_plot()

    def local_optimisation(self, obj_id, previous_tf_array, max_scale):
        tf_arr = optimal_local_transform(
            obj_id,
            self.cat_data,
            scale_bound=(self.settings.init_f, None),
            max_angle=self.settings.max_a,
            # limit the maximum translation to the radius of the scaled object
            max_t=self.settings.max_t * max_scale,
            padding=self.padding,
        )

        new_tf = previous_tf_array + tf_arr
        new_scale = previous_tf_array[0] * tf_arr[0]
        if new_scale > max_scale:
            new_scale = max_scale

        new_tf[0] = new_scale
        self.tf_arrs[obj_id] = new_tf
        self.object_coords[obj_id] = new_tf[4:]
        self.pbar3.update()

    def compute_cat_cells(self):
        self.log("Computing CAT cells")

        # TRANSFORM MESHES TO OBJECT COORDINATES, SCALE, ROTATION
        obj_points = [
            trimesh.transform_points(
                self.shape.points.copy(), nlc.construct_transform_matrix(transform_data)
            )
            for transform_data in self.tf_arrs
        ]

        # COMPUTE CAT CELLS
        self.cat_data = cat.compute_cat_cells(
            obj_points, self.container.points, self.object_coords
        )

    def terminate_step(self, i_b):
        are_scaled = [arr[0] >= self.scaling_barrier_list[i_b] for arr in self.tf_arrs]
        count = 0
        for i in range(self.n_objs):
            if are_scaled[i]:
                count += 1
                self.log(f"Object {i} has reached the scaling barrier", LOG_LVL_DEBUG)

        self.log(f"{count}/{self.n_objs} objects have reached the scaling barrier", LOG_LVL_INFO)
        self.log("self.report()", LOG_LVL_INFO)
        if count == self.n_objs:
            return True
        return False

    # ----------------------------------------------------------------------------------------------
    # VALIDITY CHECKS
    # ----------------------------------------------------------------------------------------------

    def check_validity(self):
        self.check_cat_boundaries()
        self.has_container_violations()
        self.has_object_overlap()

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

    def has_object_overlap(self):
        self.log("checking for collisions", LOG_LVL_DEBUG)
        p_meshes = self.final_meshes_after(self.shape)

        extents = np.array(
            [(get_max_bounds(mesh.bounds), mesh.volume) for mesh in p_meshes]
        )
        self.log(f"all mesh bounds, volumes: {extents}", LOG_LVL_DEBUG)
        n, colls, coll_meshes = compute_collisions(p_meshes)

        if n > 0:
            self.log(
                f"! collision found for {n} objects with total of {colls} contacts",
                LOG_LVL_ERROR,
            )
            return self.add_meshes_to_plot(coll_meshes)

    def check_cat_boundaries(self):
        self.log("checking for cat boundary violations", LOG_LVL_DEBUG)
        p_meshes = self.final_meshes_after(self.shape)
        cat_meshes = self.final_cat_meshes()
        n, violations, meshes = compute_cat_violations(p_meshes, cat_meshes)

        if n > 0:
            self.log(
                f"! cat boundary violation for objects {violations}", LOG_LVL_WARNING
            )
            self.add_meshes_to_plot(meshes)

    def has_container_violations(self):
        self.log("checking for container violations", LOG_LVL_DEBUG)
        p_meshes = self.final_meshes_after(self.shape)
        n, violations, meshes = compute_container_violations(p_meshes, self.container)

        if n > 0:
            self.log(f"! container violation for objects {violations}", LOG_LVL_ERROR)
            self.add_meshes_to_plot(meshes)
            return meshes

    @staticmethod
    def default_setup() -> "Optimizer":
        DATA_FOLDER = "./../../data/mesh/"

        mesh_volume = 1.3
        container_volume = 10

        loaded_mesh = pv.read(DATA_FOLDER + "RBC_normal.stl")
        container = pv.Sphere()

        # Scale the mesh and container to the desired volume
        container = scale_to_volume(container, container_volume)
        original_mesh = scale_and_center_mesh(loaded_mesh, mesh_volume)
        print_mesh_info(original_mesh, "original mesh")

        settings = SimSettings(
            itn_max=100,
            n_scaling_steps=10,
            r=0.3,
            final_scale=1.0,
            sample_rate=1000,
            log_lvl=LOG_LVL_WARNING,
            init_f=0.1,
            max_t=0.4**(1 / 3),
            padding=1e-3,
            sample_rate_ratio=2,

            # dynamic_simplification=True,
        )
        plotter = None
        optimizer = Optimizer(original_mesh, container, settings, plotter)
        return optimizer

    @staticmethod
    def simple_shapes_setup() -> "Optimizer":
        mesh_volume = 2

        container_volume = 10

        original_mesh = pv.Cube().extract_surface()
        container = pv.Cube().extract_surface()

        container = scale_to_volume(container, container_volume)
        original_mesh = scale_and_center_mesh(original_mesh, mesh_volume)

        settings = SimSettings(
            itn_max=1,
            n_scaling_steps=1,
            r=0.3,
            final_scale=0.5,
            sample_rate=600,
            log_lvl=LOG_LVL_ERROR,
            init_f=0.1,
            sample_rate_ratio=1,
            padding=1e-5,
        )
        plotter = None
        optimizer = Optimizer(original_mesh, container, settings, plotter)
        return optimizer


# # %%
# for _i in range(1000):
#     optimizer = Optimizer.simple_shapes_setup()
#     optimizer.setup()s
# %%

# # %%
# optimizer = Optimizer.simple_shapes_setup()
optimizer = Optimizer.default_setup()
optimizer.setup()
# %%
optimizer.run()

# %%

reload(plots)
save_path = f"dump/upscaling_{time()}.gif"
plots.generate_gif(optimizer, save_path)

# %%


def plot_step(optimizer, step):
    plotter = pv.Plotter()
    for cat_cell in optimizer.cat_meshes(step):
        plotter.add_mesh(cat_cell, opacity=0.6, color="yellow")

    # for obj in optimizer.meshes_after(step):
    #     plotter.add_mesh(obj, opacity=0.8, color="red")
    plotter.add_mesh(optimizer.container, opacity=0.4)
    plotter.add_text(optimizer.status(step).table_str, position="upper_left")
    tet, isotet, _ = optimizer.reconstruct_delaunay(step)
    plotter.add_mesh(tet, color="gray", opacity=0.2)
    plotter.add_mesh(isotet, color="blue", opacity=0.2, show_edges=True)

    plotter.show()


plot_step(optimizer, 15)
# iteration = 3
# # enumerate
# plotter = plots.plot_full_comparison(
#     optimizer.meshes_before(iteration, optimizer.shape),
#     # optimizer.final_meshes_before(optimizer.shape),
#     optimizer.meshes_after(iteration, optimizer.shape),
#     # optimizer.final_cat_meshes(),
#     optimizer.cat_meshes(iteration),
#     optimizer.container,
# )
# plotter.show()
# %%


# %%
obj_i, step = 1, 37
plots.plot_step_comparison(
    optimizer.mesh_before(step, obj_i, optimizer.shape),
    optimizer.mesh_after(step, obj_i, optimizer.shape),
    optimizer.cat_mesh(step, obj_i),
)
# %%
reload(plots)
plots.plot_step_single(optimizer.mesh_before(step, obj_i, optimizer.shape), optimizer.cat_mesh(step, obj_i), cat_opacity=1)

# %%
# store cat mesh in file
issue_name = "cat_incorrect1"
cat_mesh = optimizer.cat_mesh(step, obj_i)
filename = f"{issue_name}-cat[o{obj_i}i{step}].stl"
cat_mesh.save("../dump/" + filename)

# %%


@pprofile
def profile_optimizer():
    optimizer.run()


profile_optimizer()
# # %%


# %%


fig, ax = plt.subplots()

a = [0.05, 0.15, 0.25]
b = [0.1, 0.2, 0.3, 0.5]

x = np.linspace(0, 1, 100)

for ai in a:
    for bi in b:
        print(f"{ai} {bi}`")
        ax.plot(mesh_simplification_condition(x, ai, bi), label=f"a:{ai:.2f},  b:{bi:.2f}")
ax.legend()
# %%
