# %%
import sys

sys.path.append("../irregular_object_packing/")
sys.path.append("../irregular_object_packing/irregular_object_packing/")


# %%

from collections import namedtuple
from copy import copy
import pandas as pd
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from pyvista import PolyData
from scipy.optimize import minimize
from tqdm.auto import tqdm
import trimesh
import pyvista as pv
from irregular_object_packing.mesh.transform import scale_and_center_mesh, scale_to_volume
from irregular_object_packing.mesh.utils import print_mesh_info
from irregular_object_packing.tools.profile import pprofile
import irregular_object_packing.packing.plots as plots
from irregular_object_packing.packing.OptimizerData import OptimizerData

# pv.set_jupyter_backend("panel")
LOG_LVL_SEVERE = 3
LOG_LVL_WARNING = 2
LOG_LVL_INFO = 1
LOG_LVL_DEBUG = 0
LOG_LVL_NO_LOG = -1
LOG_PREFIX = ["[DEBUG]", "[INFO]", "[WARNING]", "[SEVERE]"]

from irregular_object_packing.packing import (
    initialize as init,
    nlc_optimisation as nlc,
    chordal_axis_transform as cat,
)


def pyvista_to_trimesh(mesh: PolyData):
    points = mesh.points
    faces = mesh.faces.reshape(mesh.n_faces, 4)[:, 1:]
    return trimesh.Trimesh(vertices=points, faces=faces)


def trimesh_to_pyvista(mesh: trimesh.Trimesh):
    return pv.wrap(mesh)


# NOTE: Unfortunately this method produces too many issues i dont know to deal with rn. reconsidering design...
def downsample_mesh(mesh: trimesh.Trimesh, sample_rate: float):
    nvertices = len(mesh.vertices)
    if not nvertices > sample_rate:
        return mesh

    target_reduction = sample_rate / nvertices

    pv_mesh = pv.wrap(mesh)
    trimesh = pyvista_to_trimesh(pv_mesh.decimate(target_reduction).clean().triangulate().extract_geometry())
    return trimesh


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
    return mesh.collision(container, mode)


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

    for i, (mesh, cat_mesh) in enumerate(zip(p_meshes, cat_meshes)):
        violation = compute_boundary_violation(mesh, cat_mesh)
        if violation[1] > 0:
            violations.append(i)
            viol_meshes.append(violation[0])
            n += 1
    return n, violations, viol_meshes


## %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ### NLC optimisation with CAT cells single interation
# We will now combine the NLC optimisation with the CAT cells to create a single iteration of the optimisation.
#


def optimal_local_transform(
    obj_id, cat_data, scale_bound=(0.1, None), max_angle=1 / 12 * np.pi, max_t=None, margin=None
):
    """Computes the optimal local transform for a given object id. This will return the transformation parameters that
    maximises scale with respect to a local coordinate system of the object. This is possible due to the `obj_coords`.
    """

    r_bound = (-max_angle, max_angle)
    t_bound = (0, max_t)
    bounds = [scale_bound, r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]
    x0 = np.array([scale_bound[0], 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    constraint_dict = {
        "type": "ineq",
        "fun": nlc.local_constraints_from_cat,
        "args": (
            obj_id,
            cat_data,
            margin,
        ),
    }
    res = minimize(nlc.objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict)
    return res.x


# SimSettings as dataclass:
@dataclass
class SimSettings:
    sample_rate: int = 50
    """The sample rate of the object surface mesh"""
    max_a: float = 1 / 12 * np.pi
    """The maximum rotation angle per growth step"""
    max_t: float = None
    """The maximum translation per growth step"""
    init_f: float = 0.1
    """Final scale"""
    final_scale: float = 1.0
    """The initial scale factor"""
    itn_max: int = 1
    """The maximum number of iterations per scaling step"""
    n_scaling_steps: int = 1
    """The number of scaling steps"""
    r: float = 0.3
    """The coverage rate"""
    plot_intermediate: bool = False
    """Whether to plot intermediate results"""
    log_lvl: int = -1
    """The log level maximum level is 3"""


class Optimizer(OptimizerData):
    shape0: trimesh.Trimesh
    shape: trimesh.Trimesh
    container0: trimesh.Trimesh
    container: trimesh.Trimesh
    settings: SimSettings
    cat_data: cat.CatData
    tf_arrs: np.ndarray[np.ndarray]
    object_coords: np.ndarray
    prev_tf_arrs: np.ndarray[np.ndarray]

    def __init__(self, shape: trimesh.Trimesh, container: trimesh.Trimesh, settings: SimSettings, plotter=None):
        self.shape0 = shape
        self.shape = shape
        self.container0 = container
        self.container = container
        self.pv_shape = trimesh_to_pyvista(shape).decimate_pro(0.01)
        self.settings = settings
        self.cat_data = None
        self.tf_arrs = np.empty(0)
        self.object_coords = np.empty(0)
        self.prev_tf_arrs = np.empty(0)
        self.plotter: pv.Plotter = plotter
        self.data_index = -1
        self.objects = None
        self.pbar1 = self.pbar2 = None
        self.margin = None

    # ----------------------------------------------------------------------------------------------
    # SETUP functions
    # ----------------------------------------------------------------------------------------------
    def setup(self):
        self.resample_meshes()
        self.object_coords = init.init_coordinates(
            self.container,
            self.shape,
            coverage_rate=self.settings.r,
            c_scale=0.9,
        )

        init_f = self.settings.init_f
        object_rotations = np.random.uniform(-np.pi, np.pi, (self.n_objs, 3))

        # SET TRANSFORM DATA
        self.tf_arrs = np.empty((self.n_objs, 7))
        self.prev_tf_arrs = np.empty((self.n_objs, 7))

        for i in range(self.n_objs):
            tf_arr_i = np.array([init_f, *object_rotations[i], *self.object_coords[i]])
            self.tf_arrs[i] = tf_arr_i

        self.update_data(-1, -1)
        self.setup_pbars()
        self.margin = self.pv_shape.volume ** (1 / 3) / self.settings.sample_rate * (1 / 10)
        if self.plotter is not None:
            self.plotter.show(interactive=True, interactive_update=True)

    def setup_pbars(self):
        self.pbar1 = tqdm(range(self.settings.n_scaling_steps), desc="scaling \t", position=0)
        self.pbar2 = tqdm(range(self.settings.itn_max), desc="Iteration\t", position=1)

    # ----------------------------------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------------------------------
    def update_data(self, i_b, i):
        self.log(f"Updating data for {i_b=}, {i=}")
        self.add(self.tf_arrs, self.cat_data, (i_b, i))

    def log(self, msg, log_lvl=0):
        if log_lvl > self.settings.log_lvl:
            return

        msg = LOG_PREFIX[log_lvl] + msg
        if self.pbar1 is None:
            print(msg)
        else:
            self.pbar1.write(msg)

    def report(self):
        df = pd.DataFrame(data=self.tf_arrs, columns=["scale", "r_x", "ry", "rz", "t_x", "t_y", "t_z"])
        return df

    def mesh_sample_rate(self, k=0):
        return self.settings.sample_rate  # currently simple

    def container_sample_rate(self):
        return self.settings.sample_rate * 2 * self.n_objs

    def resample_meshes(self):
        self.log("resampling meshes", LOG_LVL_DEBUG)
        self.container = self.container0
        self.shape = self.shape0
        # currently doing nothing
        # self.container = downsample_mesh(self.container0.to_mesh(), self.container_sample_rate())
        # self.shape = downsample_mesh(self.shape0, self.mesh_sample_rate(0))

    @property
    def n_objs(self):
        return len(self.object_coords)

    def add_to_plot(self, coll_meshes):
        if self.plotter is None:
            return

        for mesh in coll_meshes:
            self.plotter.add_mesh(mesh)

        self.plotter.render()

    def update_plot(self):
        if self.plotter is None:
            return

        self.plotter.clear_actors()
        self.plotter.add_mesh(self.container.to_mesh(), color="white", opacity=0.2)
        colors = plots.generate_tinted_colors(self.n_objs)
        for i, mesh in enumerate(self.final_meshes_after(self.pv_shape)):
            self.plotter.add_mesh(mesh, color=colors[1][i], opacity=0.7)

        cat_meshes = [
            PolyData(*cat.face_coord_to_points_and_faces(self.cat_data, obj_id)) for obj_id in range(self.n_objs)
        ]

        for i, mesh in enumerate(cat_meshes):
            self.plotter.add_mesh(mesh, color=colors[0][i], opacity=0.5)

        self.plotter.update()

    def run(self):
        self.setup()

        scaling_barrier = np.linspace(
            self.settings.init_f, self.settings.final_scale, num=self.settings.n_scaling_steps + 1
        )[1:]

        self.check_object_overlap()
        self.check_container_violations()

        for i_b in range(0, self.settings.n_scaling_steps):
            self.pbar1.set_postfix(ƒ_max=f"{scaling_barrier[i_b]:.2f}")
            self.pbar2.reset()

            for i in range(self.settings.itn_max):
                self.iteration(scaling_barrier[i_b])

                # administrative stuff
                self.update_data(i_b, i)
                self.check_validity()
                self.pbar2.update()
            self.pbar1.update()

    # ----------------------------------------------------------------------------------------------
    # Optimisation
    # ----------------------------------------------------------------------------------------------
    def iteration(self, scale_bound):
        """Perform a single iteration of the optimisation"""
        # DOWN SAMPLE MESHES
        container_points = trimesh.sample.sample_surface_even(self.container, self.container_sample_rate())[0]
        # container_points = self.container.to_mesh().vertices
        sample_points = trimesh.sample.sample_surface_even(self.shape, self.mesh_sample_rate())[0]
        # sample_points = self.shape.vertices

        # TRANSFORM MESHES TO OBJECT COORDINATES, SCALE, ROTATION
        obj_points = [
            trimesh.transform_points(sample_points.copy(), nlc.construct_transform_matrix(transform_data))
            for transform_data in self.tf_arrs
        ]

        # COMPUTE CAT CELLS
        self.log("Computing CAT cells")
        self.cat_data = cat.compute_cat_cells(obj_points, container_points, self.object_coords)
        self.check_closed_cells()
        self.update_plot()

        # GROWTH-BASED OPTIMISATION
        for obj_id, transform_data_i in enumerate(self.tf_arrs):
            self.pbar2.set_postfix(obj_id=obj_id)
            self.local_optimisation(obj_id, transform_data_i, scale_bound)

        self.update_plot()

    def local_optimisation(self, obj_id, transform_data_i, max_scale):
        tf_arr = optimal_local_transform(
            obj_id,
            self.cat_data,
            scale_bound=(self.settings.init_f, None),
            max_angle=self.settings.max_a,
            max_t=self.settings.max_t,
            margin=self.margin,
        )

        # final guess... #FIXME
        new_tf = transform_data_i + tf_arr
        new_scale = new_tf[0]
        if new_scale > max_scale:
            new_scale = max_scale

        new_tf[0] = new_scale
        self.tf_arrs[obj_id] = new_tf
        self.object_coords[obj_id] = new_tf[4:]

    # ----------------------------------------------------------------------------------------------
    # VALIDITY CHECKS
    # ----------------------------------------------------------------------------------------------

    def check_validity(self):
        self.check_cat_boundaries()
        self.check_container_violations()
        self.check_object_overlap()

    def check_closed_cells(self):
        cat_cells = [
            PolyData(*cat.face_coord_to_points_and_faces(self.cat_data, obj_id)) for obj_id in range(self.n_objs)
        ]
        for i, cell in enumerate(cat_cells):
            if not cell.is_manifold:
                self.log(f"CAT cell of object {i} is not manifold", log_lvl=LOG_LVL_WARNING)

    def check_object_overlap(self):
        self.log("checking for collisions", LOG_LVL_DEBUG)
        p_meshes = self.final_meshes_after(self.pv_shape)
        i, colls, coll_meshes = compute_collisions(p_meshes)

        if i > 0:
            self.log(f"! collision found for {i} objectts with total of {colls} contacts", LOG_LVL_SEVERE)
            return self.add_to_plot(coll_meshes)

    def check_cat_boundaries(self):
        self.log("checking for cat boundary violations", LOG_LVL_DEBUG)
        p_meshes = self.final_meshes_after(self.pv_shape)
        cat_meshes = self.final_cat_meshes()
        n, violations, meshes = compute_cat_violations(p_meshes, cat_meshes)

        if n > 0:
            self.log(f"! cat boundary violation for objects {violations}", LOG_LVL_SEVERE)
            self.add_to_plot(meshes)

    def check_container_violations(self):
        self.log("checking for container violations", LOG_LVL_DEBUG)
        p_meshes = self.final_meshes_after(self.pv_shape)
        n, violations, meshes = compute_container_violations(p_meshes, pv.wrap(self.container.to_mesh()))

        if n > 0:
            self.log(f"! container violation for objects {violations}", LOG_LVL_SEVERE)
            self.add_to_plot(meshes)

    @staticmethod
    def default_setup() -> "Optimizer":
        DATA_FOLDER = "./data/mesh/"

        # mesh_volume = 0.2
        mesh_volume = 0.3
        container_volume = 10

        loaded_mesh = trimesh.load_mesh(DATA_FOLDER + "RBC_normal.stl")
        # loaded_mesh = trimesh.primitives.Box().to_mesh()
        # container = trimesh.primitives.Box()
        container = trimesh.primitives.Cylinder()

        # Scale the mesh and container to the desired volume
        container = scale_to_volume(container, container_volume)
        original_mesh = scale_and_center_mesh(loaded_mesh, mesh_volume)

        settings = SimSettings(
            itn_max=1,
            n_scaling_steps=1,
            r=0.3,
            final_scale=0.1,
            sample_rate=100,
            log_lvl=3,
            init_f=0.05,  # NOTE: Smaller than paper
        )
        plotter = None
        # plotter = pv.Plotter()
        optimizer = Optimizer(original_mesh, container, settings, plotter)
        return optimizer


# %%
from importlib import reload


optimizer = Optimizer.default_setup()
optimizer.run()


# %%


# %%
reload(plots)
plotter = pv.Plotter()
# enumerate
plots.plot_full_comparison(
    optimizer.meshes_before(0, optimizer.pv_shape),
    # optimizer.final_meshes_before(optimizer.pv_shape),
    optimizer.final_meshes_after(optimizer.pv_shape),
    optimizer.final_cat_meshes(),
    optimizer.container.to_mesh(),
    plotter,
)
# %%
obj_i = 0
plots.plot_step_comparison(
    optimizer.mesh_before(0, obj_i, optimizer.pv_shape),
    optimizer.mesh_after(0, obj_i, optimizer.pv_shape),
    optimizer.cat_mesh(0, obj_i),
)


# %%
@pprofile
def profile_optimizer():
    optimizer.run()


# profile_optimizer()

# %%
