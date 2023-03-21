# %%
import sys

sys.path.append("../irregular_object_packing/")
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
from pyvistaqt import BackgroundPlotter
from irregular_object_packing.mesh.transform import scale_and_center_mesh, scale_to_volume
from irregular_object_packing.mesh.utils import print_mesh_info
import irregular_object_packing.packing.plots as plots

pv.set_jupyter_backend("panel")

# %%


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


def downsample_mesh(mesh: trimesh.Trimesh, sample_rate: float):
    target_reduction = sample_rate / mesh.vertices

    pv_mesh = pv.wrap(mesh)
    trimesh = pyvista_to_trimesh(pv_mesh.decimate(target_reduction).triangulate().clean().extract_geometry())
    return trimesh


def compute_collisions(p_meshes: list[PolyData], plotter: BackgroundPlotter = None):
    i, colls = 0, 0
    coll_meshes = []
    for mesh_1, mesh_2 in combinations(p_meshes, 2):
        col, n_contacts = mesh_1.collision(mesh_2, 1)
        if col:
            coll_meshes.append(col)
        if n_contacts > 0:
            i += 1
            colls += n_contacts
    return i, colls, coll_meshes


## %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ### NLC optimisation with CAT cells single interation
# We will now combine the NLC optimisation with the CAT cells to create a single iteration of the optimisation.
#


def optimal_transform(k, irop_data, scale_bound=(0.1, None), max_angle=1 / 12 * np.pi, max_t=None):
    r_bound = (-max_angle, max_angle)
    t_bound = (0, max_t)
    bounds = [scale_bound, r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]
    x0 = np.array([scale_bound[0], 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    constraint_dict = {
        "type": "ineq",
        "fun": nlc.constraints_from_dict,
        "args": (
            k,
            irop_data,
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
    log_lvl: int = 3
    """The log level maximum level is 3"""


class Optimizer:
    shape0: trimesh.Trimesh
    shape: trimesh.Trimesh
    container0: trimesh.Trimesh
    container: trimesh.Trimesh
    settings: SimSettings
    cat_data: cat.CatData
    tf_arrs: np.ndarray[np.ndarray]
    object_coords: np.ndarray
    prev_tf_arrs: np.ndarray[np.ndarray]

    def __init__(self, shape: trimesh.Trimesh, container: trimesh.Trimesh, settings: SimSettings):
        self.shape0 = shape
        self.shape = shape
        self.container0 = container
        self.container = container
        self.settings = settings
        self.cat_data = None
        self.tf_arrs = np.empty(0)
        self.object_coords = np.empty(0)
        self.prev_tf_arrs = np.empty(0)
        self.data = {}
        self.plotter = None
        self.data_index = -1
        self.objects = None

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

        if self.settings.plot_intermediate:
            self.plotter = BackgroundPlotter()
            self.plotter.set_background("white")

    def setup_pbars(self):
        self.pbar1 = tqdm(range(self.settings.n_scaling_steps), desc="scaling \t", position=0)
        self.pbar2 = tqdm(range(self.settings.itn_max), desc="Iteration\t", position=1)

    def update_data(self, i_b, i):
        self.data[self.data_index] = {"tf_arrs": self.tf_arrs.copy(), "cat_data": copy(self.cat_data)}
        self.data[(i_b, i)] = self.data[self.data_index]  # nice lil' reference
        self.data_index += 1

    def log(self, msg, loglvl=0):
        if loglvl >= self.settings.loglvl:
            if self.pbar1 is None:
                self.pbar1.write(msg)
            else:
                print(msg)

    def run(self):
        self.setup()

        scaling_barrier = np.linspace(
            self.settings.init_f, self.settings.final_scale, num=self.settings.n_scaling_steps + 1
        )[1:]
        self.check_overlap()

        for i_b in range(0, self.settings.n_scaling_steps):
            self.pbar1.set_postfix(ƒ_max=f"{scaling_barrier[i_b]:.2f}")
            self.pbar2.reset()
            sample_rate = self.mesh_sample_rate(0)

            for i in range(self.settings.itn_max):
                self.iteration(scaling_barrier[i_b])

                # administrative stuff
                self.check_overlap()
                self.update_data(i_b, i)
                self.pbar2.update()
            self.pbar1.update()

        coll = self.check_overlap()

    # ----------------------------------------------------------------------------------------------
    # Optimisation
    # ----------------------------------------------------------------------------------------------
    def iteration(self, scale_bound):
        """Perform a single iteration of the optimisation"""
        obj_points = [
            trimesh.transform_points(self.shape.vertices.copy(), nlc.construct_transform_matrix(transform_data))
            for transform_data in self.tf_arrs
        ]

        self.cat_data = cat.compute_cat_cells(obj_points, self.container.vertices, self.object_coords)
        self.update_plot()

        for obj_i, transform_data_i in enumerate(self.tf_arrs):
            self.pbar2.set_postfix(obj_id=obj_i)
            self.local_optimisation(obj_i, self.cat_data, transform_data_i, scale_bound)
            self.update_plot()

    def local_optimisation(self, obj_id, cat_data, transform_data_i, max_scale):
        # self.prev_tf_arrs[obj_id] = transform_data_i.copy()
        tf_arr = optimal_transform(
            obj_id,
            cat_data,
            scale_bound=(self.settings.init_f, max_scale),
            max_angle=self.settings.max_a,
            max_t=self.settings.max_t,
        )
        new_tf = transform_data_i + tf_arr
        new_tf[0] = tf_arr[0]  # * transform_data_i[0]
        self.tf_arrs[obj_id] = new_tf
        self.object_coords[obj_id] = new_tf[4:]

    # ----------------------------------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------------------------------
    @property
    def n_objs(self):
        return len(self.object_coords)

    def resample_meshes(self):
        self.container = downsample_mesh(self.container0, self.container_sample_rate())
        self.shape = downsample_mesh(self.shape0, self.mesh_sample_rate(0))

    def check_overlap(self, plotter: BackgroundPlotter = None):
        p_meshes = self.get_processed_meshes()
        i, colls, coll_meshes = compute_collisions(plotter, p_meshes)

        if i > 0:
            self.log(f"! collision found for {i} objectts with total of {colls} contacts", 3)
            if self.plotter is None: return
                
            for mesh in coll_meshes:
                self.plotter.add_mesh(mesh)

            self.plotter.render()


    def mesh_sample_rate(self, k):
        return self.settings.sample_rate  # currently simple

    def container_sample_rate(self):
        return self.settings.sample_rate * 5

    def get_processed_meshes(self) -> list[PolyData]:
        object_meshes = []

        for i in range(self.n_objs):
            transform_matrix = nlc.construct_transform_matrix(self.tf_arrs[i])
            object_mesh = self.shape.copy().apply_transform(transform_matrix)
            object_meshes.append(pv.wrap(object_mesh).decimate(0.1))

        return object_meshes

    def get_cat_meshes(self) -> list[PolyData]:
        cat_meshes = []

        for i in range(self.n_objs):
            cat_points, cat_faces = cat.face_coord_to_points_and_faces(self.cat_data, i)
            poly_data = pv.PolyData(cat_points, cat_faces)

            cat_meshes.append(poly_data)

        return cat_meshes

    def update_plot(self):
        if self.plotter is None: return

        self.plotter.clear_actors()
        self.plotter.add_mesh(self.container, color="grey", opacity=0.2)
        colors = plots.generate_tinted_colors(self.n_objs)
        for i, mesh in enumerate(self.get_processed_meshes()):
            self.plotter.add_mesh(mesh, color=colors[1][i])

        for i, mesh in enumerate(self.get_cat_meshes()):
            self.plotter.add_mesh(mesh, color=colors[0][i])

        self.plotter.render()

    @staticmethod
    def default_setup() -> "Optimizer":
        DATA_FOLDER = "./data/mesh/"

        mesh_volume = 0.99
        container_volume = 10

        loaded_mesh = trimesh.load_mesh(DATA_FOLDER + "RBC_normal.stl")
        container = trimesh.primitives.Cylinder(radius=1, height=1)

        # Scale the mesh and container to the desired volume
        container = scale_to_volume(container, container_volume)
        original_mesh = scale_and_center_mesh(loaded_mesh, mesh_volume)

        settings = SimSettings(
            itn_max=2,
            n_scaling_steps=2,
            final_scale=0.3,
            sample_rate=30,
        )
        optimizer = Optimizer(original_mesh, container, settings)
        optimizer.run()
        return optimizer

    def report(self):
        df = pd.DataFrame(data=self.tf_arrs, columns=["scale", "r_x", "ry", "rz", "t_x", "t_y", "t_z"])
        return df


# %%
from importlib import reload

reload(plots)

optimizer = Optimizer.default_setup()
optimizer.report()


# # %%
# plotter = pv.Plotter()
# # enumerate
# tints = plots.generate_tinted_colors(optimizer.n_objs)

# for i, mesh in enumerate(optimizer.get_processed_meshes()):
#     plotter.add_mesh(mesh, color=tints[1][i], opacity=0.8)

# for i, mesh in enumerate(optimizer.get_cat_meshes()):
#     plotter.add_mesh(mesh, color=tints[0][i], opacity=0.5)

# # plotter.add_mesh(optimizer.container, color="grey", opacity=0.2)

# plotter.show(
#     interactive=False,
# )
# # %%
# %%
