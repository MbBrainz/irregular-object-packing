# %%
from collections import namedtuple
from copy import copy
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from pyvista import PolyData
from scipy.optimize import minimize
from tqdm.auto import tqdm
import trimesh
import pyvista as pv
import sys
from irregular_object_packing.mesh.transform import scale_and_center_mesh, scale_to_volume
from irregular_object_packing.mesh.utils import print_mesh_info

sys.path.append("../irregular_object_packing/")

from irregular_object_packing.packing import initialize, nlc_optimisation as nlc, chordal_axis_transform as cat

## %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ### NLC optimisation with CAT cells single interation
# We will now combine the NLC optimisation with the CAT cells to create a single iteration of the optimisation.
#
# %%


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
    max_angle: float = 1 / 12 * np.pi
    max_t: float = None
    initial_scale: float = 0.1
    itn_max: int = 1
    n_scaling_steps: int = 1
    coverage_rate: float = 0.3


class Optimizer:
    shapes: list[trimesh.Trimesh]
    container: trimesh.Trimesh
    settings: SimSettings
    cat_data: cat.CatData
    transform_data: np.ndarray[np.ndarray]
    object_coords: np.ndarray
    previous_transform_data: np.ndarray[np.ndarray]
    n_objects: int = 0

    def __init__(self, shapes: list[trimesh.Trimesh], container: trimesh.Trimesh, settings: SimSettings):
        self.shapes = shapes
        self.container = container
        self.settings = settings
        self.cat_data = None
        self.transform_data = np.empty(0)
        self.object_coords = np.empty(0)
        self.previous_transform_data = np.empty(0)
        self.tf_log = {}
        self.cat_log = {}

    def run(self):
        # NOTE: Hardcoded mesh index 0
        m_id = 0
        self.setup()

        scaling_barier = np.linspace(self.settings.initial_scale, 1, num=self.settings.n_scaling_steps + 1)
        self.check_overlap()

        for i_b in (pbar1 := tqdm(range(0, len(scaling_barier) - 1), desc="scaling_barrier", position=0)):
            pbar1.set_postfix([("max_scale", scaling_barier[i_b + 1])])

            # for i in range(self.settings.itn_max):
            for i in (
                pbar2 := tqdm(
                    range(self.settings.itn_max), desc="Iteration", postfix=["obj_id", dict(value=0)], position=1
                )
            ):
                mesh_sample_rate = self.mesh_sample_rate(m_id)

                # Resample surface meshes
                container_points = trimesh.sample.sample_surface_even(
                    self.container, self.container_sample_rate()
                )[0]
                shape_points_list = [
                    trimesh.sample.sample_surface_even(mesh, mesh_sample_rate[i])[0]
                    for i, mesh in enumerate(self.shapes)
                ]
                mesh = shape_points_list[m_id]

                obj_points = [
                    trimesh.transform_points(mesh.copy(), nlc.construct_transform_matrix(transform_data))
                    for transform_data in self.transform_data
                ]

                self.cat_data = cat.compute_cat_cells(obj_points, container_points, self.object_coords)

                scale_bound = (scaling_barier[i_b], scaling_barier[i_b + 1])
                for obj_i, transform_data_i in enumerate(self.transform_data):
                    pbar2.set_postfix(obj_id=obj_i)
                    self.previous_transform_data[obj_i] = transform_data_i.copy()
                    tf_arr = optimal_transform(
                        obj_i,
                        self.cat_data,
                        scale_bound=scale_bound,
                        max_angle=self.settings.max_angle,
                        max_t=self.settings.max_t,
                    )
                    new_tf = transform_data_i + tf_arr
                    new_tf[0] = tf_arr[0]  # * transform_data_i[0]
                    self.transform_data[obj_i] = new_tf
                    self.object_coords[obj_i] = new_tf[4:]

                self.cat_log[i_b * self.settings.itn_max + i] = copy(self.cat_data)
                self.tf_log[i_b * self.settings.itn_max + i + 1] = self.transform_data.copy()

        self.cat_log[0] = copy(self.cat_log[1])
        coll = self.check_overlap()

        print(f"Collision: {coll}")
        print(f"Optimisation complete. scales: {[t[0] for t in self.transform_data]}")

    def setup(self):
        m_id = 0
        object_coords = initialize.place_objects(
            self.container, self.shapes[m_id], coverage_rate=self.settings.coverage_rate, c_scale=0.9
        )
        self.n_objects = len(object_coords)

        object_rotations = np.random.uniform(-np.pi, np.pi, (self.n_objects, 3))
        object_scales = np.ones(self.n_objects) * self.settings.initial_scale

        # SET TRANSFORM DATA
        self.transform_data = np.empty((self.n_objects, 7))
        self.previous_transform_data = np.empty((self.n_objects, 7))
        for i in range(self.n_objects):
            tf_i = np.array([object_scales[i], *object_rotations[i], *object_coords[i]])
            self.transform_data[i] = tf_i

        self.tf_log[0] = self.transform_data.copy()
        self.object_coords = object_coords

    def check_overlap(self, pbar: tqdm = None):
        p_meshes = self.get_processed_meshes()
        i, colls = 0, 0
        for mesh_1, mesh_2 in combinations(p_meshes, 2):
            col, n_contacts = mesh_1.collision(mesh_2, 1)
            if n_contacts > 0:
                i += 1
                colls += n_contacts

        if i > 0:
            if pbar != None:
                pbar.write(f"! collision found for {i} objectts with total of {colls} contacts")
            else:
                print(f"! collision found for {i} objectts with total of {colls} contacts")
        # return coll_data

    def mesh_sample_rate(self, k):
        # TODO: implement a function that returns the sample rate for each mesh based on the relative volume wrt the container
        return [self.settings.sample_rate for i in range(len(self.shapes))]  # currently simple

    def container_sample_rate(self):
        return self.settings.sample_rate * 10

    def get_processed_meshes(self) -> list[PolyData]:
        object_meshes = []

        for i in range(self.n_objects):
            transform_matrix = nlc.construct_transform_matrix(self.transform_data[i])
            object_mesh = self.shapes[0].copy().apply_transform(transform_matrix)
            object_meshes.append(pv.wrap(object_mesh))

        return object_meshes

    def get_cat_meshes(self) -> list[PolyData]:
        cat_meshes = []

        for i in range(self.n_objects):
            cat_points, cat_faces = cat.face_coord_to_points_and_faces(self.cat_data, i)
            poly_data = pv.PolyData(cat_points, cat_faces)

            cat_meshes.append(poly_data)

        return cat_meshes


# %%
