# %%
from collections import namedtuple
from copy import copy
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import trimesh
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
    x0 = np.array([0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

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


SimSettings = namedtuple("sim_settings", ["sample_rate", "max_angle", "max_t", "initial_scale", "i_max"])


class Optimizer:
    meshes: list[trimesh.Trimesh]
    container: trimesh.Trimesh
    settings: SimSettings
    cat_data: cat.CatData
    transform_data: np.ndarray[np.ndarray]
    previous_transform_data: np.ndarray[np.ndarray]
    n_objects: int = 0

    def __init__(self, meshes: list[trimesh.Trimesh], container: trimesh.Trimesh, settings: SimSettings):
        self.meshes = meshes
        self.container = container
        self.settings = settings
        self.cat_data = None
        self.transform_data = np.empty(0)
        self.previous_transform_data = np.empty(0)
        self.transform_log = []
        self.cat_log = []

    def run(self):
        # NOTE: Hardcoded mesh index 0
        m_id = 0
        object_coords = self.setup()

        for i in (pbar := tqdm(range(self.settings.i_max), desc="Iteration", postfix=["obj_id ", dict(value=0)])):
            mesh_sample_rate = self.mesh_sample_rate(m_id)

            # Resample surface meshes
            container_points = trimesh.sample.sample_surface_even(self.container, self.container_sample_rate())[0]
            meshes = [
                trimesh.sample.sample_surface_even(mesh, mesh_sample_rate[i])[0]
                for i, mesh in enumerate(self.meshes)
            ]
            mesh = meshes[m_id]

            obj_points = [
                trimesh.transform_points(mesh.copy(), nlc.construct_transform_matrix(transform_data))
                for transform_data in self.transform_data
            ]

            self.cat_data = cat.compute_cat_cells(obj_points, container_points, object_coords)

            scale_bound = (0.1, None)
            for k, transform_data in enumerate(self.transform_data):
                pbar.set_postfix(obj_id=k)
                self.previous_transform_data[k] = transform_data.copy()
                tf_arr = optimal_transform(
                    k, self.cat_data, scale_bound, max_angle=self.settings.max_angle, max_t=self.settings.max_t
                )
                new_tf = transform_data + tf_arr
                new_tf[0] = tf_arr[0] * transform_data[0]
                self.transform_data[k] = new_tf

            self.cat_log.append(copy(self.cat_data))
            self.transform_log.append(self.transform_data.copy())

    def setup(self):
        m_id = 0
        object_coords = initialize.place_objects(self.container, self.meshes[m_id], coverage_rate=0.3, c_scale=0.9)
        self.n_objects = len(object_coords)

        object_rotations = np.random.uniform(-np.pi, np.pi, (self.n_objects, 3))
        object_scales = np.ones(self.n_objects) * self.settings.initial_scale

        # SET TRANSFORM DATA
        self.transform_data = np.empty((self.n_objects, 7))
        self.previous_transform_data = np.empty((self.n_objects, 7))
        for i in range(self.n_objects):
            tf_i = np.array([object_scales[i], *object_rotations[i], *object_coords[i]])
            self.transform_data[i] = tf_i

        self.transform_log.append(self.transform_data.copy())
        return object_coords

    def mesh_sample_rate(self, k):
        # TODO: implement a function that returns the sample rate for each mesh based on the relative volume wrt the container
        return [self.settings.sample_rate for i in range(len(self.meshes))]  # currently simple

    def container_sample_rate(self):
        return self.settings.sample_rate * 10
