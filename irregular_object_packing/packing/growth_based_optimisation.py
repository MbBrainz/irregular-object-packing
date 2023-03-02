# %%
from collections import namedtuple
import numpy as np
from scipy.optimize import minimize
import trimesh
import sys
from irregular_object_packing.mesh.transform import scale_and_center_mesh, scale_to_volume

sys.path.append("../irregular_object_packing/")
from irregular_object_packing.mesh.utils import print_mesh_info

from irregular_object_packing.packing import initialize, nlc_optimisation as nlc

# from irregular_object_packing.packing import packing as pkn
import irregular_object_packing.packing.chordal_axis_transform as cat

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


SimSettings = namedtuple("sim_settings", ["sample_rate", "max_angle", "max_t", "initial_scale"])


class Simulation:
    meshes: list[trimesh.Trimesh]
    container: trimesh.Trimesh
    settings: SimSettings
    cat_data: cat.IropData
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

    def mesh_sample_rate(self, k):
        # TODO: implement a function that returns the sample rate for each mesh based on the relative volume wrt the container
        return [self.settings.sample_rate for i in range(len(self.meshes))]  # currently simple

    def container_sample_rate(self):
        return self.settings.sample_rate * 10

    def run(self):
        # NOTE: Hardcoded mesh index 0
        m_id = 0
        object_coords = initialize.place_objects(self.container, self.meshes[m_id], coverage_rate=0.3, c_scale=0.9)
        self.n_objects = len(object_coords)

        # object_rotations = initialize.random_rotations(len(object_coords))
        object_rotations = np.random.uniform(-np.pi, np.pi, (self.n_objects, 3))
        object_scales = np.ones(self.n_objects) * self.settings.initial_scale

        # SET TRANSFORM DATA
        self.transform_data = np.empty((self.n_objects, 7))
        self.previous_transform_data = np.empty((self.n_objects, 7))
        for i in range(self.n_objects):
            tf_i = np.array([object_scales[i], *object_rotations[i], *object_coords[i]])
            self.transform_data[i] = tf_i

        # for i in range(len(self.settings.i_max)):
        # first iteration

        mesh_sample_rate = self.mesh_sample_rate(m_id)

        # RESAMPLE SURFACE MESHES
        container_points = trimesh.sample.sample_surface_even(self.container, self.container_sample_rate())[0]
        meshes = [
            trimesh.sample.sample_surface_even(self.meshes[i], mesh_sample_rate[i])[0]
            for i in range(len(self.meshes))
        ]

        # single mesh for simplicity
        mesh = meshes[m_id]

        obj_points = []

        for obj_id in range(self.n_objects):
            M = nlc.construct_transform_matrix(self.transform_data[obj_id])
            object_i = mesh.copy()
            points = trimesh.transform_points(object_i, M)
            obj_points.append(points)

        self.cat_data = cat.compute_cat_cells(obj_points, container_points, object_coords)

        scale_bound = (0.1, None)
        for k in range(self.n_objects):
            self.previous_transform_data[k] = self.transform_data[k]
            tf_arr = optimal_transform(
                k, self.cat_data, scale_bound, max_angle=self.settings.max_angle, max_t=self.settings.max_t
            )
            new_tf = self.transform_data[k] + tf_arr
            new_tf[0] = tf_arr[0] * self.transform_data[k][0]
            self.transform_data[k] = new_tf


mesh_volume = 0.1
container_volume = 10
coverage_rate = 0.3

DATA_FOLDER = "../../data/mesh/"
loaded_mesh = trimesh.load_mesh(DATA_FOLDER + "RBC_normal.stl")
print_mesh_info(loaded_mesh, "loaded mesh")
# trimesh.Scene([loaded_mesh]).show()
# Scale the mesh to the desired volume
original_mesh = scale_and_center_mesh(loaded_mesh, mesh_volume)
print_mesh_info(original_mesh, "scaled mesh")

container = trimesh.primitives.Sphere(radius=1, center=[0, 0, 0])
# container = trimesh.primitives.Cylinder(radius=1, height=1)
print_mesh_info(container, "original container")

container = scale_to_volume(container, container_volume)
print_mesh_info(container, "scaled container")

settings = SimSettings(sample_rate=200, max_angle=1 / 12 * np.pi, max_t=None, initial_scale=0.1)
simulation = Simulation([original_mesh], container, settings)

# %%
simulation.run()
# %%
from irregular_object_packing.packing.plots import plot_step_comparison


test_k = 1
cat_cell_mesh = cat.cat_mesh_from_data(simulation.cat_data, test_k)

plot_step_comparison(
    original_mesh,
    [simulation.previous_transform_data[test_k], simulation.transform_data[test_k]],
    cat_cell_mesh,
)

# %%
print(simulation.previous_transform_data[test_k])
print(simulation.transform_data[test_k])

# %%
