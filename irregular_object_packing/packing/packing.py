"""This file contains the main functions to pack objects in a container. It is the main file for the packing module. Initially, this might be a bit cluttered, but it will be cleaned up in the future."""
# %%
import numpy as np
import pyvista as pv
import trimesh

from irregular_object_packing.mesh.transform import scale_and_center_mesh, scale_to_volume, translation_matrix
from irregular_object_packing.mesh.utils import print_mesh_info
from irregular_object_packing.packing.chordal_axis_transform import (
    compute_cat_cells,
    face_coord_to_points_and_faces,
)
from irregular_object_packing.packing.initialize import create_packed_scene, place_objects, save_image
from irregular_object_packing.packing.plots import create_plot

# lets define a mesh size and a container size
mesh_volume = 0.1
container_volume = 10
coverage_rate = 0.3


DATA_FOLDER = "./data/mesh/"
loaded_mesh = trimesh.load_mesh(DATA_FOLDER + "RBC_normal.stl")
print_mesh_info(loaded_mesh, "loaded mesh")
trimesh.Scene([loaded_mesh]).show()


# %%
# Scale the mesh to the desired volume
original_mesh = scale_and_center_mesh(loaded_mesh, mesh_volume)
print_mesh_info(original_mesh, "scaled mesh")
# %%
# mesh = trimesh.primitives.Capsule(radius=1, height=1)
container = trimesh.primitives.Cylinder(radius=1, height=1)
print_mesh_info(container, "original container")

container = scale_to_volume(container, container_volume)
print_mesh_info(container, "scaled container")

# %%
# Initial placement of the objects
objects_coords = place_objects(container, original_mesh, coverage_rate=coverage_rate, c_scale=0.9)

scene = create_packed_scene(container, objects_coords, original_mesh, rotate=True)
scene.show()

# %%
# get vertices of the object meshes and the container

# we resample to simplify and get a more uniform distribution of points
mesh = trimesh.sample.sample_surface_even(original_mesh, 1000)[0]
obj_points = []
rot_matrices = []
for i in range(len(objects_coords)):
    # get the object
    object = mesh.copy()
    # random rotation
    M_rot = trimesh.transformations.random_rotation_matrix()
    points = trimesh.transform_points(object, M_rot)
    rot_matrices.append(M_rot)
    # apply the transformation
    points = trimesh.transform_points(points, translation_matrix(np.array([0, 0, 0]), objects_coords[i]))

    obj_points.append(points)

container_points = trimesh.sample.sample_surface_even(container, 10000)[0]

# %%
# compute the cat cells for each object
cat_cells = compute_cat_cells(obj_points, container_points)

# %%
cat_points, poly_faces = face_coord_to_points_and_faces(cat_cells[0])

# %%
from tqdm import tqdm

# Volumetric downscale before optimizing the packing
down_scale = 0.1

object_meshes = []
cat_meshes = []
lim = len(cat_cells.keys())
for k, v in tqdm(cat_cells.items()):
    if k >= lim - 1:
        break
    cat_points, poly_faces = face_coord_to_points_and_faces(v)
    polydata = pv.PolyData(cat_points, poly_faces)
    cat_meshes.append(polydata)

    object_mesh = original_mesh.copy()
    object_mesh.apply_transform(rot_matrices[k])
    object_mesh.apply_scale(down_scale ** (1 / 3))
    object_mesh.vertices = trimesh.transformations.transform_points(
        object_mesh.vertices, translation_matrix(np.array([0, 0, 0]), objects_coords[k])
    )
    # object_mesh.transform(np.eye(4), object_coords[k])
    object_meshes.append(object_mesh)

# %%
create_plot(objects_coords, object_meshes, cat_meshes, container.to_mesh())
# %%
