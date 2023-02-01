# %%
import numpy as np
import pyvista as pv
import trimesh

from irregular_object_packing.packing.chordal_axis_transform import (
    compute_cat_cells,
    face_coord_to_points_and_faces,
)
from irregular_object_packing.packing.initialize import create_packed_scene, pack_objects, save_image
from irregular_object_packing.packing.plots import create_plot
from irregular_object_packing.tools.profile import cprofile, pprofile


def translation_matrix(x0, x1):
    return np.array([[1, 0, 0, x1[0] - x0[0]], [0, 1, 0, x1[1] - x0[1]], [0, 0, 1, x1[2] - x0[2]], [0, 0, 0, 1]])


DATA_FOLDER = "./data/mesh/"
original_mesh = trimesh.load_mesh(DATA_FOLDER + "yog.obj")
# mesh = trimesh.primitives.Capsule(radius=1, height=1)
container = trimesh.primitives.Cylinder(radius=1, height=1)

container = container.apply_scale(40)

objects_coords = pack_objects(container, original_mesh, 0.2)

scene = create_packed_scene(container, objects_coords, original_mesh)

# %%
mesh = trimesh.sample.sample_surface_even(original_mesh, 1000)[0]
obj_points = []
for object_coords in objects_coords:
    # get the object
    object = mesh.copy()
    # apply the transformation
    points = trimesh.transform_points(object, translation_matrix(np.array([0, 0, 0]), object_coords))
    # object = object.apply_transform(translation_matrix(np.array([0, 0, 0]), object_coords))
    # get the points of the object
    # points = object.vertices
    # add the points to the list of points
    obj_points.append(points)


# %%


@pprofile
def prof_compute_cat_cells():
    return compute_cat_cells(obj_points, trimesh.sample.sample_surface_even(container, 10000)[0])


# cat_cells = prof_compute_cat_cells()

cat_cells = compute_cat_cells(obj_points, trimesh.sample.sample_surface_even(container, 10000)[0])
# %%


@pprofile
def prof_face_coord_to_points_and_faces():
    return face_coord_to_points_and_faces(cat_cells[0])


# cat_points, poly_faces = prof_face_coord_to_points_and_faces()

cat_points, poly_faces = face_coord_to_points_and_faces(cat_cells[0])


# %%
from tqdm import tqdm

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
    object_mesh.apply_scale(0.3)
    object_mesh.vertices = trimesh.transformations.transform_points(
        object_mesh.vertices, translation_matrix(np.array([0, 0, 0]), objects_coords[k])
    )
    # object_mesh.transform(np.eye(4), object_coords[k])
    object_meshes.append(object_mesh)


# %%
# TODO: create the scene with the cat_meshes
# TODO: make sure the compute cat cells stops printing

create_plot(objects_coords, object_meshes, cat_meshes, container.to_mesh())
# %%
