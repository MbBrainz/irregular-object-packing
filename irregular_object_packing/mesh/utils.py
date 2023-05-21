# %%

import numpy as np
import pyvista as pv
from trimesh import Trimesh


def print_mesh_info(mesh: pv.PolyData, description="", suppress_scientific=True):
    with np.printoptions(precision=4, suppress=suppress_scientific):
        print(
            f"Mesh info {description}: {mesh}, \nvolume: {mesh.volume}, \nbounding box:"
            f" {mesh.bounds} \ncenter of mass: {mesh.center_of_mass()}\n"
        )


def pyvista_to_trimesh(mesh: pv.PolyData):
    tri_container = mesh.extract_surface().triangulate() # type: ignore
    faces_as_array = tri_container.faces.reshape((tri_container.n_faces, 4))[:, 1:] # type: ignore
    tri_container = Trimesh(tri_container.points, faces_as_array) # type: ignore
    return tri_container


def convert_faces_to_polydata_input(faces: np.ndarray):
    """Convert a list of faces represented by points with coordinates to
    a list of points and a list of faces represented by the number of points and point
    ids. This function is used to convert the data so that it can be used by the
    pyvista.PolyData class.

    Note: Currently this function assumes that the indices of the points
    are not global with respect to other meshes.
    """
    cat_points = []
    poly_faces = []
    n_entries = 0
    for face in faces:
        len_face = len(face)
        # assert len_face == 3, f"len_face: {len_face}"
        if len_face == 3:
            n_entries += 4
        if len_face == 4:
            n_entries += 5
            # n_entries += 8

    poly_faces = np.empty(n_entries, dtype=np.int32)
    points = {}

    counter = 0
    face_len = 0
    idx = 0

    for _i, face in enumerate(faces):
        face_len = len(face)
        new_face = np.zeros(face_len)
        for i, vertex in enumerate(face):
            # if face_len != 3:
            #     raise NotImplementedError("Only triangular faces are supported")
            vertex = tuple(vertex)
            if vertex not in points.keys():
                points[vertex] = counter
                cat_points.append(vertex)
                counter += 1

            new_face[i] = points[vertex]

        # For a face with 4 points, we create 2 triangles,
        # Because pyvista does not support quads correctly, while it says it does.
        # The issue is that when you supply a quad, it will create 2 triangles,
        # but the triangles will overlap by half, like an open envelope shape.
        if face_len == 3:
            poly_faces[idx] = 3
            poly_faces[idx + 1] = new_face[0]
            poly_faces[idx + 2] = new_face[1]
            poly_faces[idx + 3] = new_face[2]
            idx += 4

        elif face_len == 4:  # make a quadrillateral (NOTE: Skipped by raise)
            poly_faces[idx] = 4
            poly_faces[idx + 1] = new_face[0]
            poly_faces[idx + 2] = new_face[1]
            poly_faces[idx + 3] = new_face[2]
            poly_faces[idx + 4] = new_face[3]
            idx += 5

    return cat_points, poly_faces

def polydata_from_cat_cell(cat_cell)-> pv.PolyData:
    return pv.PolyData(*convert_faces_to_polydata_input(cat_cell))

def cat_meshes_from_cells(cat_cells):
    return [polydata_from_cat_cell(cat_cell) for cat_cell in cat_cells]

