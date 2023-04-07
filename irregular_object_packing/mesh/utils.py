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
