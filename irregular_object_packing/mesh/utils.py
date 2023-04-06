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
    points = mesh.points
    faces = mesh.faces.reshape(mesh.n_faces, 4)[:, 1:]
    return Trimesh(vertices=points, faces=faces)
