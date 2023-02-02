import numpy as np
import trimesh


def print_mesh_info(mesh: trimesh.Trimesh, description="", suppress_scientific=True):
    with np.printoptions(precision=4, suppress=suppress_scientific):
        print(
            f"Mesh info {description}: {mesh}, \nvolume: {mesh.volume}, \nbounding box: {mesh.bounding_box.bounds} \ncenter of mass: {mesh.center_mass}\n"
        )
