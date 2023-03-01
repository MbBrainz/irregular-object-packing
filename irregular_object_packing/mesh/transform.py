import numpy as np
import trimesh


def translation_matrix(x0, x1):
    return np.array([[1, 0, 0, x1[0] - x0[0]], [0, 1, 0, x1[1] - x0[1]], [0, 0, 1, x1[2] - x0[2]], [0, 0, 0, 1]])


def scale_to_volume(mesh: trimesh.Trimesh, target_volume):
    current_volume = mesh.volume
    scale_factor = (target_volume / current_volume) ** (1 / 3)
    # scaled_mesh = mesh.copy()

    # scaled_mesh.apply_scale(scale_factor)
    # return scaled_mesh
    return mesh.copy().apply_scale(scale_factor)


def scale_and_center_mesh(mesh: trimesh.Trimesh, target_volume):
    mesh = scale_to_volume(mesh, target_volume)
    M_t = translation_matrix(mesh.center_mass, [0, 0, 0])
    mesh.apply_transform(M_t)
    return mesh
