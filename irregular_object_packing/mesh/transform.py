import numpy as np
import pyvista as pv


def translation_matrix(x0, x1) -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, x1[0] - x0[0]],
            [0, 1, 0, x1[1] - x0[1]],
            [0, 0, 1, x1[2] - x0[2]],
            [0, 0, 0, 1],
        ]
    )


def scale_to_volume(mesh: pv.PolyData, target_volume) -> pv.PolyData:
    current_volume = mesh.volume
    scale_factor = (target_volume / current_volume) ** (1 / 3)

    return mesh.scale(scale_factor)


def scale_and_center_mesh(mesh: pv.PolyData, target_volume) -> pv.PolyData:
    mesh = scale_to_volume(mesh, target_volume)
    mesh.translate(-1 * np.array(mesh.center_of_mass()), inplace=True)
    return mesh
