import numpy as np
import pyvista as pv


def test_polydata_is_manifold():
    """I was not sure if the pv library is able to determine if a mesh is manifold,
    if you create it in the way like below. It seems to work."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, -1]])

    # mesh faces of a closed pyramid
    faces = np.hstack(
        [
            [4, 0, 1, 2, 3],  # square
            [3, 0, 1, 4],  # triangle
            [3, 1, 2, 4],  # triangle
            [3, 2, 3, 4],
            [3, 3, 0, 4],
        ]
    )

    surf = pv.PolyData(vertices, faces)
    assert surf.is_manifold
