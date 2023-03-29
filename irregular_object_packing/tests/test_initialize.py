from itertools import combinations
import unittest
import numpy as np
from pyvista import PolyData
import pyvista as pv
from irregular_object_packing.mesh.transform import scale_and_center_mesh, scale_to_volume
from irregular_object_packing.packing.initialize import (
    random_coordinate_within_bounds,
    get_min_bounding_mesh,
    init_coordinates,
)


class TestInitialize(unittest.TestCase):
    def test_random_coordinate_within_bounds(self):
        bounding_box = np.array([[-1, -1, -1], [1, 1, 1]])
        N = 1000
        coords = random_coordinate_within_bounds(bounding_box, N)

        self.assertEqual(coords.shape, (N, 3))

        # TODO: fix
        # self.assertTrue(np.all(coords[0, :] >= bounding_box[0]))
        # self.assertTrue(np.all(coords[1, :] >= bounding_box[1]))

    # def test_get_min_bounding_mesh(self):
    #     mesh = PolyData(np.random.rand(10, 3))
    #     bounding_mesh = get_min_bounding_mesh(mesh)

    #     self.assertIsInstance(bounding_mesh, PolyData)
    #     self.assertTrue(hasattr(bounding_mesh, "volume"))

    def test_init_coordinates(self):
        container = pv.Sphere(radius=1)
        assert container.is_manifold, "container not closed surface"
        mesh = pv.Sphere(radius=0.1)
        coverage_rate = 0.3
        container_volume = 1
        mesh_volume = 0.1
        # Scale the mesh and container to the desired volume
        container = scale_to_volume(container, container_volume)
        mesh = scale_and_center_mesh(mesh, mesh_volume)

        f_init = 0.1
        # this computation only works for spheres
        min_distance = (mesh_volume * f_init) ** (1 / 3)

        coords, skipped = init_coordinates(container, mesh, coverage_rate, f_init)

        is_inside_list = [np.linalg.norm(coord) <= 1 for coord in coords]
        self.assertListEqual(is_inside_list, [True] * len(coords))

        not_too_close_list = [np.linalg.norm((c1 - c2)) >= min_distance for c1, c2 in combinations(coords, 2)]
        self.assertTrue(all(not_too_close_list))

        self.assertTrue(len(coords) * mesh_volume >= container_volume * coverage_rate)
        self.assertIsInstance(skipped, int)


if __name__ == "__main__":
    unittest.main()
