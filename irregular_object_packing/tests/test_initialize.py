import unittest
import numpy as np
from pyvista import PolyData
import pyvista as pv
from irregular_object_packing.packing.initialize import (
    random_coordinate_within_bounds,
    get_min_bounding_mesh,
    init_coordinates,
)


class TestFunctions(unittest.TestCase):
    def test_random_coordinate_within_bounds(self):
        bounding_box = np.array([[-1, -1, -1], [1, 1, 1]])
        N = 1000
        coords = random_coordinate_within_bounds(bounding_box, N)

        self.assertEqual(coords.shape, (N, 3))
        self.assertTrue(np.all(coords[0, :] >= bounding_box[0]))
        self.assertTrue(np.all(coords[1, :] >= bounding_box[1]))

    # def test_get_min_bounding_mesh(self):
    #     mesh = PolyData(np.random.rand(10, 3))
    #     bounding_mesh = get_min_bounding_mesh(mesh)

    #     self.assertIsInstance(bounding_mesh, PolyData)
    #     self.assertTrue(hasattr(bounding_mesh, "volume"))

    def test_init_coordinates(self):
        container = pv.Cylinder().extract_surface().smooth(10)
        mesh = PolyData(np.random.rand(5, 3))
        coverage_rate = 0.3
        f_init = 0.1

        coords, skipped = init_coordinates(container, mesh, coverage_rate, f_init)

        self.assertIsInstance(coords, np.ndarray)
        self.assertIsInstance(skipped, int)


if __name__ == "__main__":
    unittest.main()
