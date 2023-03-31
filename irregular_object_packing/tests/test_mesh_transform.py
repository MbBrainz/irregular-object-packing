import unittest

import pyvista as pv

from irregular_object_packing.mesh.transform import *


class TestMeshScaling(unittest.TestCase):
    def test_scale_down_to_volume(self):
        mesh = pv.Sphere(radius=2)
        scaled_mesh = scale_to_volume(mesh, target_volume=1)
        self.assertAlmostEqual(scaled_mesh.volume, 1, places=5)

    def test_scale_down_and_center_mesh(self):
        mesh = pv.Sphere(radius=2, center=[1, 1, 1])
        scaled_mesh = scale_and_center_mesh(mesh, target_volume=1)
        self.assertAlmostEqual(scaled_mesh.volume, 1, places=5)
        self.assertTrue(
            np.array_equal(
                np.array(np.round(scaled_mesh.center, 6)), np.array([0, 0, 0])
            )
        )

    def test_scale_up_and_center_mesh(self):
        mesh = pv.Sphere(radius=2, center=[1, 1, 1])
        scaled_mesh = scale_and_center_mesh(mesh, target_volume=8)
        self.assertAlmostEqual(scaled_mesh.volume, 8, places=5)
        self.assertTrue(
            np.array_equal(
                np.round(np.array(scaled_mesh.center), 6), np.array([0, 0, 0])
            )
        )

    def test_scale_up_to_volume(self):
        mesh = pv.Sphere(radius=2)
        scaled_mesh = scale_to_volume(mesh, target_volume=8)
        self.assertAlmostEqual(scaled_mesh.volume, 8, places=5)
