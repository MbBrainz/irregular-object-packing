import unittest

import pyvista as pv

from irregular_object_packing.mesh.sampling import (
    downsample_pv_mesh,
    resample_pyvista_mesh,
    upsample_pv_mesh,
)

# Include the provided functions here or import them from a separate module

class TestMeshResampling(unittest.TestCase):

    def setUp(self):
        self.sphere_low_res = pv.Sphere(radius=1, theta_resolution=10, phi_resolution=10)
        self.sphere_high_res = pv.Sphere(radius=1, theta_resolution=50, phi_resolution=50)

    def test_downsample_pv_mesh(self):
        target_faces = 100
        new_mesh = downsample_pv_mesh(self.sphere_high_res, target_faces)
        self.assertLessEqual(new_mesh.n_faces, target_faces)

    def test_downsample_pv_mesh_raises_error(self):
        target_faces = self.sphere_low_res.n_faces + 10
        self.assertRaises(ValueError, downsample_pv_mesh, self.sphere_low_res, target_faces)

    def test_upsample_pv_mesh(self):
        target_faces = 400
        new_mesh = upsample_pv_mesh(self.sphere_low_res, target_faces)
        self.assertGreaterEqual(new_mesh.n_faces, target_faces)

    def test_upsample_pv_mesh_raises_error(self):
        target_faces = self.sphere_high_res.n_faces - 10
        self.assertRaises(ValueError, upsample_pv_mesh, self.sphere_high_res, target_faces)

    def test_resample_pyvista_mesh_upsample(self):
        target_faces = 500
        new_mesh = resample_pyvista_mesh(self.sphere_low_res, target_faces)
        self.assertGreaterEqual(new_mesh.n_faces, target_faces)

    def test_resample_pyvista_mesh_downsample(self):
        target_faces = 150
        new_mesh = resample_pyvista_mesh(self.sphere_high_res, target_faces)
        self.assertLessEqual(new_mesh.n_faces, target_faces)

if __name__ == '__main__':
    unittest.main()
