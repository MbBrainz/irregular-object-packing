import unittest

import pyvista as pv

from irregular_object_packing.mesh.collision import (
    compute_cat_violations,
    compute_collision,
    compute_container_violations,
    compute_object_collisions,
)


class TestComputeCollision(unittest.TestCase):
    def setUp(self):
        self.mesh1 = pv.Cube()  # Cube centered at (0, 0, 0)
        self.mesh2 = pv.Cube()  # Another cube centered at (0, 0, 0)

    def test_non_colliding_objects(self):
        self.mesh2.translate([2, 2, 2], inplace=True)  # Move the second cube so it does not collide with the first
        self.assertIsNone(compute_collision(self.mesh1, self.mesh2, False))

    def test_touching_objects(self):
        self.mesh2.translate([1, 1, 1], inplace=True)  # Move the second cube so it touches the first
        self.assertIsNotNone(compute_collision(self.mesh1, self.mesh2, False))

    def test_overlapping_objects(self):
        # The cubes are initially overlapping, no need to move them
        self.assertIsNotNone(compute_collision(self.mesh1, self.mesh2, False))

class TestComputeObjectCollisions(unittest.TestCase):
    def setUp(self):
        self.meshes = [pv.Cube(), pv.Cube(), pv.Cube()]
        self.meshes[1].translate([2, 2, 2], inplace=True)  # Move the second cube so it does not collide with the first
        self.meshes[2].translate([4, 4, 4], inplace=True)  # Move the third cube so it does not collide with the others

    def test_non_colliding_objects(self):
        self.assertEqual(len(compute_object_collisions(self.meshes, False)), 0)

    def test_touching_objects(self):
        self.meshes[1].translate([-1, -1, -1], inplace=True)  # Move the second cube so it touches the first
        self.assertEqual(len(compute_object_collisions(self.meshes, False)), 1)

    def test_overlapping_objects(self):
        self.meshes[1].translate([-1, -1, -1], inplace=True)  # Move the second cube so it overlaps with the first
        self.assertEqual(len(compute_object_collisions(self.meshes, False)), 1)

class TestComputeContainerViolations(unittest.TestCase):
    # Similar structure to TestComputeCollision, but with a list of PolyData objects and a container
    def setUp(self):
        self.meshes = [pv.Cube(), pv.Cube()]
        self.container = pv.Cube(x_length=6, y_length=6, z_length=6)
        self.meshes[1].translate([2, 2, 2], inplace=True)  # Move the second cube so it does not collide with the container

    def test_non_colliding_objects(self):
        self.assertEqual(len(compute_container_violations(self.meshes, self.container, False)), 0)

    def test_touching_objects(self):
        self.meshes[0].translate([-2.5, -2.5, -2.5], inplace=True)  # Move the second cube so it touches the container
        self.assertEqual(len(compute_container_violations(self.meshes, self.container, False)), 1)

    def test_overlapping_objects(self):
        self.meshes[0].translate([-3, -3, -3], inplace=True)  # Move the second cube so it overlaps with the container
        self.assertEqual(len(compute_container_violations(self.meshes, self.container, False)), 1)

    def test_all_overlapping_objects(self):
        self.meshes[0].translate([-3, -3, -3], inplace=True)  # Move the first cube so it overlaps with the container
        self.meshes[1].translate([0.5,0.5, 0.5], inplace=True)  # Move the second cube so it overlaps with the container
        self.assertEqual(len(compute_container_violations(self.meshes, self.container, False)), 2)

class TestComputeCatViolations(unittest.TestCase):
    def setUp(self):
        self.meshes = [pv.Cube(), pv.Cube()]
        self.cat_meshes = [pv.Cube(), pv.Cube()]
        self.cat_meshes[0].scale([1.0001, 1.0001, 1.0001], inplace=True)  # Scale the first cat_mesh so it does not collide with the first cube
        self.cat_meshes[1].scale([1.0001, 1.0001, 1.0001], inplace=True)  # Scale the second cat_mesh so it does not collide with the second cube
        self.meshes[1].translate([2, 2, 2], inplace=True)  # Move the second cube so it does not collide with the second cat_mesh
        self.cat_meshes[1].translate([2, 2, 2], inplace=True)  # Move the second cat_mesh so it does not collide with the second cube

    def test_non_colliding_objects(self):

       self.assertEqual(len(compute_cat_violations(self.meshes, self.cat_meshes, False)), 0)

    def test_touching_objects(self):
        self.meshes[0].scale([1.0001, 1.0001, 1.0001], inplace=True)  # Scale the first cube so it touches the first cat_mesh
        self.meshes[1].scale([1.0001, 1.0001, 1.0001], inplace=True)  # Scale the second cube so it touches the second cat_mesh
        self.assertEqual(len(compute_cat_violations(self.meshes, self.cat_meshes, False)), 2)

    def test_single_overlapping_object(self):
        self.meshes[0].scale([1.0001, 1.0001, 1.0001], inplace=True)  # Scale the cat_meshes so they do not collide with the cubes
        self.assertEqual(len(compute_cat_violations(self.meshes, self.cat_meshes, False)), 1)


if __name__ == '__main__':
    unittest.main()
