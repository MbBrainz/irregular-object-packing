import unittest
from itertools import combinations

import numpy as np
import pyvista as pv

from irregular_object_packing.mesh.transform import (
    scale_and_center_mesh,
    scale_to_volume,
)
from irregular_object_packing.packing.initialize import (
    coord_is_correct,
    generate_initial_coordinates,
    get_max_radius,
    grid_initialisation,
    pyvista_to_trimesh,
    random_coordinate_within_bounds,
)


class TestInitialize(unittest.TestCase):
    def setUp(self) -> None:
        sphere = pv.Sphere(radius=1)
        box = pv.Box(bounds=[-1, 1, -1, 1, -1, 1])
        cylinder = pv.PolyData(pv.Cylinder(radius=1, height=2)).clean()

        self.containers = [sphere, box, cylinder]
        self.tri_containers = [pyvista_to_trimesh(c) for c in self.containers]
        self.shapes = [sphere, box, cylinder]

    def prepare_scale(self, m_vol, c_vol):
        scaled_cs = []
        for c in self.containers:
            c = scale_to_volume(c, c_vol)
            scaled_cs += [c]
        self.containers = scaled_cs

        scaled_shapes = []
        for c in self.shapes:
            c = scale_and_center_mesh(c, m_vol)
            scaled_shapes += [c]
        self.shapes = scaled_shapes

        return super().setUp()

    def test_random_coordinate_within_bounds(self):
        bounding_box = np.array([[-1, -1, -1], [1, 1, 1]])
        coords = random_coordinate_within_bounds(bounding_box)
        self.assertEqual(coords.shape, (3,))
        self.assertTrue(np.all(coords >= bounding_box[0]))
        self.assertTrue(np.all(coords <= bounding_box[1]))

    # def test_get_min_bounding_mesh(self):
    #     mesh = PolyData(np.random.rand(10, 3))
    #     bounding_mesh = get_min_bounding_mesh(mesh)

    #     self.assertIsInstance(bounding_mesh, PolyData)
    #     self.assertTrue(hasattr(bounding_mesh, "volume"))

    def test_init_coordinates_coords(self):
        coverage_rate = 0.3
        container_volume = 10
        mesh_volume = 0.1

        self.prepare_scale(mesh_volume, container_volume)
        f_init = 0.1
        # this computation only works for spheres
        min_distances = [
            get_max_radius(mesh) * 2 * f_init ** (1 / 3) for mesh in self.shapes
        ]

        coords, skipped = generate_initial_coordinates(
            self.containers[0], self.shapes[0], coverage_rate, f_init
        )
        self.assert_correct_coordinates(
            coords,
            coverage_rate,
            container_volume,
            mesh_volume,
            min_distances[0],
            container=self.containers[0],
            descr="sphere",
        )

        coords, _ = generate_initial_coordinates(
            self.containers[1], self.shapes[1], coverage_rate, f_init
        )
        self.assert_correct_coordinates(
            coords,
            coverage_rate,
            container_volume,
            mesh_volume,
            min_distances[1],
            container=self.containers[1],
            descr="box",
        )

        coords, _ = generate_initial_coordinates(
            self.containers[2], self.shapes[2], coverage_rate, f_init
        )
        self.assert_correct_coordinates(
            coords,
            coverage_rate,
            container_volume,
            mesh_volume,
            min_distances[2],
            container=self.containers[2],
            descr="cylinder",
        )

    def assert_correct_coordinates(
        self,
        coords,
        coverage_rate,
        container_volume,
        mesh_volume,
        min_distance,
        container,
        descr,
    ):
        # enough coordinates
        self.assertTrue(len(coords) * mesh_volume >= container_volume * coverage_rate)

        # all coordinates are inside the container
        is_inside_list = [np.linalg.norm(coord) <= 1 for coord in coords]
        is_inside_list = pv.PolyData(coords).select_enclosed_points(container)[
            "SelectedPoints"
        ]
        self.assertListEqual(
            is_inside_list.tolist(),
            [1] * len(coords),
            msg=f"not all coords are inside the container ({descr})",
        )

        # all coordinates are far enough apart
        not_too_close_list = [
            np.linalg.norm((c1 - c2)) >= min_distance
            for c1, c2 in combinations(coords, 2)
        ]
        self.assertTrue(
            all(not_too_close_list),
            msg=f"not all coords are far enough apart ({descr})",
        )

    def test_grid_initialisation(self):
        container_volume = 10
        mesh_volume = 0.1

        self.prepare_scale(mesh_volume, container_volume)
        for container, shape in zip(self.containers, self.shapes, strict=True):
            coords = grid_initialisation(container, shape, 0.3, 0.1)

            self.assertAlmostEqual(len(coords) * shape.volume, container.volume * 0.3, places=3)


    def test_get_max_radius(self):
        mesh = pv.Sphere(radius=1)
        max_radius = get_max_radius(mesh)
        self.assertAlmostEqual(max_radius, 1, msg="max radius should be 1")

        mesh = pv.Box(bounds=(-1, 1, -1, 1, -1, 1))
        max_radius = get_max_radius(mesh)
        self.assertAlmostEqual(
            max_radius, np.sqrt(3), msg="max radius should be sqrt(3)"
        )


class TestCoordIsCorrect(unittest.TestCase):
    def setUp(self):
        self.coord = np.array([0, 0, 0])
        self.min_distance = 1.0
        self.object_coords = [
            np.array([1.5, 0, 0]),
            np.array([0, 1.5, 0]),
            np.array([0, 0, 1.5]),
        ]

        # Create a container mesh (a simple box)
        self.c_box = pyvista_to_trimesh(pv.Box(bounds=(-2, 2, -2, 2, -2, 2)))
        self.c_sphere = pyvista_to_trimesh(pv.Sphere(radius=2))

    def test_coord_is_inside_and_valid(self):
        self.assertTrue(
            coord_is_correct(
                self.coord, self.c_box, self.object_coords, self.min_distance
            ),
            "coord should be in box",
        )
        self.assertTrue(
            coord_is_correct(
                self.coord, self.c_sphere, self.object_coords, self.min_distance
            ),
            "coord should be in sphere",
        )

    def test_coord_is_outside_and_invalid(self):
        outside_coord = np.array([3, 0, 0])
        self.assertFalse(
            coord_is_correct(
                outside_coord, self.c_box, self.object_coords, self.min_distance
            ),
            "coord should be outside box",
        )
        self.assertFalse(
            coord_is_correct(
                outside_coord, self.c_sphere, self.object_coords, self.min_distance
            ),
            "coord should be outside sphere",
        )

    def test_coord_is_inside_but_too_close_to_other_objects(self):
        close_object_coord = np.array([0.5, 0, 0])
        self.object_coords[0] = close_object_coord
        # self.assertFalse(coord_is_correct(self.coord, self.container, self.object_coords, self.min_distance))
        self.assertFalse(
            coord_is_correct(
                close_object_coord, self.c_box, self.object_coords, self.min_distance
            ),
            "coord should be too close to other object",
        )
        self.assertFalse(
            coord_is_correct(
                close_object_coord, self.c_sphere, self.object_coords, self.min_distance
            ),
            "coord should be too close to other object",
        )

    def test_coord_is_inside_but_too_close_to_container(self):
        close_container_coord = np.array([1.9, 0, 0])
        self.assertFalse(
            coord_is_correct(
                close_container_coord, self.c_box, self.object_coords, self.min_distance
            ),
            "coord should be too close to container",
        )
        self.assertFalse(
            coord_is_correct(
                close_container_coord,
                self.c_sphere,
                self.object_coords,
                self.min_distance,
            ),
            "coord should be too close to container",
        )


if __name__ == "__main__":
    unittest.main()
if __name__ == "__main__":
    unittest.main()
