import unittest

import numpy as np
from scipy.optimize import minimize

from irregular_object_packing.packing.nlc_optimisation import (
    construct_transform_matrix,
    local_constraint_multiple_points,
    objective,
    transform_v,
)


class TestNLCConstraintOptimisationLocal(unittest.TestCase):
    def setUp(self) -> None:
        self.points = {
            # Box of size 2x2x2 centered at the origin
            1: np.array([-1, -1, 1], dtype=np.float64),
            2: np.array([1, -1, 1], dtype=np.float64),
            3: np.array([1, 1, 1], dtype=np.float64),
            4: np.array([-1, 1, 1], dtype=np.float64),
            5: np.array([-1, -1, 0], dtype=np.float64),
            6: np.array([1, -1, 0], dtype=np.float64),
            7: np.array([1, 1, 0], dtype=np.float64),
            8: np.array([-1, 1, 0], dtype=np.float64),
            # points to test at the edges of the box but stil scalable
            9: np.array([0, 0, 1]),
            10: np.array([0, 1, 0.0]),
            11: np.array([1, 0.0, 0.0]),
            # points to test at the edges of the box but not scalable
            12: np.array([1, 1, 1]),
            13: np.array([1, 1, 0]),
            14: np.array([1, 0, 1]),
        }
        self.box_coords = [point for id, point in self.points.items() if id <= 8]

        # Define facets [(face, normal)]
        self.facets = [
            (np.array([1, 2, 3, 4]), np.array([0, 0, -1])),
            (np.array([5, 6, 7, 8]), np.array([0, 0, 1])),
            (np.array([1, 2, 6, 5]), np.array([0, 1, 0])),
            (np.array([2, 3, 7, 6]), np.array([-1, 0, 0])),
            (np.array([3, 4, 8, 7]), np.array([0, -1, 0])),
            (np.array([4, 1, 5, 8]), np.array([+1, 0, 0])),
        ]

        return super().setUp()

    def test_basic_nlcp(self):
        """This test case covers a general case where we have a box and 3 points that we allow to scale within the box"""
        x0 = np.array([0.9, 0.01, 0.01, 0.01, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, None)
        f_bounds = (0, None)  # scale bound

        T, opt_tf = compute_optimal_tf(x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds)
        ## %%
        resulting_points = []
        for point in v:
            res_v = transform_v(self.points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_already_at_boundary(self):
        x0 = np.array([1, 0, 0, 0, 0, 0, 0])
        v = [12, 13, 14]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, None)
        f_bounds = (0, None)  # scale bound

        T, opt_tf = compute_optimal_tf(x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds)
        self.assertListEqual(
            opt_tf.tolist(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "transformation should be identity"
        )

    # -----------------------------------------------------------
    # Test cases where we fix bound parameters ⬇︎
    # -----------------------------------------------------------
    def test_nlcp_no_translation(self):
        x0 = np.array([0.9, 0.01, 0.01, 0.01, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, 0)
        f_bounds = (0, None)  # scale bound

        T, opt_tf = compute_optimal_tf(x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds)

        self.assertListEqual(opt_tf[4:].tolist(), [0.0, 0.0, 0.0], "Translation should be 0")
        resulting_points = []
        for point in v:
            res_v = transform_v(self.points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_rotation(self):
        x0 = np.array([0.9, 0.0, 0.0, 0.0, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (0, 0)
        t_bounds = (0, None)
        f_bounds = (0, None)  # scale bound

        T, opt_tf = compute_optimal_tf(x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds)

        self.assertListEqual(opt_tf[1:4].tolist(), [0.0, 0.0, 0.0], "Rotation should be 0")
        resulting_points = []
        for point in v:
            res_v = transform_v(self.points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_scale(self):
        x0 = np.array([1.0, 0.01, 0.01, 0.01, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, None)
        f_bounds = (1, 1)

        T, opt_tf = compute_optimal_tf(x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds)
        self.assertEqual(opt_tf[0], 1.0, "Scale should be 1")
        resulting_points = []
        for point in v:
            res_v = transform_v(self.points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_scale_no_rotation(self):
        x0 = np.array([1.0, 0.0, 0.0, 0.0, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (0, 0)
        t_bounds = (0, None)
        f_bounds = (1, 1)

        T, opt_tf = compute_optimal_tf(x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds)
        self.assertEqual(opt_tf[0], 1.0, "Scale should be 1")
        self.assertListEqual(opt_tf[1:4].tolist(), [0.0, 0.0, 0.0], "Rotation should be 0")
        resulting_points = []
        for point in v:
            res_v = transform_v(self.points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_scale_no_translation(self):
        x0 = np.array([1.0, 0.01, 0.01, 0.01, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, 0)
        f_bounds = (1, 1)

        T, opt_tf = compute_optimal_tf(x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds)
        self.assertEqual(opt_tf[0], 1.0, "Scale should be 1")
        self.assertListEqual(opt_tf[4:].tolist(), [0.0, 0.0, 0.0], "Translation should be 0")
        resulting_points = []
        for point in v:
            res_v = transform_v(self.points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_translation_no_rotation(self):
        x0 = np.array([1.0, 0.0, 0.0, 0.0, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (0, 0)
        t_bounds = (0, 0)
        f_bounds = (0.5, None)

        T, opt_tf = compute_optimal_tf(x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds)
        self.assertGreater(opt_tf[0], 0.5, "Scale should be greater than 0.5")
        self.assertListEqual(opt_tf[1:4].tolist(), [0.0, 0.0, 0.0], "Rotation should be 0")
        self.assertListEqual(opt_tf[4:].tolist(), [0.0, 0.0, 0.0], "Translation should be 0")
        resulting_points = []
        for point in v:
            res_v = transform_v(self.points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_scale_no_translation_no_rotation(self):
        x0 = np.array([1.0, 0.0, 0.0, 0.0, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (0, 0)
        t_bounds = (0, 0)
        f_bounds = (1, 1)

        T, opt_tf = compute_optimal_tf(x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds)
        self.assertEqual(opt_tf[0], 1.0, "Scale should be 1")
        self.assertListEqual(opt_tf[1:4].tolist(), [0.0, 0.0, 0.0], "Rotation should be 0")
        self.assertListEqual(opt_tf[4:].tolist(), [0.0, 0.0, 0.0], "Translation should be 0")
        resulting_points = []
        for point in v:
            res_v = transform_v(self.points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords)

    # ----------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------


def compute_optimal_tf(x0, v, sets_of_faces, points, f_bounds, r_bounds, t_bounds, obj_coords=np.array([0, 0, 0])):
    bounds = [f_bounds, r_bounds, r_bounds, r_bounds, t_bounds, t_bounds, t_bounds]
    # constraint_dict = {"type": "ineq", "fun": constraint_multiple_points, "args": (v, [facets, facets, facets])}
    constraint_dict = {
        "type": "ineq",
        "fun": local_constraint_multiple_points,
        "args": (
            v,
            sets_of_faces,
            points,
            obj_coords,
            None,
        ),
    }
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict)
    T = construct_transform_matrix(res.x)
    return T, res.x


def is_point_within_box(point, box_coords, tolerance=1e-9):
    min_coords = np.min(box_coords, axis=0)
    max_coords = np.max(box_coords, axis=0)
    return np.all(np.isclose(min_coords, point, rtol=0, atol=tolerance) | (min_coords <= point)) and np.all(
        np.isclose(point, max_coords, rtol=0, atol=tolerance) | (point <= max_coords)
    )

    return np.all(min_coords <= point) and np.all(point <= max_coords)


def assert_point_within_box(test_case: unittest.TestCase, point, box_coords, tolerance=1e-9):
    is_inside = is_point_within_box(point, box_coords, tolerance=tolerance)
    test_case.assertTrue(is_inside, f"Point {point} is not inside the box defined by {box_coords}")


def get_face_coords(facet, points):
    return [points[p_id] for p_id in facet[0]]


class TestNLCConstraintOptimisationWithGlobal(unittest.TestCase):
    """This test case is meant to test the optimisation of the NLC constraint with a global coodinate system
    This means that we initialize a container away from the origin and use the object coordinates to transform the
    container to the local system, perform the optimisation, and then transform the container and the object back to the global system.
    """

    def setUp(self) -> None:
        self.obj_coord = np.array([2, 2, 2])
        self.local_points = {
            # Box of size 2x2x2 centered at the origin
            1: np.array([-1, -1, 1], dtype=np.float64),
            2: np.array([1, -1, 1], dtype=np.float64),
            3: np.array([1, 1, 1], dtype=np.float64),
            4: np.array([-1, 1, 1], dtype=np.float64),
            5: np.array([-1, -1, 0], dtype=np.float64),
            6: np.array([1, -1, 0], dtype=np.float64),
            7: np.array([1, 1, 0], dtype=np.float64),
            8: np.array([-1, 1, 0], dtype=np.float64),
            # points to test at the edges of the box but stil scalable
            9: np.array([0, 0, 1]),
            10: np.array([0, 1, 0.0]),
            11: np.array([1, 0.0, 0.0]),
            # points to test at the edges of the box but not scalable
            12: np.array([1, 1, 1]),
            13: np.array([1, 1, 0]),
            14: np.array([1, 0, 1]),
        }
        self.points = {k: v + self.obj_coord for k, v in self.local_points.items()}

        self.facets = [
            (np.array([1, 2, 3, 4]), np.array([0, 0, -1])),
            (np.array([5, 6, 7, 8]), np.array([0, 0, 1])),
            (np.array([1, 2, 6, 5]), np.array([0, 1, 0])),
            (np.array([2, 3, 7, 6]), np.array([-1, 0, 0])),
            (np.array([3, 4, 8, 7]), np.array([0, -1, 0])),
            (np.array([4, 1, 5, 8]), np.array([+1, 0, 0])),
        ]
        self.box_coords = np.array([self.points[p_id] for p_id in range(1, 9)])
        self.tf0 = np.array([1.0, 0.0, 0.0, 0.0, 2, 2, 2])
        self.local_box_coords = [coord - self.obj_coord for coord in self.box_coords]

        return super().setUp()

    def test_nlcp_basic(self):
        x0 = np.array([1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, None)
        f_bounds = (0, None)

        T_local, opt_tf = compute_optimal_tf(
            x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds, obj_coords=self.obj_coord
        )

        # Check if the local system is correct
        local_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T_local)
            local_points.append(res_v)
            assert_point_within_box(self, res_v, self.local_box_coords)

        # This part is based on Optimizer.local_optimisation()
        new_tf = self.tf0 + opt_tf
        new_tf[0] = opt_tf[0]
        T = construct_transform_matrix(new_tf)

        resulting_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T)

            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_not_scalable(self):
        x0 = np.array([1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        v = [12, 13, 14]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, None)
        f_bounds = (0, None)

        T_local, opt_tf = compute_optimal_tf(
            x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds, obj_coords=self.obj_coord
        )

        # Check if the local system is correct
        local_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T_local)
            local_points.append(res_v)
            assert_point_within_box(self, res_v, self.local_box_coords, tolerance=1e-8)

        # This part is based on Optimizer.local_optimisation()
        new_tf = self.tf0 + opt_tf
        new_tf[0] = opt_tf[0]
        T = construct_transform_matrix(new_tf)

        resulting_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T)

            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords, tolerance=1e-8)

    def test_nlcp_no_translation(self):
        x0 = np.array([1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, 0)
        f_bounds = (0, None)

        T_local, opt_tf = compute_optimal_tf(
            x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds, obj_coords=self.obj_coord
        )

        # Check if the local system is correct
        local_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T_local)
            local_points.append(res_v)
            assert_point_within_box(self, res_v, self.local_box_coords, tolerance=1e-8)

        # This part is based on Optimizer.local_optimisation()
        new_tf = self.tf0 + opt_tf
        new_tf[0] = opt_tf[0]
        T = construct_transform_matrix(new_tf)

        resulting_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T)

            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords, tolerance=1e-8)

    def test_nlcp_no_rotation(self):
        x0 = np.array([1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (0, 0)
        t_bounds = (0, None)
        f_bounds = (0, None)

        T_local, opt_tf = compute_optimal_tf(
            x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds, obj_coords=self.obj_coord
        )

        # Check if the local system is correct
        local_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T_local)
            local_points.append(res_v)
            assert_point_within_box(self, res_v, self.local_box_coords, tolerance=1e-8)

        # This part is based on Optimizer.local_optimisation()
        new_tf = self.tf0 + opt_tf
        new_tf[0] = opt_tf[0]
        T = construct_transform_matrix(new_tf)

        resulting_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T)

            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords, tolerance=1e-8)

    def test_nlcp_no_scaling(self):
        x0 = np.array([1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, None)
        f_bounds = (1, 1)

        T_local, opt_tf = compute_optimal_tf(
            x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds, obj_coords=self.obj_coord
        )

        # Check if the local system is correct
        local_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T_local)
            local_points.append(res_v)
            assert_point_within_box(self, res_v, self.local_box_coords, tolerance=1e-8)

        # This part is based on Optimizer.local_optimisation()
        new_tf = self.tf0 + opt_tf
        new_tf[0] = opt_tf[0]
        T = construct_transform_matrix(new_tf)

        resulting_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T)

            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords, tolerance=1e-8)

    def test_nlcp_no_scaling_no_translation(self):
        x0 = np.array([1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, 0)
        f_bounds = (1, 1)

        T_local, opt_tf = compute_optimal_tf(
            x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds, obj_coords=self.obj_coord
        )

        # Check if the local system is correct
        local_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T_local)
            local_points.append(res_v)
            assert_point_within_box(self, res_v, self.local_box_coords, tolerance=1e-8)

        # This part is based on Optimizer.local_optimisation()
        new_tf = self.tf0 + opt_tf
        new_tf[0] = opt_tf[0]
        T = construct_transform_matrix(new_tf)

        resulting_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T)

            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords, tolerance=1e-8)

    def test_nlcp_no_scaling_no_translation_no_rotation(self):
        x0 = np.array([1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (0, 0)
        t_bounds = (0, 0)
        f_bounds = (1, 1)

        T_local, opt_tf = compute_optimal_tf(
            x0, v, facets_sets, self.points, f_bounds, r_bounds, t_bounds, obj_coords=self.obj_coord
        )

        # Check if the local system is correct
        local_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T_local)
            local_points.append(res_v)
            assert_point_within_box(self, res_v, self.local_box_coords, tolerance=1e-8)

        # This part is based on Optimizer.local_optimisation()
        new_tf = self.tf0 + opt_tf
        new_tf[0] = opt_tf[0]
        T = construct_transform_matrix(new_tf)

        resulting_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T)

            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords, tolerance=1e-8)
