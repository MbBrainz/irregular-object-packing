from irregular_object_packing.packing.nlc_optimisation import *
import numpy as np
from scipy.optimize import minimize
import unittest


class TestNLCConstraintOptimisation(unittest.TestCase):
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

        # Define facets
        self.facets = [
            # (face, normal)
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

        T, opt_tf = self.compute_optimal_tf(x0, v, facets_sets, f_bounds, r_bounds, t_bounds)
        ## %%
        resulting_points = []
        for points in v:
            res_v = transform_v(self.points[points], T)
            resulting_points.append(res_v)
            test_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_already_at_boundary(self):
        x0 = np.array([1, 0, 0, 0, 0, 0, 0])
        v = [12, 13, 14]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, None)
        f_bounds = (0, None)  # scale bound

        T, opt_tf = self.compute_optimal_tf(x0, v, facets_sets, f_bounds, r_bounds, t_bounds)
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

        T, opt_tf = self.compute_optimal_tf(x0, v, facets_sets, f_bounds, r_bounds, t_bounds)

        self.assertListEqual(opt_tf[4:].tolist(), [0.0, 0.0, 0.0], "Translation should be 0")
        resulting_points = []
        for points in v:
            res_v = transform_v(self.points[points], T)
            resulting_points.append(res_v)
            test_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_rotation(self):
        x0 = np.array([0.9, 0.0, 0.0, 0.0, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (0, 0)
        t_bounds = (0, None)
        f_bounds = (0, None)  # scale bound

        T, opt_tf = self.compute_optimal_tf(x0, v, facets_sets, f_bounds, r_bounds, t_bounds)

        self.assertListEqual(opt_tf[1:4].tolist(), [0.0, 0.0, 0.0], "Rotation should be 0")
        resulting_points = []
        for points in v:
            res_v = transform_v(self.points[points], T)
            resulting_points.append(res_v)
            test_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_scale(self):
        x0 = np.array([1.0, 0.01, 0.01, 0.01, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, None)
        f_bounds = (1, 1)

        T, opt_tf = self.compute_optimal_tf(x0, v, facets_sets, f_bounds, r_bounds, t_bounds)
        self.assertEqual(opt_tf[0], 1.0, "Scale should be 1")
        resulting_points = []
        for points in v:
            res_v = transform_v(self.points[points], T)
            resulting_points.append(res_v)
            test_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_scale_no_rotation(self):
        x0 = np.array([1.0, 0.0, 0.0, 0.0, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (0, 0)
        t_bounds = (0, None)
        f_bounds = (1, 1)

        T, opt_tf = self.compute_optimal_tf(x0, v, facets_sets, f_bounds, r_bounds, t_bounds)
        self.assertEqual(opt_tf[0], 1.0, "Scale should be 1")
        self.assertListEqual(opt_tf[1:4].tolist(), [0.0, 0.0, 0.0], "Rotation should be 0")
        resulting_points = []
        for points in v:
            res_v = transform_v(self.points[points], T)
            resulting_points.append(res_v)
            test_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_scale_no_translation(self):
        x0 = np.array([1.0, 0.01, 0.01, 0.01, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (-1 / 12 * np.pi, 1 / 12 * np.pi)
        t_bounds = (0, 0)
        f_bounds = (1, 1)

        T, opt_tf = self.compute_optimal_tf(x0, v, facets_sets, f_bounds, r_bounds, t_bounds)
        self.assertEqual(opt_tf[0], 1.0, "Scale should be 1")
        self.assertListEqual(opt_tf[4:].tolist(), [0.0, 0.0, 0.0], "Translation should be 0")
        resulting_points = []
        for points in v:
            res_v = transform_v(self.points[points], T)
            resulting_points.append(res_v)
            test_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_translation_no_rotation(self):
        x0 = np.array([1.0, 0.0, 0.0, 0.0, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (0, 0)
        t_bounds = (0, 0)
        f_bounds = (0.5, None)

        T, opt_tf = self.compute_optimal_tf(x0, v, facets_sets, f_bounds, r_bounds, t_bounds)
        self.assertGreater(opt_tf[0], 0.5, "Scale should be greater than 0.5")
        self.assertListEqual(opt_tf[1:4].tolist(), [0.0, 0.0, 0.0], "Rotation should be 0")
        self.assertListEqual(opt_tf[4:].tolist(), [0.0, 0.0, 0.0], "Translation should be 0")
        resulting_points = []
        for points in v:
            res_v = transform_v(self.points[points], T)
            resulting_points.append(res_v)
            test_point_within_box(self, res_v, self.box_coords)

    def test_nlcp_no_scale_no_translation_no_rotation(self):
        x0 = np.array([1.0, 0.0, 0.0, 0.0, 0, 0, 0])
        v = [9, 10, 11]
        facets_sets = [self.facets, self.facets, self.facets]
        r_bounds = (0, 0)
        t_bounds = (0, 0)
        f_bounds = (1, 1)

        T, opt_tf = self.compute_optimal_tf(x0, v, facets_sets, f_bounds, r_bounds, t_bounds)
        self.assertEqual(opt_tf[0], 1.0, "Scale should be 1")
        self.assertListEqual(opt_tf[1:4].tolist(), [0.0, 0.0, 0.0], "Rotation should be 0")
        self.assertListEqual(opt_tf[4:].tolist(), [0.0, 0.0, 0.0], "Translation should be 0")
        resulting_points = []
        for points in v:
            res_v = transform_v(self.points[points], T)
            resulting_points.append(res_v)
            test_point_within_box(self, res_v, self.box_coords)

    # ----------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------
    def compute_optimal_tf(self, x0, v, facets_sets, f_bounds, r_bounds, t_bounds):
        bounds = [f_bounds, r_bounds, r_bounds, r_bounds, t_bounds, t_bounds, t_bounds]
        # constraint_dict = {"type": "ineq", "fun": constraint_multiple_points, "args": (v, [facets, facets, facets])}
        constraint_dict = {
            "type": "ineq",
            "fun": constraint_multiple_points,
            "args": (
                v,
                facets_sets,
                self.points,
                np.array([0, 0, 0]),
                None,
            ),
        }
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict)
        T = construct_transform_matrix(res.x)
        return T, res.x


def is_point_within_box(point, box_coords):
    min_coords = np.min(box_coords, axis=0)
    max_coords = np.max(box_coords, axis=0)
    return np.all(min_coords <= point) and np.all(point <= max_coords)


def test_point_within_box(test_case: unittest.TestCase, point, box_coords):
    is_inside = is_point_within_box(point, box_coords)
    test_case.assertTrue(is_inside, f"Point {point} is not inside the box defined by {box_coords}")


def get_face_coords(facet, points):
    return [points[p_id] for p_id in facet[0]]
