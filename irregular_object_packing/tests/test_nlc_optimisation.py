# ruff: noqa: E501
import unittest
from dataclasses import astuple, dataclass

import numpy as np
from parameterized import parameterized
from scipy.optimize import minimize

from irregular_object_packing.packing.nlc_optimisation import (
    construct_transform_matrix,
    local_constraint_multiple_points,
    objective,
    transform_v,
)

RB = 1 / 12 * np.pi


@dataclass
class NLCTestParams:
    name: str
    v: np.ndarray = (9, 10, 11)
    f_init: float = 0.9
    r_init: float = 0.0
    t_init: float = 0.0
    f_bounds: tuple[float, float] = (0.0, None)
    r_bounds: tuple[float, float] = (-1 / 12 * np.pi, 1 / 12 * np.pi)
    t_bounds: tuple[float, float] = (0.0, None)
    padding: float = 0.0
    expected_f: float = None

    def tuple(self):
        return astuple(self)

    @property
    def list(self):
        return list(astuple(self))


TEST_CASES = [
    NLCTestParams(
        "no bounds",
        t_bounds=(None, None),
        r_bounds=(None, None),
        f_bounds=(0, None),
    ).list,
    NLCTestParams(
        "no r",
        r_bounds=(0.0, 0.0),
    ).list,
    NLCTestParams(
        "no t",
        t_bounds=(0, 0),
    ).list,
    NLCTestParams(
        "no f",
        f_bounds=(1, 1),
    ).list,
    NLCTestParams(
        "no r or t",
        r_bounds=(0, 0),
        t_bounds=(0, 0),
    ).list,
    NLCTestParams(
        "no r or f",
        r_bounds=(0, 0.0),
        f_bounds=(1, 1),
    ).list,
    NLCTestParams(
        "no t or f",
        t_bounds=(0.0, 0.0),
        f_bounds=(1, 1),
    ).list,
    NLCTestParams(
        "no r,t or f",
        r_bounds=(0.0, 0.0),
        t_bounds=(0.0, 0.0),
        f_bounds=(1, 1),
    ).list,
    NLCTestParams(
        "at limits",
        v=(12, 13, 14),
        expected_f=1.0,
    ).list,
    NLCTestParams(
        "no bounds with p",
        t_bounds=(None, None),
        r_bounds=(None, None),
        f_bounds=(0, None),
        padding=0.1,
    ).list,
    NLCTestParams(
        "no r with p",
        r_bounds=(0.0, 0.0),
        padding=0.1,
    ).list,
    NLCTestParams(
        "no t with p",
        t_bounds=(0, 0),
        padding=0.1,
    ).list,
    NLCTestParams(
        "no f with p",
        f_init=1.0,
        r_init=0.1,
        t_init=0.1,
        f_bounds=(1, 1),
        padding=0.1,
    ).list,
    NLCTestParams(
        "no r or t with p",
        r_bounds=(0, 0),
        t_bounds=(0, 0),
        padding=0.1,
    ).list,
    NLCTestParams(
        "no r or f with p",
        r_bounds=(0.0, 0.0),
        f_bounds=(1.0, 1.0),
        f_init=1.0,
        padding=0.1,
    ).list,
    NLCTestParams(
        "no t or f with p",
        t_bounds=(0.0, 0.0),
        f_bounds=(1, 1),
        padding=0.1,
    ).list,
    NLCTestParams(
        "no r,t or f with p",
        f_init=1.0,
        r_bounds=(0.0, 0.0),
        t_bounds=(0.0, 0.0),
        f_bounds=(1.0, 1.0),
        padding=0.1,
        expected_f=1.0,
    ).list,
    NLCTestParams(
        "at limits with p",
        v=(12, 13, 14),
        padding=0.1,
        expected_f=0.9,
    ).list,
]


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
        self.faces = [
            (np.array([1, 2, 3, 4]), np.array([0, 0, -1])),
            (np.array([5, 6, 7, 8]), np.array([0, 0, 1])),
            (np.array([1, 2, 6, 5]), np.array([0, 1, 0])),
            (np.array([2, 3, 7, 6]), np.array([-1, 0, 0])),
            (np.array([3, 4, 8, 7]), np.array([0, -1, 0])),
            (np.array([4, 1, 5, 8]), np.array([+1, 0, 0])),
        ]
        self.face_sets = [self.faces, self.faces, self.faces]
        return super().setUp()

    @parameterized.expand(TEST_CASES)
    def test_local(
        self,
        name,
        v,
        f_init,
        r_init,
        t_init,
        f_bounds,
        r_bounds,
        t_bounds,
        padding,
        expected_f,
    ):
        x0 = [f_init] + [r_init] * 3 + [t_init] * 3

        T, opt_tf = compute_optimal_tf(
            x0,
            v,
            self.face_sets,
            self.points,
            f_bounds,
            r_bounds,
            t_bounds,
            padding=padding,
        )
        resulting_points = []
        for point in v:
            res_v = transform_v(self.points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.box_coords, padding=padding)

        if expected_f is not None:
            self.assertAlmostEqual(opt_tf[0], expected_f, places=4)

    # ----------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------


def compute_optimal_tf(
    x0,
    v,
    sets_of_faces,
    points,
    f_bounds,
    r_bounds,
    t_bounds,
    obj_coords=np.array([0, 0, 0]),
    padding=0.0,
):
    bounds = [f_bounds, r_bounds, r_bounds, r_bounds, t_bounds, t_bounds, t_bounds]
    constraint_dict = {
        "type": "ineq",
        "fun": local_constraint_multiple_points,
        "args": (
            v,
            sets_of_faces,
            points,
            obj_coords,
            padding,
        ),
    }
    res = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict
    )
    T = construct_transform_matrix(res.x)
    return T, res.x


def is_point_within_box(point, box_coords, tolerance=1e-8, padding=0.0):
    min_coords = np.min(box_coords, axis=0) + np.array([padding, padding, padding])
    max_coords = np.max(box_coords, axis=0) - np.array([padding, padding, padding])
    tol = np.array([tolerance, tolerance, tolerance])
    return np.all(
        np.isclose(min_coords, point, rtol=0, atol=tolerance)
        | (min_coords - tol <= point)
    ) and np.all(
        np.isclose(point, max_coords, rtol=0, atol=tolerance)
        | (point <= max_coords + tol)
    )

    # return np.all(min_coords <= point) and np.all(point <= max_coords)


def assert_point_within_box(
    test_case: unittest.TestCase, point, box_coords, tolerance=1e-7, padding=0.0
):
    is_inside = is_point_within_box(
        point, box_coords, tolerance=tolerance, padding=padding
    )
    test_case.assertTrue(
        is_inside,
        f"Point {point} is not inside the box defined by {box_coords} with padding {padding}",
    )


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

        self.faces = [
            (np.array([1, 2, 3, 4]), np.array([0, 0, -1])),
            (np.array([5, 6, 7, 8]), np.array([0, 0, 1])),
            (np.array([1, 2, 6, 5]), np.array([0, 1, 0])),
            (np.array([2, 3, 7, 6]), np.array([-1, 0, 0])),
            (np.array([3, 4, 8, 7]), np.array([0, -1, 0])),
            (np.array([4, 1, 5, 8]), np.array([+1, 0, 0])),
        ]
        self.face_sets = [self.faces, self.faces, self.faces]
        self.local_box_coords = np.array([self.points[p_id] for p_id in range(1, 9)])
        self.box_coords = [coord + self.obj_coord for coord in self.local_box_coords]

        return super().setUp()

    @parameterized.expand(TEST_CASES)
    def test(
        self,
        name,
        v,
        f_init,
        r_init,
        t_init,
        f_bounds,
        r_bounds,
        t_bounds,
        padding,
        expected_f,
    ):
        x0 = [f_init] + [r_init] * 3 + [t_init] * 3

        T_local, opt_tf = compute_optimal_tf(
            x0,
            v,
            self.face_sets,
            self.points,
            f_bounds,
            r_bounds,
            t_bounds,
            padding=padding,
        )

        # Check if the local system is correct
        local_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T_local)
            local_points.append(res_v)
            assert_point_within_box(
                self, res_v, self.local_box_coords, tolerance=1e-7, padding=padding
            )

        if expected_f is not None:
            self.assertAlmostEqual(opt_tf[0], expected_f, places=5)

        # This part is based on Optimizer.local_optimisation()
        new_tf = opt_tf
        # new_tf[0] = opt_tf[0]
        T = construct_transform_matrix(new_tf)

        resulting_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T)

            resulting_points.append(res_v)
            assert_point_within_box(
                self, res_v, self.box_coords, tolerance=1e-7, padding=padding
            )
