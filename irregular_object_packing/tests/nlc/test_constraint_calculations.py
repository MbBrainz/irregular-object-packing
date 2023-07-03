import unittest
from dataclasses import astuple, dataclass

import numpy as np
from parameterized import parameterized

from irregular_object_packing.packing.nlc_optimisation import (
    construct_transform_matrix_from_array,
    local_constraint_for_vertex,
    local_constraint_vertices,
)
from irregular_object_packing.tests.helpers import float_array


@dataclass
class Case:
    """A test case for the constraint calculations."""
    name: str
    vertex: np.ndarray
    fpoint: np.ndarray
    fnormal: np.ndarray
    tf_array: np.ndarray
    expected: float

    @property
    def tuple(self) -> tuple:
        return astuple(self)

    @property
    def list(self) -> list:
        return list(self.tuple)


CASES_WITHOUT_TRANSFORMATION = [
    Case(
        name="close_left_plus",
        vertex=np.array([0, 0, 0.9], dtype=np.float64),
        fpoint=np.array([0, 0, 1], dtype=np.float64),
        fnormal=np.array([0, 0, -1], dtype=np.float64),
        tf_array=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=0.1,
    ).list,
    Case(
        name="close_left_minus",
        vertex=np.array([0, 0, 0.9], dtype=np.float64),
        fpoint=np.array([0, 0, 1], dtype=np.float64),
        fnormal=np.array([0, 0, 1], dtype=np.float64),
        tf_array=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=-0.1,
    ).list,
    Case(
        name="close_right_plus",
        vertex=np.array([0, 0, 1.1], dtype=np.float64),
        fpoint=np.array([0, 0, 1], dtype=np.float64),
        fnormal=np.array([0, 0, 1], dtype=np.float64),
        tf_array=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=0.1,
    ).list,
    Case(
        name="close_right_minus",
        vertex=np.array([0, 0, 1.1], dtype=np.float64),
        fpoint=np.array([0, 0, 1], dtype=np.float64),
        fnormal=np.array([0, 0, -1], dtype=np.float64),
        tf_array=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=-0.1,
    ).list,
    Case(
        name="far_left_plus",
        vertex=np.array([0, 0, 0.9], dtype=np.float64),
        fpoint=np.array([0, 0, 2], dtype=np.float64),
        fnormal=np.array([0, 0, -1], dtype=np.float64),
        tf_array=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=1.1,
    ).list,
    Case(
        name="exact_left_plus",
        vertex=np.array([0, 0, 1], dtype=np.float64),
        fpoint=np.array([0, 0, 1], dtype=np.float64),
        fnormal=np.array([0, 0, -1], dtype=np.float64),
        tf_array=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=0,
    ).list,
    Case(
        name="exact_left_minus",
        vertex=np.array([0, 0, 1], dtype=np.float64),
        fpoint=np.array([0, 0, 1], dtype=np.float64),
        fnormal=np.array([0, 0, 1], dtype=np.float64),
        tf_array=np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=0,
    ).list
]

CASES_WITH_TRANSFORMATION = [
    Case(
        name="close_left_plus_half",
        vertex=np.array([0, 0, 0.9], dtype=np.float64),
        fpoint=np.array([0, 0, 1], dtype=np.float64),
        fnormal=np.array([0, 0, -1], dtype=np.float64),
        tf_array=np.array([0.5**3, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=0.55,
    ).list,
    Case(
        name="close_left_minus_half",
        vertex=np.array([0, 0, 0.9], dtype=np.float64),
        fpoint=np.array([0, 0, 1], dtype=np.float64),
        fnormal=np.array([0, 0, 1], dtype=np.float64),
        tf_array=np.array([0.5**3, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=-0.55,
    ).list,
    Case(
        name="close_right_plus_one&half",
        vertex=np.array([0, 0, 0.9], dtype=np.float64),
        fpoint=np.array([0, 0, 1], dtype=np.float64),
        fnormal=np.array([0, 0, 1], dtype=np.float64),
        tf_array=np.array([1.5**3, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=0.35,
    ).list,
    Case(
        name="close_right_minus_half",
        vertex=np.array([0, 0, 1.8], dtype=np.float64),
        fpoint=np.array([0, 0, 1], dtype=np.float64),
        fnormal=np.array([0, 0, -1], dtype=np.float64),
        tf_array=np.array([0.5**3, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=0.1,
    ).list,
    Case(
        name="far_left_plus",
        vertex=np.array([0, 0, 0.9], dtype=np.float64),
        fpoint=np.array([0, 0, 2], dtype=np.float64),
        fnormal=np.array([0, 0, -1], dtype=np.float64),
        tf_array=np.array([0.5**3, 0, 0, 0, 0, 0, 0], dtype=np.float64),
        expected=1.55,
    ).list,
]


class ConstraintCalculations(unittest.TestCase):
    @parameterized.expand(CASES_WITHOUT_TRANSFORMATION)
    def test_local_constraints_no_tf(self, name, vertex, fpoint, fnormal, tf_array, expected):
        T = construct_transform_matrix_from_array(tf_array)
        constraint = local_constraint_for_vertex(vertex, fpoint, fnormal, T)
        self.assertAlmostEqual(constraint, expected, places=7)

    @parameterized.expand(CASES_WITHOUT_TRANSFORMATION)
    def test_global_constraints_no_tf(self, name, vertex, fpoint, fnormal, tf_array, expected):
        """This test case should give the same results as the previous
        test case, even though we give a global coordinate, as there is no transformation of the vertex"""
        T = construct_transform_matrix_from_array(tf_array)
        constraint = local_constraint_for_vertex(vertex, fpoint, fnormal, T, obj_coord=float_array([1, 1, 1]))
        self.assertAlmostEqual(constraint, expected, places=7)

    @parameterized.expand(CASES_WITHOUT_TRANSFORMATION)
    def test_local_constraints_padding(self, name, vertex, fpoint, fnormal, tf_array, expected):
        T = construct_transform_matrix_from_array(tf_array)
        constraint = local_constraint_for_vertex(vertex, fpoint, fnormal, T, padding=0.1)
        self.assertAlmostEqual(constraint, expected - 0.1, places=7)

    @parameterized.expand(CASES_WITH_TRANSFORMATION)
    def test_global_constraints_scaled(self, name, vertex, fpoint, fnormal, tf_array, expected):
        T = construct_transform_matrix_from_array(tf_array)
        constraint = local_constraint_for_vertex(vertex, fpoint, fnormal, T)
        self.assertAlmostEqual(constraint, expected, places=7)


class MultipleConstraints(unittest.TestCase):
    def setUp(self) -> None:
        self.tf_array = float_array([1, 0, 0, 0, 0, 0, 0])
        return super().setUp()

    def test_zero_length(self):
        arr = float_array([])
        constraints = local_constraint_vertices(self.tf_array, arr, float_array([0, 0, 0]))
        self.assertEqual(constraints.shape, (0,))

    def test_one_length(self):
        arr = float_array([[[0, 0, 0.9],
                           [0, 0, 1],
                           [0, 0, 1]]])
        constraints = local_constraint_vertices(self.tf_array, arr, float_array([0, 0, 0]))
        self.assertEqual(constraints.shape, (1,))

    def test_two_length(self):
        arr = float_array([[[0, 0, 0.9],
                           [0, 0, 1],
                           [0, 0, 1]],
                          [[0, 0, 0.9],
                           [0, 0, 1],
                           [0, 0, 1]]])
        constraints = local_constraint_vertices(self.tf_array, arr, float_array([0, 0, 0]))
        self.assertEqual(constraints.shape, (2,))
        self.assertEqual(constraints[0], constraints[1])

class UpdateTransformArray(unittest.TestCase):
    def setUp(self) -> None:
        self.global_tf_array = float_array([1, 0, 0, 0, 2, 2, 2])
        return super().setUp()

    #From the optimisation input we have a vertex-face_point-face_normal combination, which is relevant to a global coordinate
    # The optimisation function internally converts this to the local system seee L:165,169
    # The normal of the face can be left alone, as it is independent of the coordinate system

    '''
    The original tranform array is [f,rx, ry, rz ,t_x,t_y,t_z]
    The new transform array is computed as if tf_array_original = [1,0,0,0,0,0,0]
    so tf_new = [f', rx', ry', rz', t_x', t_y', t_z']

    the new tf array should be [f' * f, rx' + rx, ry' + ry, rz' + rz, t_x' + t_x, t_y' + t_y, t_z' + t_z]


    '''
