# ruff: noqa: E501
import unittest
from dataclasses import astuple, dataclass

import numpy as np
from parameterized import parameterized

from irregular_object_packing.packing.nlc_optimisation import (
    compute_optimal_transform,
    construct_transform_matrix,
    construct_transform_matrix_from_array,
    transform_v,
    update_transform_array,
)

RB = 1 / 12 * np.pi

np.random.seed(0)

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
        "no rotation",
        r_bounds=(0.0, 0.0),
    ).list,
    NLCTestParams(
        "no translation",
        t_bounds=(0, 0),
    ).list,
    NLCTestParams(
        "no scaling",
        f_bounds=(1, 1),
    ).list,
    NLCTestParams(
        "no rotation or translation",
        r_bounds=(0, 0),
        t_bounds=(0, 0),
    ).list,
    NLCTestParams(
        "no rotation or scaling",
        r_bounds=(0, 0.0),
        f_bounds=(1, 1),
    ).list,
    NLCTestParams(
        "no translation or scaling",
        t_bounds=(0.0, 0.0),
        f_bounds=(1, 1),
    ).list,
    NLCTestParams(
        "no rotation, translation or scaling",
        r_bounds=(0.0, 0.0),
        t_bounds=(0.0, 0.0),
        f_bounds=(1, 1),
    ).list,
    # NLCTestParams(
    #     "at limits",
    #     v=(12, 13, 14),
    #     expected_f=1.0,
    # ).list,
    NLCTestParams(
        "normal bounds with padding",
        t_bounds=(None, None),
        r_bounds=(-1/12 * np.pi, 1/12 * np.pi),
        f_bounds=(0, None),
        padding=0.1,
    ).list,
    NLCTestParams(
        "no rotation with padding",
        r_bounds=(0.0, 0.0),
        padding=0.1,
    ).list,
    NLCTestParams(
        "no translation with padding",
        t_bounds=(0, 0),
        padding=0.1,
    ).list,
    # NLCTestParams(
    #     "no scaling with padding",
    #     f_init=1.0,
    #     r_init=0.1,
    #     t_init=0.1,
    #     f_bounds=(1, 1),
    #     padding=0.1,
    # ).list,
    NLCTestParams(
        "no rotation or translation with padding",
        r_bounds=(0, 0),
        t_bounds=(0, 0),
        padding=0.1,
    ).list,
    NLCTestParams(
        "no rotation or scaling with padding",
        r_bounds=(0.0, 0.0),
        f_bounds=(1.0, 1.0),
        f_init=1.0,
        padding=0.1,
    ).list,
    NLCTestParams(
        "no translation or scaling with padding",
        t_bounds=(0.0, 0.0),
        f_bounds=(1, 1),
        padding=0.1,
    ).list,
    NLCTestParams(
        "no rotation, translation or scaling with padding",
        f_init=1.0,
        r_bounds=(0.0, 0.0),
        t_bounds=(0.0, 0.0),
        f_bounds=(1.0, 1.0),
        padding=0.1,
        expected_f=1.0,
    ).list,
    # NLCTestParams(
    #     "at limits with padding",
    #     v=(12, 13, 14),
    #     padding=0.1,
    #     expected_f=0.9,
    # ).list,
]

# normals are facing inwards
face_unit_normals = np.array([
            [ 0, 0, +1], # bottom
            [ 0, 0, -1], # top
            [+1, 0, 0],  # right
            [-1, 0, 0],  # left
            [ 0, +1, 0], # front
            [ 0, -1, 0], # back
        ], dtype=np.float64)

face_points = np.array([
        [0.0, 0.0, -1.0], # bottom
        [0.0, 0.0, 1.0], # top
        [-1.0, 0.0, 0.0], # right
        [1.0, 0.0, 0.0], # left
        [0, -1.0, 0.0], # front
        [0, 1.0, 0.0], # back
    ], dtype=np.float64)

def nlc_struct(vi, qj, nj):
    for vec in [vi, qj, nj]:
        assert vec.shape == (3,)

    nlc_array =  np.array([
        vi,
        qj,
        nj], dtype=np.float64)
    assert nlc_array.shape == (3, 3)

    return nlc_array


class TestNLCConstraintOptimisation(unittest.TestCase):
    def setUp(self) -> None:
        self.obj_coord = np.array([2, 2, 2], dtype=np.float64)
        self.local_points = {
            # Box of size 2x2x2 centered at the origin
            1: np.array([-1, -1, 1], dtype=np.float64),
            2: np.array([1, -1, 1], dtype=np.float64),
            3: np.array([1, 1, 1], dtype=np.float64),
            4: np.array([-1, 1, 1], dtype=np.float64),
            5: np.array([-1, -1, -1], dtype=np.float64),
            6: np.array([1, -1, -1], dtype=np.float64),
            7: np.array([1, 1, -1], dtype=np.float64),
            8: np.array([-1, 1, -1], dtype=np.float64),
            # points to test at the edges of the box but stil scalable
            9: np.array([0, 0, 0.9]),
            10: np.array([0, 0.9, 0.0]),
            11: np.array([0.9, 0.0, 0.0]),
            # points to test at the edges of the box but not scalable
            12: np.array([1, 1, 1]),
            13: np.array([1, 1, 0]),
            14: np.array([1, 0, 1]),
        }

        self.global_points = {
            k: v + self.obj_coord for k, v in self.local_points.items()
        }
        self.local_box_coords = [
            point for id, point in self.local_points.items() if id <= 8
        ]
        self.global_box_coords = [
            point for id, point in self.global_points.items() if id <= 8
        ]

        # Define facets [(face, normal)]
        self.faces = [
            np.array([1, 2, 3, 4]) ,
            np.array([5, 6, 7, 8]) ,
            np.array([1, 2, 6, 5]) ,
            np.array([2, 3, 7, 6]) ,
            np.array([3, 4, 8, 7]) ,
            np.array([4, 1, 5, 8]) ,
        ]

        self.local_face_coordinates = [
            np.array([self.local_points[i] for i in face]) for face in self.faces
        ]

        self.global_face_coordinates = [
            np.array([self.global_points[i] for i in face]) for face in self.faces
        ]
        self.face_normals = [
            np.array([0, 0, -1]),
            np.array([0, 0, 1]),
            np.array([0, 1, 0]),
            np.array([-1, 0, 0]),
            np.array([0, -1, 0]),
            np.array([+1, 0, 0]),
        ]
        return super().setUp()

    def local_vertex_fpoint_normal_arr(self, points):
        local_vertex_fpoint_normal_arr = []
        for i in points:
            for j, face in enumerate(self.local_face_coordinates):
                assert np.shape(face) == (4, 3)
                vertex_fpoint_fnormal = np.array([
                    self.local_points[i],
                    face[0],
                    self.face_normals[j],
                ])
                local_vertex_fpoint_normal_arr.append(vertex_fpoint_fnormal)

        assert np.shape(local_vertex_fpoint_normal_arr)[1:] == (3, 3)
        return local_vertex_fpoint_normal_arr


    def global_vertex_fpoint_normal_arr(self, points):
        global_vertex_fpoint_normal_arr = []
        for i in points:
            for j, face in enumerate(self.global_face_coordinates):
                assert np.shape(face) == (4, 3)
                vertex_fpoint_fnormal = np.array([
                    self.global_points[i],
                    face[0],
                    self.face_normals[j],
                ])
                global_vertex_fpoint_normal_arr.append(vertex_fpoint_fnormal)
        return global_vertex_fpoint_normal_arr

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

        array = self.local_vertex_fpoint_normal_arr(v),

        res_tf = compute_optimal_transform(
            np.array([0, 0, 0]),
            array[0],
            padding=padding,
            max_scale=100.0,
            scale_bound=f_bounds,
            max_angle=r_bounds[1],
            max_t=t_bounds[1],
        )

        opt_tf = update_transform_array(x0, res_tf, 100)
        resulting_points = []
        T = construct_transform_matrix_from_array(opt_tf)
        for point in v:
            res_v = transform_v(self.local_points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.local_box_coords, self.local_points[point], padding=padding)

        if expected_f is not None:
            self.assertAlmostEqual(opt_tf[0], expected_f, places=4)

    @parameterized.expand(TEST_CASES)
    def test_global(
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

        vtx_fpoint_fnormal = self.global_vertex_fpoint_normal_arr(v),

        res_tf = compute_optimal_transform(
            np.array([0, 0, 0]),
            vtx_fpoint_fnormal[0],
            padding=padding,
            max_scale=np.inf,
            scale_bound=f_bounds,
            max_angle=r_bounds[1],
            max_t=t_bounds[1],
        )

        new_tf = update_transform_array(x0, res_tf, np.inf)

        T = construct_transform_matrix(new_tf[0], new_tf[1:4], new_tf[4:])
        T = construct_transform_matrix(res_tf[0], res_tf[1:4], res_tf[4:])

        resulting_points = []
        for point in v:
            res_v = transform_v(self.global_points[point], T)

            resulting_points.append(res_v)

        for res_v in resulting_points:
            assert_point_within_box(
                self, res_v, self.global_box_coords, self.global_points[point], tolerance=1e-7, padding=padding
            )

    # ----------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------


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
    test_case: unittest.TestCase, point, box_coords, o_point, tolerance=1e-7, padding=0.0
):
    is_inside = is_point_within_box(
        point, box_coords, tolerance=tolerance, padding=padding
    )
    test_case.assertTrue(
        is_inside,
        f"Point {point} is not inside the box defined by {box_coords} with padding {padding}. Was {o_point}",
    )


def get_face_coords(facet, points):
    return [points[p_id] for p_id in facet[0]]


if __name__ == "__main__":
    unittest.main()
