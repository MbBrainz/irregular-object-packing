# ruff: noqa: E501
import unittest
from dataclasses import astuple, dataclass

import numpy as np
import tetgen
from parameterized import parameterized
from pyvista import PolyData

from irregular_object_packing.cat.chordal_axis_transform import (
    compute_cat_faces,
    face_coord_to_points_and_faces,
)
from irregular_object_packing.mesh.transform import scale_and_center_mesh
from irregular_object_packing.packing.nlc_optimisation import (
    compute_optimal_transform,
    construct_transform_matrix,
    construct_transform_matrix_from_array,
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
    NLCTestParams(
        "at limits",
        v=(12, 13, 14),
        expected_f=1.0,
    ).list,
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
    NLCTestParams(
        "no scaling with padding",
        f_init=1.0,
        r_init=0.1,
        t_init=0.1,
        f_bounds=(1, 1),
        padding=0.1,
    ).list,
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
    NLCTestParams(
        "at limits with padding",
        v=(12, 13, 14),
        padding=0.1,
        expected_f=0.9,
    ).list,
]


class TestNLCConstraintOptimisation(unittest.TestCase):
    def setUp(self) -> None:
        self.obj_coord = np.array([2, 2, 2], dtype=np.float64)
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
        return local_vertex_fpoint_normal_arr


    def global_vertex_fpoint_normal_arr(self):
        global_vertex_fpoint_normal_arr = []
        for i in range(self.global_points.__len__()):
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

        T, opt_tf = compute_optimal_transform(
            x0,
            np.array([0, 0, 0]),
            array,
            padding=padding,
            max_scale=3.0,
            scale_bound=f_bounds,
            max_angle=r_bounds[1],
            max_t=t_bounds[1],
        )
        resulting_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T)
            resulting_points.append(res_v)
            assert_point_within_box(self, res_v, self.local_box_coords, padding=padding)

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

        array = self.global_vertex_fpoint_normal_arr(v),

        T_local, new_tf = compute_optimal_transform(
            x0,
            self.obj_coord,
            array,
            padding=padding,
            max_scale=3.0,
            scale_bound=f_bounds,
            max_angle=r_bounds[1],
            max_t=t_bounds[1],
        )

        T = construct_transform_matrix(new_tf[0], new_tf[1:4], new_tf[4:])

        resulting_points = []
        for point in v:
            res_v = transform_v(self.local_points[point], T)

            resulting_points.append(res_v)
            assert_point_within_box(
                self, res_v, self.global_box_coords, tolerance=1e-7, padding=padding
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


class TestCatBoxOptimization(unittest.TestCase):
    def setUp(self):
        self.obj_points = np.array(
            [
                [2, 1, 1],
                [11, 3, 1],
                [10, 7, 1],
                [1, 5, 1],
                [2, 1, 3],
                [11, 3, 3],
                [10, 7, 3],
                [1, 5, 3],
            ], dtype=np.float64)
        self.init_center = np.mean(self.obj_points, axis=0)
        self.obj_faces = np.hstack(np.array([
            [4, 0, 1, 2, 3],
            [4, 4, 5, 6, 7],
            [4, 0, 1, 5, 4],
            [4, 1, 2, 6, 5],
            [4, 2, 3, 7, 6],
            [4, 3, 0, 4, 7],
        ]), )

        self.container_points = np.array([
            [0, 0, 0],
            [15, 0, 0],
            [15, 10, 0],
            [0, 10, 0],
            [0, 0, 4],
            [15, 0, 4],
            [15, 10, 4],
            [0, 10, 4],
        ], dtype=np.float64)
        self.container_center = np.mean(self.container_points, axis=0)
        self.container_faces = self.obj_faces.copy()
        self.obj = PolyData(self.obj_points, self.obj_faces)
        self.obj0 = scale_and_center_mesh(self.obj, self.obj.volume)
        self.container = PolyData(self.container_points, self.container_faces)

        self.tet_input = (self.container + self.obj).triangulate()

        tet = tetgen.TetGen(self.tet_input)
        tet.tetrahedralize(order=1, mindihedral=0, minratio=0, steinerleft=0, quality=False)

        self.cat_data = compute_cat_faces(
            tet.grid, [set(map(tuple, self.obj_points)), set(map(tuple, self.container_points))],
            self.init_center,
        )

        self.obj_cat_cell = face_coord_to_points_and_faces(self.cat_data, 0)
        self.previous_transform_array = np.array([1, 0, 0, 0] + list(self.init_center))

    def test_optimize_cat_box(self):
        new_tf_array = compute_optimal_transform(
            0,
            self.previous_transform_array,
            1,
            (0, None),
            1 / 12 * np.pi,
            None,
            0,
            cat_data=self.cat_data,
        )

        # transform the object
        new_obj = self.obj0.transform(construct_transform_matrix_from_array(new_tf_array), inplace=False)

        inside = new_obj.select_enclosed_points(self.container, tolerance=1e-8)
        pts = new_obj.extract_points(inside['SelectedPoints'].view(bool), adjacent_cells=False)

        self.assertEqual(new_obj.n_points, pts.n_points)
        self.assertEqual(new_obj.n_cells, pts.n_cells)
        self.assertListEqual(list(pts.points), list(new_obj.points))

        # the new center should be almost equal to the center of the container
        self.assertAlmostEqual(new_tf_array[4], self.obj_cat_cell[0])
        self.assertAlmostEqual(new_tf_array[5], self.obj_cat_cell[1])
        self.assertAlmostEqual(new_tf_array[6], self.obj_cat_cell[2])


if __name__ == "__main__":
    unittest.main()
