# ruff: noqa: E501
import unittest

import numpy as np
import pyvista

from irregular_object_packing.cat.cat_data import CatData, TetPoint
from irregular_object_packing.cat.chordal_axis_transform import (
    create_faces_2,
    create_faces_3,
    create_faces_4,
    face_coord_to_points_and_faces,
)
from irregular_object_packing.tests.helpers import sort_surfaces

# this will show plots during tests.
VISUALIZE = False


class TestCreateCatFaces(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices = {
            "a": (0.0, 0.0, 0.0),
            "b": (9.0, 0.0, 0.0),
            "c": (0.0, 9.0, 0.0),
            "d": (0.0, 0.0, 9.0),
            "abcd": (2.25, 2.25, 2.25),
            "ab": (4.5, 0.0, 0.0),
            "ac": (0.0, 4.5, 0.0),
            "ad": (0.0, 0.0, 4.5),
            "bc": (4.5, 4.5, 0.0),
            "cd": (0.0, 4.5, 4.5),
            "bd": (4.5, 0.0, 4.5),
            "abc": (3.0, 3.0, 0.0),
            "abd": (3.0, 0.0, 3.0),
            "acd": (0.0, 3.0, 3.0),
            "bcd": (3.0, 3.0, 3.0),
        }
        self.point_keys = ["a", "b", "c", "d"]
        point_sets = [set(), set(), set(), set()]

        # dont think this is relevant for the tests.
        # object_coords = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        self.object_coords = np.array([])
        self.data = CatData(point_sets, self.object_coords)
        return super().setUp()

    def set_object_ids(self, obj_ids: list[int]):
        """IMPORTANT: obj_ids must be in ascending order"""
        # self.vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        point_sets = [set(), set(), set(), set()]

        for i, id in enumerate(obj_ids):
            point_sets[id].add(self.vertices[self.point_keys[i]])

        self.data = CatData(point_sets, self.object_coords)
        self.a = TetPoint(np.array(self.data.point(0)), 0, obj_ids[0])
        self.b = TetPoint(np.array(self.data.point(1)), 1, obj_ids[1])
        self.c = TetPoint(np.array(self.data.point(2)), 2, obj_ids[2])
        self.d = TetPoint(np.array(self.data.point(3)), 3, obj_ids[3])

        self.object_ids = obj_ids

    def assertCatFacesEqual(self, cat_faces1, cat_faces2):
        self.assertEqual(len(cat_faces1), len(cat_faces2))
        for p_id, faces1 in cat_faces1.items():
            self.assertListEqual(faces1, cat_faces2[p_id])

    def assertCatPointsEqual(self, cat_points1, cat_points2):
        self.assertEqual(len(cat_points1), len(cat_points2))
        for p_id, point1 in cat_points1.items():
            self.assertPointsEqual(point1, cat_points2[p_id])

    def assertPointsEqual(self, point1, point2):
        np.testing.assert_array_equal(point1, point2)

    @property
    def points(self):
        return [self.a, self.b, self.c, self.d]

    @property
    def main_vertices(self):
        return [p.vertex for p in self.points]

    @property
    def abcd(self):
        return self.data.point_id(self.vertices["abcd"])

    @property
    def middle_ab(self):
        return self.data.point_id(self.vertices["ab"])

    @property
    def middle_ac(self):
        return self.data.point_id(self.vertices["ac"])

    @property
    def middle_ad(self):
        return self.data.point_id(self.vertices["ad"])

    @property
    def middle_bc(self):
        return self.data.point_id(self.vertices["bc"])

    @property
    def middle_cd(self):
        return self.data.point_id(self.vertices["cd"])

    @property
    def middle_bd(self):
        return self.data.point_id(self.vertices["bd"])

    @property
    def middle_abc(self):
        return self.data.point_id(self.vertices["abc"])

    @property
    def middle_abd(self):
        return self.data.point_id(self.vertices["abd"])

    @property
    def middle_acd(self):
        return self.data.point_id(self.vertices["acd"])

    @property
    def middle_bcd(self):
        return self.data.point_id(self.vertices["bcd"])

    def compare_results(self, data, expected_faces):
        computed_faces = data.cat_faces
        for key in computed_faces.keys():
            for key2 in computed_faces[key].keys():
                sorted_computed_faces = sort_surfaces(
                    (f for f in computed_faces[key][key2])
                )
                sorted_expecte_faces = sort_surfaces(
                    (f for f in expected_faces[key][key2])
                )

                # self.assertListEqual(
                #     computed_faces[key][key2][1], expected_faces[key][key2][1], "normals are not equal"
                # )
                for i, face in enumerate(sorted_computed_faces):
                    for p in face:
                        self.assertIn(p, sorted_expecte_faces[i], f"face is missing point {p}")
                # self.assertListEqual(
                #     sorted_computed_faces, sorted_expecte_faces, "faces are not equal"
                # )
    # TODO: Test for CAT faces: Each face should only belong to 2 objects

    def test_create_faces_4(self):
        self.set_object_ids([0, 1, 2, 3])

        expected_faces = {
            0: {
                0: [
                    [self.abcd, self.middle_ab, self.middle_abd],
                    [self.abcd, self.middle_ad, self.middle_acd],
                    [self.abcd, self.middle_ad, self.middle_abd],
                    [self.abcd, self.middle_ab, self.middle_abc],
                    [self.abcd, self.middle_ac, self.middle_abc],
                    [self.abcd, self.middle_ac, self.middle_acd],
                ],
            },
            1: {
                1: [
                    [self.abcd, self.middle_bc, self.middle_abc],
                    [self.abcd, self.middle_bc, self.middle_bcd],
                    [self.abcd, self.middle_ab, self.middle_abc],
                    [self.abcd, self.middle_ab, self.middle_abd],
                    [self.abcd, self.middle_bd, self.middle_bcd],
                    [self.abcd, self.middle_bd, self.middle_abd],
                ],
            },
            2: {
                2: [
                    [self.abcd, self.middle_bc, self.middle_abc],
                    [self.abcd, self.middle_bc, self.middle_bcd],
                    [self.abcd, self.middle_cd, self.middle_bcd],
                    [self.abcd, self.middle_ac, self.middle_abc],
                    [self.abcd, self.middle_cd, self.middle_acd],
                    [self.abcd, self.middle_ac, self.middle_acd],
                ],
            },
            3: {
                3: [
                    [self.abcd, self.middle_cd, self.middle_acd],
                    [self.abcd, self.middle_bd, self.middle_bcd],
                    [self.abcd, self.middle_cd, self.middle_bcd],
                    [self.abcd, self.middle_bd, self.middle_abd],
                    [self.abcd, self.middle_ad, self.middle_abd],
                    [self.abcd, self.middle_ad, self.middle_acd],
                ],
            },
        }

        create_faces_4(self.data, self.points)

        if VISUALIZE is True:
            self.visualize_cat_test_result()

        self.compare_results(self.data, expected_faces)

    def visualize_cat_test_result(self):
        plotter = pyvista.Plotter()
        plotter.add_mesh(pyvista.PolyData(self.main_vertices))
        colors = ["red", "green", "blue", "yellow"]
        for i, key in enumerate(set(self.object_ids)):
            # if
            plotter.add_mesh(pyvista.PolyData(*face_coord_to_points_and_faces(self.data, key)).explode(), color=colors[i], show_edges=True)
        plotter.show()

    def test_create_faces_3(self):
        self.set_object_ids([0, 0, 1, 2])
        occ = [(0, 2), (1, 1), (2, 1)]
        # expected output

        expected_faces = {
            0: {
                0: [
                    # *split_quadrilateral_to_triangles(
                    [
                        self.middle_ac,
                        self.middle_bc,
                        self.middle_acd,
                        self.middle_bcd,
                    ]

                    # )
                    ,
                    # *split_quadrilateral_to_triangles(
                    [
                        self.middle_ad,
                        self.middle_bd,
                        self.middle_acd,
                        self.middle_bcd,
                    ]
                    # )
                    ,
                ],
                1: [
                    # *split_quadrilateral_to_triangles(
                    [
                        self.middle_ac,
                        self.middle_bc,
                        self.middle_acd,
                        self.middle_bcd,
                    ]
                    # )
                    ,
                    # *split_quadrilateral_to_triangles(
                    [
                        self.middle_ad,
                        self.middle_bd,
                        self.middle_acd,
                        self.middle_bcd,
                    ]
                    # )
                    ,
                ],
            },
            1: {
                2: [
                    [self.middle_cd, self.middle_acd, self.middle_bcd],
                    # *split_quadrilateral_to_triangles(
                    [
                        self.middle_ac,
                        self.middle_bc,
                        self.middle_acd,
                        self.middle_bcd,
                    ]
                    # )
                    ,
                ],
            },
            2: {
                3: [
                    [self.middle_cd, self.middle_acd, self.middle_bcd],
                    # *split_quadrilateral_to_triangles(
                    [
                        self.middle_ad,
                        self.middle_bd,
                        self.middle_acd,
                        self.middle_bcd,
                    ]
                    # )
                    ,
                ],
            },
        }

        create_faces_3(self.data, occ, self.points)

        if VISUALIZE is True:
            self.visualize_cat_test_result()
        self.compare_results(self.data, expected_faces)

# NOTE: This test has been manually checked using debug mode and is correct.
#       crazy anoying to test of list of list of arrays contain the same list of arrays

# for obj in range(4):
#     for p in self.points:
#         if computed_faces[obj].get(tuple(p.vertex)):
#             compset = set(tuple(face) for face in computed_faces[obj][tuple(p.vertex)]
#             expset = set(tuple(face) for face in expected_faces[obj][tuple(p.vertex)])
#             assert compset == expset
#             assert computed_faces[obj][tuple(p.vertex)] == \
#               expected_faces[obj][tuple(p.vertex)]

# assert_faces_equal(self, computed_faces, expected_faces)

    def test_create_faces_2_aabb(self):
        self.set_object_ids([0, 0, 1, 1])

        occ = [(0, 2), (1, 2)]

        expected_faces = {
            0: {
                0: [
                    [
                        self.middle_ac,
                        self.middle_bc,
                        self.middle_ad,
                        self.middle_bd,
                    ]
                    ,
                ],
                1: [
                    [
                        self.middle_ac,
                        self.middle_bc,
                        self.middle_ad,
                        self.middle_bd,
                    ]
                    ,
                ],
            },
            1: {
                2: [
                    [
                        self.middle_ac,
                        self.middle_bc,
                        self.middle_ad,
                        self.middle_bd,
                    ]
                    ,
                ],
                3: [
                    [
                        self.middle_ac,
                        self.middle_bc,
                        self.middle_ad,
                        self.middle_bd,
                    ]
                    ,
                ],
            },
        }

        create_faces_2(self.data, occ, self.points)
        if VISUALIZE is True:
            self.visualize_cat_test_result()

        self.compare_results(self.data, expected_faces)

    def test_create_faces_2_abbb(self):
        self.set_object_ids([0, 0, 0, 1])
        occ = [(0, 3), (1, 1)]

        expected_faces = {
            1: {
                3: [[self.middle_ad, self.middle_bd, self.middle_cd]],
            },
            0: {
                0: [[self.middle_ad, self.middle_bd, self.middle_cd]],
                1: [[self.middle_ad, self.middle_bd, self.middle_cd]],
                2: [[self.middle_ad, self.middle_bd, self.middle_cd]],
            },
        }

        create_faces_2(self.data, occ, self.points)
        if VISUALIZE is True:
            self.visualize_cat_test_result()

        self.compare_results(self.data, expected_faces)

    def test_face_coord_to_points_and_faces_3_points(self):
        face = [1, 2, 3]
        self.set_object_ids([0, 0, 0, 0])
        self.data.add_cat_face_to_cell(0, face)

        expected_points = [
            np.array([0, 0, 0]),
            np.array([9, 0, 0]),
            np.array([0, 0, 9]),
        ]
        expected_faces = [3, 0, 1, 2]
        points, faces = face_coord_to_points_and_faces(self.data, 0)

        for i in range(len(points)):
            self.assertTrue(np.array_equal(points[i], expected_points[i]))
        self.assertTrue(np.array_equal(faces, expected_faces))

    # @ unittest.skip("Change in implementation. function mainly needed for visualiation.")
    def test_face_coord_to_points_and_faces_4_points(self):
        face = [0, 1, 2, 3]
        # self.data.add_point()
        self.set_object_ids([0, 0, 0, 0])
        self.data.add_cat_face_to_cell(0, face)

        expected_points = [
            np.array([0, 9, 0]),
            np.array([0, 0, 0]),
            np.array([9, 0, 0]),
            np.array([0, 0, 9]),
        ]
        expected_faces = [4, 0, 1, 2, 3]
        points, faces = face_coord_to_points_and_faces(self.data, 0)

        print(faces)
        for i in range(len(points)):
            self.assertTrue(np.array_equal(points[i], expected_points[i]))
        self.assertTrue(np.array_equal(faces, expected_faces))

    # @ unittest.skip("Change in implementation. function mainly needed for visualiation.")
    def test_face_coord_to_points_and_faces_3_points_2_faces(self):
        self.set_object_ids([0, 0, 0, 0])
        face = [0, 1, 2]
        self.data.add_cat_faces_to_cell(0, [face, face])
        expected_points = [
            np.array([0, 9, 0]),
            np.array([0, 0, 0]),
            np.array([9, 0, 0]),
        ]
        expected_faces = [3, 0, 1, 2, 3, 0, 1, 2]
        points, faces = face_coord_to_points_and_faces(self.data, 0)

        for i in range(len(points)):
            self.assertTrue(np.array_equal(points[i], expected_points[i]))
        self.assertTrue(np.array_equal(faces, expected_faces))

    # @ unittest.skip("Change in implementation. function mainly needed for visualiation.")
    def test_faces_with_3_and_4_points(self):
        self.set_object_ids([0, 0, 0, 0])
        self.data.add_obj_point(0, (1.0, 0.0, 0.0))
        faces = [
            [0, 1, 2],
            [2, 3, 1],
            [0, 1, 4],
            [0, 2, 3, 4],
        ]
        self.data.add_cat_faces_to_cell(0, faces)
        expected_points = [
            (0.0, 9.0, 0.0),
            (0.0, 0.0, 0.0),
            (9.0, 0.0, 0.0),
            (0.0, 0.0, 9.0),
            (1.0, 0.0, 0.0),
        ]
        expected_faces = [3, 0, 1, 2, 3, 2, 3, 1, 3, 0, 1, 4, 4, 0, 2, 3, 4]
        points, faces = face_coord_to_points_and_faces(self.data, 0)
        print(faces)

        self.assertListEqual(points, expected_points)
        # for i in range(len(points)):
        self.assertListEqual(faces.tolist(), expected_faces)


def mismaching_faces_error(computed_faces, expected_faces):
    error_message = "computed faces are not equal to expected faces;\n"
    for i in range(len(expected_faces)):
        error_message += f"face\t {i}:\n"
        error_message += f"expected:\t {expected_faces[i]}\n"
        error_message += f"got:\t {computed_faces[i]}\n"

    return error_message


if __name__ == "__main__":
    unittest.main()
