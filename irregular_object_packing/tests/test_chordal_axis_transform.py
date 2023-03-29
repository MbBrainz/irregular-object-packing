import numpy as np
import unittest

from irregular_object_packing.packing.chordal_axis_transform import (
    TetPoint,
    CatData,
    create_faces_2,
    create_faces_3,
    create_faces_4,
    face_coord_to_points_and_faces,
)
from irregular_object_packing.packing.utils import (
    sort_faces_dict,
    compute_face_normal,
    split_quadrilateral_to_triangles,
)
from irregular_object_packing.tests.test_helpers import sort_points_in_dict, sort_surfaces


class TestCreateCatFaces(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices = {
            "a": (0, 0, 0),
            "b": (9, 0, 0),
            "c": (0, 9, 0),
            "d": (0, 0, 9),
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
                sorted_computed_faces = sort_surfaces(map(lambda f: f[0], computed_faces[key][key2]))
                sorted_expecte_faces = sort_surfaces(map(lambda f: f[0], expected_faces[key][key2]))

                # self.assertListEqual(
                #     computed_faces[key][key2][1], expected_faces[key][key2][1], "normals are not equal"
                # )
                self.assertListEqual(sorted_computed_faces, sorted_expecte_faces, "faces are not equal")

    def test_create_faces_4(self):
        self.set_object_ids([0, 1, 2, 3])

        expected_faces = {
            0: {
                0: [
                    (face, compute_face_normal(self.data.get_face(face), self.a.vertex))
                    for face in [
                        [self.abcd, self.middle_ab, self.middle_abd],
                        [self.abcd, self.middle_ad, self.middle_acd],
                        [self.abcd, self.middle_ad, self.middle_abd],
                        [self.abcd, self.middle_ab, self.middle_abc],
                        [self.abcd, self.middle_ac, self.middle_abc],
                        [self.abcd, self.middle_ac, self.middle_acd],
                    ]
                ],
            },
            1: {
                1: [  # from face b
                    (face, compute_face_normal(self.data.get_face(face), self.b.vertex))
                    for face in [
                        [self.abcd, self.middle_bc, self.middle_abc],
                        [self.abcd, self.middle_bc, self.middle_bcd],
                        [self.abcd, self.middle_ab, self.middle_abc],
                        [self.abcd, self.middle_ab, self.middle_abd],
                        [self.abcd, self.middle_bd, self.middle_bcd],
                        [self.abcd, self.middle_bd, self.middle_abd],
                    ]
                ],
            },
            2: {
                2: [  # from face c _____
                    (face, compute_face_normal(self.data.get_face(face), self.c.vertex))
                    for face in [
                        [self.abcd, self.middle_bc, self.middle_abc],
                        [self.abcd, self.middle_bc, self.middle_bcd],
                        [self.abcd, self.middle_cd, self.middle_bcd],
                        [self.abcd, self.middle_ac, self.middle_abc],
                        [self.abcd, self.middle_cd, self.middle_acd],
                        [self.abcd, self.middle_ac, self.middle_acd],
                    ]
                ],
            },
            3: {
                3: [  # from face d
                    (face, compute_face_normal(self.data.get_face(face), self.d.vertex))
                    for face in [
                        [self.abcd, self.middle_cd, self.middle_acd],
                        [self.abcd, self.middle_bd, self.middle_bcd],
                        [self.abcd, self.middle_cd, self.middle_bcd],
                        [self.abcd, self.middle_bd, self.middle_abd],
                        [self.abcd, self.middle_ad, self.middle_abd],
                        [self.abcd, self.middle_ad, self.middle_acd],
                    ]
                ],
            },
        }

        create_faces_4(self.data, self.points)

        self.compare_results(self.data, expected_faces)

    def test_create_faces_3(self):
        self.set_object_ids([0, 0, 1, 2])
        occ = [(0, 2), (1, 1), (2, 1)]
        # expected output

        expected_faces = {
            0: {
                0: [
                    (face, compute_face_normal(self.data.get_face(face), self.a.vertex))
                    for face in [
                        *split_quadrilateral_to_triangles(
                            [self.middle_ac, self.middle_bc, self.middle_acd, self.middle_bcd]
                        ),
                        *split_quadrilateral_to_triangles(
                            [self.middle_ad, self.middle_bd, self.middle_acd, self.middle_bcd]
                        ),
                    ]
                ],
                1: [
                    (face, compute_face_normal(self.data.get_face(face), self.b.vertex))
                    for face in [
                        *split_quadrilateral_to_triangles(
                            [self.middle_ac, self.middle_bc, self.middle_acd, self.middle_bcd]
                        ),
                        *split_quadrilateral_to_triangles(
                            [self.middle_ad, self.middle_bd, self.middle_acd, self.middle_bcd]
                        ),
                    ]
                ],
            },
            1: {
                2: [
                    (face, compute_face_normal(self.data.get_face(face), self.c.vertex))
                    for face in [
                        [self.middle_cd, self.middle_acd, self.middle_bcd],
                        *split_quadrilateral_to_triangles(
                            [self.middle_ac, self.middle_bc, self.middle_acd, self.middle_bcd]
                        ),
                    ]
                ],
            },
            2: {
                3: [
                    (face, compute_face_normal(self.data.get_face(face), self.d.vertex))
                    for face in [
                        [self.middle_cd, self.middle_acd, self.middle_bcd],
                        *split_quadrilateral_to_triangles(
                            [self.middle_ad, self.middle_bd, self.middle_acd, self.middle_bcd]
                        ),
                    ]
                ],
            },
        }

        create_faces_3(self.data, occ, self.points)

        self.compare_results(self.data, expected_faces)

        # NOTE: This test has been manually checked using debug mode and is correct.
        #       crazy anoying to test of list of list of arrays contain the same list of arrays

        # for obj in range(4):
        #     for p in self.points:
        #         if computed_faces[obj].get(tuple(p.vertex)):
        #             compset = set(tuple(face) for face in computed_faces[obj][tuple(p.vertex)])
        #             expset = set(tuple(face) for face in expected_faces[obj][tuple(p.vertex)])
        #             assert compset == expset
        #             # assert computed_faces[obj][tuple(p.vertex)] == expected_faces[obj][tuple(p.vertex)]

        # assert_faces_equal(self, computed_faces, expected_faces)

    def test_create_faces_2_aabb(self):
        self.set_object_ids([0, 0, 1, 1])

        occ = [(0, 2), (1, 2)]

        expected_faces = {
            0: {
                0: [
                    (face, compute_face_normal(self.data.get_face(face), self.a.vertex))
                    for face in [
                        *split_quadrilateral_to_triangles(
                            [self.middle_ac, self.middle_bc, self.middle_ad, self.middle_bd]
                        ),
                    ]
                ],
                1: [
                    (face, compute_face_normal(self.data.get_face(face), self.b.vertex))
                    for face in [
                        *split_quadrilateral_to_triangles(
                            [self.middle_ac, self.middle_bc, self.middle_ad, self.middle_bd]
                        ),
                    ]
                ],
            },
            1: {
                2: [
                    (face, compute_face_normal(self.data.get_face(face), self.c.vertex))
                    for face in [
                        *split_quadrilateral_to_triangles(
                            [self.middle_ac, self.middle_bc, self.middle_ad, self.middle_bd]
                        ),
                    ]
                ],
                3: [
                    (face, compute_face_normal(self.data.get_face(face), self.d.vertex))
                    for face in [
                        *split_quadrilateral_to_triangles(
                            [self.middle_ac, self.middle_bc, self.middle_ad, self.middle_bd]
                        ),
                    ]
                ],
            },
        }

        create_faces_2(self.data, occ, self.points)
        self.compare_results(self.data, expected_faces)

    def test_create_faces_2_abbb(self):
        self.set_object_ids([0, 0, 0, 1])
        occ = [(0, 3), (1, 1)]

        expected_faces = {
            1: {
                3: [
                    (face, compute_face_normal(self.data.get_face(face), self.d.vertex))
                    for face in [[self.middle_ad, self.middle_bd, self.middle_cd]]
                ],
            },
            0: {
                0: [
                    (face, compute_face_normal(self.data.get_face(face), self.d.vertex))
                    for face in [[self.middle_ad, self.middle_bd, self.middle_cd]]
                ],
                1: [
                    (face, compute_face_normal(self.data.get_face(face), self.d.vertex))
                    for face in [[self.middle_ad, self.middle_bd, self.middle_cd]]
                ],
                2: [
                    (face, compute_face_normal(self.data.get_face(face), self.d.vertex))
                    for face in [[self.middle_ad, self.middle_bd, self.middle_cd]]
                ],
            },
        }

        create_faces_2(self.data, occ, self.points)
        self.compare_results(self.data, expected_faces)

    def test_face_coord_to_points_and_faces_3_points(self):
        face = ([1, 2, 3], [1.0, 1.0, 1.0])
        self.set_object_ids([0, 0, 0, 0])
        self.data.add_cat_face_to_cell(0, face)

        expected_points = [np.array([0, 0, 0]), np.array([9, 0, 0]), np.array([0, 0, 9])]
        expected_faces = [3, 0, 1, 2]
        points, faces = face_coord_to_points_and_faces(self.data, 0)

        for i in range(len(points)):
            self.assertTrue(np.array_equal(points[i], expected_points[i]))
        self.assertTrue(np.array_equal(faces, expected_faces))

    @unittest.skip("Change in implementation. function mainly needed for visualiation.")
    def test_face_coord_to_points_and_faces_4_points(self):
        face = [0, 1, 2, 3]
        # self.data.add_point()
        self.set_object_ids([0, 0, 0, 0])
        self.data.add_cat_face_to_cell(0, face)

        expected_points = [np.array([0, 9, 0]), np.array([0, 0, 0]), np.array([9, 0, 0]), np.array([0, 0, 9])]
        expected_faces = [4, 0, 1, 2, 3]
        points, faces = face_coord_to_points_and_faces(self.data, 0)

        print(faces)
        for i in range(len(points)):
            self.assertTrue(np.array_equal(points[i], expected_points[i]))
        self.assertTrue(np.array_equal(faces, expected_faces))

    # def test_face_coord_to_points_and_faces_4_points_split(self):
    #     face = [0, 1, 2, 3]
    #     # self.data.add_point()
    #     self.set_object_ids([0, 0, 0, 0])
    #     self.data.add_cat_face_to_cell(0, face)

    #     expected_points = [np.array([0, 0, 0]), np.array([9, 0, 0]), np.array([9, 9, 0]), np.array([0, 9, 0])]
    #     expected_faces = [3, 0, 1, 2, 3, 2, 3, 1]
    #     points, faces = face_coord_to_points_and_faces(self.data, 0)

    #     print(faces)
    #     for i in range(len(points)):
    #         self.assertTrue(np.array_equal(points[i], expected_points[i]))
    #     self.assertTrue(np.array_equal(faces, expected_faces))

    @unittest.skip("Change in implementation. function mainly needed for visualiation.")
    def test_face_coord_to_points_and_faces_3_points_2_faces(self):
        self.set_object_ids([0, 0, 0, 0])
        face = ([0, 1, 2], [0.0, 0.0, 1.0])
        self.data.add_cat_faces_to_cell(0, [face, face])
        expected_points = [np.array([0, 9, 0]), np.array([0, 0, 0]), np.array([9, 0, 0])]
        expected_faces = [3, 0, 1, 2, 3, 0, 1, 2]
        points, faces = face_coord_to_points_and_faces(self.data, 0)

        for i in range(len(points)):
            self.assertTrue(np.array_equal(points[i], expected_points[i]))
        self.assertTrue(np.array_equal(faces, expected_faces))

    @unittest.skip("Change in implementation. function mainly needed for visualiation.")
    def test_faces_with_3_and_4_points(self):
        face_coords = [
            [np.array([1, -1, 1]), np.array([1, 1, 1]), np.array([-1, 1, 1]), np.array([0, 0, 2])],
            [np.array([1, -1, 1]), np.array([1, 1, 1]), np.array([-1, -1, 1])],
            [np.array([1, -1, -1]), np.array([1, 4, 1]), np.array([-1, 1, 1]), np.array([0, 0, 2])],
            [np.array([1, -1, -1]), np.array([1, 4, 1]), np.array([-1, -2, 1])],
        ]
        expected_points = [
            np.array([1, -1, 1]),
            np.array([1, 1, 1]),
            np.array([-1, 1, 1]),
            np.array([0, 0, 2]),
            np.array([-1, -1, 1]),
            np.array([1, -1, -1]),
            np.array([1, 4, 1]),
            np.array([-1, -2, 1]),
        ]
        expected_faces = np.array([3, 0, 1, 2, 3, 2, 3, 1, 3, 0, 1, 4, 3, 5, 6, 2, 3, 2, 3, 6, 3, 5, 6, 7])
        points, faces = face_coord_to_points_and_faces(self.data, 0)
        print(faces)

        for i in range(len(points)):
            self.assertTrue(np.array_equal(points[i], expected_points[i]))
        self.assertTrue(np.array_equal(faces, expected_faces))


def mismaching_faces_error(computed_faces, expected_faces):
    error_message = "computed faces are not equal to expected faces;\n"
    for i in range(len(expected_faces)):
        error_message += f"face\t {i}:\n"
        error_message += f"expected:\t {expected_faces[i]}\n"
        error_message += f"got:\t {computed_faces[i]}\n"

    return error_message


if __name__ == "__main__":
    unittest.main()
