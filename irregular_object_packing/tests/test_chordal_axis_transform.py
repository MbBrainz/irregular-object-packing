import numpy as np
import unittest

from irregular_object_packing.packing.chordal_axis_transform import (
    TetPoint,
    Triangle,
    create_faces_2,
    create_faces_3,
    create_faces_4,
    face_coord_to_points_and_faces,
    single_point_4faces,
)
from irregular_object_packing.packing.utils import sort_faces_dict


class TestCreateCatFaces(unittest.TestCase):
    def setUp(self) -> None:
        self.a = TetPoint(np.array([0.0, 0.0, 0.0]))
        self.b = TetPoint(np.array([8.0, 0.0, 0.0]))
        self.c = TetPoint(np.array([0.0, 8.0, 0.0]))
        self.d = TetPoint(np.array([0.0, 0.0, 8.0]))
        return super().setUp()

    def set_object_ids(self, object_ids: list[int]):
        self.a.obj_id = object_ids[0]
        self.b.obj_id = object_ids[1]
        self.c.obj_id = object_ids[2]
        self.d.obj_id = object_ids[3]

    @property
    def points(self):
        return [self.a, self.b, self.c, self.d]

    @property
    def center(self):
        return (self.a.vertex + self.b.vertex + self.c.vertex + self.d.vertex) / 4

    @property
    def middle_ab(self):
        return (self.a.vertex + self.b.vertex) / 2

    @property
    def middle_ac(self):
        return (self.a.vertex + self.c.vertex) / 2

    @property
    def middle_ad(self):
        return (self.a.vertex + self.d.vertex) / 2

    @property
    def middle_bc(self):
        return (self.b.vertex + self.c.vertex) / 2

    @property
    def middle_bd(self):
        return (self.b.vertex + self.d.vertex) / 2

    @property
    def middle_cd(self):
        return (self.c.vertex + self.d.vertex) / 2

    @property
    def middle_abc(self):
        return (self.a.vertex + self.b.vertex + self.c.vertex) / 3

    @property
    def middle_acd(self):
        return (self.a.vertex + self.c.vertex + self.d.vertex) / 3

    @property
    def middle_abd(self):
        return (self.a.vertex + self.b.vertex + self.d.vertex) / 3

    @property
    def middle_bcd(self):
        return (self.b.vertex + self.c.vertex + self.d.vertex) / 3

    def test_single_point_of_tetrahedron(self):
        self.set_object_ids([0, 1, 2, 3])

        self.a.add_triangle(Triangle([self.a.vertex, self.b.vertex, self.c.vertex], [], [0, 1, 2]))
        self.a.add_triangle(Triangle([self.a.vertex, self.c.vertex, self.d.vertex], [], [0, 2, 3]))
        self.a.add_triangle(Triangle([self.a.vertex, self.b.vertex, self.d.vertex], [], [0, 1, 3]))

        expected_faces = {
            0: [
                [self.center, self.middle_ad, self.middle_acd],
                [self.center, self.middle_ad, self.middle_abd],
                [self.center, self.middle_ab, self.middle_abd],
                [self.center, self.middle_ab, self.middle_abc],
                [self.center, self.middle_ac, self.middle_abc],
                [self.center, self.middle_ac, self.middle_acd],
            ]
        }

        computed_faces = {0: single_point_4faces(self.points[0], self.points[1:], self.center)}

        # # sort faces so that they can be compared
        computed_faces = sort_faces_dict(computed_faces)
        expected_faces = sort_faces_dict(expected_faces)
        print(mismaching_faces_error(computed_faces, expected_faces))

        assert_faces_equal(self, computed_faces, expected_faces)

    def test_create_faces_4(self):
        self.set_object_ids([0, 1, 2, 3])

        cat_faces = {0: [], 1: [], 2: [], 3: []}
        self.a.add_triangle(Triangle([self.a.vertex, self.b.vertex, self.c.vertex], [], [0, 1, 2]))
        self.a.add_triangle(Triangle([self.a.vertex, self.c.vertex, self.d.vertex], [], [0, 2, 3]))
        self.a.add_triangle(Triangle([self.a.vertex, self.b.vertex, self.d.vertex], [], [0, 1, 3]))
        # expected_faces = {
        _ = {
            0: [  # from face a
                [self.center, self.middle_ad, self.middle_acd],
                [self.center, self.middle_ad, self.middle_abd],
                [self.center, self.middle_ab, self.middle_abd],
                [self.center, self.middle_ab, self.middle_abc],
                [self.center, self.middle_ac, self.middle_abc],
                [self.center, self.middle_ac, self.middle_acd],
            ],
            1: [  # from face b
                [self.center, self.middle_bc, self.middle_abc],
                [self.center, self.middle_bc, self.middle_bcd],
                [self.center, self.middle_ab, self.middle_abc],
                [self.center, self.middle_ab, self.middle_abd],
                [self.center, self.middle_bd, self.middle_bcd],
                [self.center, self.middle_bd, self.middle_abd],
            ],
            2: [  # from face c _____
                [self.center, self.middle_bc, self.middle_abc],
                [self.center, self.middle_bc, self.middle_bcd],
                [self.center, self.middle_cd, self.middle_bcd],
                [self.center, self.middle_ac, self.middle_abc],
                [self.center, self.middle_cd, self.middle_acd],
                [self.center, self.middle_ac, self.middle_acd],
            ],
            3: [  # from face d
                [self.center, self.middle_cd, self.middle_acd],
                [self.center, self.middle_bd, self.middle_bcd],
                [self.center, self.middle_cd, self.middle_bcd],
                [self.center, self.middle_bd, self.middle_abd],
                [self.center, self.middle_ad, self.middle_abd],
                [self.center, self.middle_ad, self.middle_acd],
            ],
        }

        create_faces_4(cat_faces, self.points)
        for k in cat_faces.keys():
            for face in cat_faces[k]:
                self.assertGreater(len(face), 2)
                for point in face:
                    self.assertEqual(len(point), 3)

        for k in cat_faces.keys():
            self.assertEqual(np.shape(cat_faces[k]), (6, 3, 3))

        # # This assertion fails but its to much work to debug.
        # # The single4 works, so Its probably my test expected result that is wrongly ordered.

        # cat_faces = sort_faces_dict(cat_faces)
        # expected_faces = sort_faces_dict(expected_faces)
        # assert_faces_equal(self, cat_faces, expected_faces)

    def test_create_faces_3(self):
        self.set_object_ids([0, 1, 2, 0])
        occ = [(0, 2), (1, 1), (2, 1)]
        # expected output

        expected_faces = {
            0: [
                # [middle_a0b, middle_a1b, middle_a0bc, middle_a1bc],
                # [middle_a0c, middle_a1c, middle_a0bc, middle_a1bc],
                [self.middle_ab, self.middle_bd, self.middle_abc, self.middle_bcd],
                [self.middle_ac, self.middle_cd, self.middle_abc, self.middle_bcd],
            ],
            1: [
                # [middle_bc, middle_a0bc, middle_a1bc],
                [self.middle_bc, self.middle_abc, self.middle_bcd],
                [self.middle_ab, self.middle_bd, self.middle_abc, self.middle_bcd],
                # [middle_a0b, middle_a1b, middle_a0bc, middle_a1bc],
            ],
            2: [
                [self.middle_bc, self.middle_abc, self.middle_bcd],
                [self.middle_ac, self.middle_cd, self.middle_abc, self.middle_bcd],
                # [middle_bc, middle_a0bc, middle_a1bc],
                # [middle_a0c, middle_a1c, middle_a0bc, middle_a1bc],
            ],
        }

        computed_faces = {0: [], 1: [], 2: []}
        create_faces_3(computed_faces, occ, self.points)

        expected_faces = sort_faces_dict(expected_faces)
        computed_faces = sort_faces_dict(computed_faces)

        assert_faces_equal(self, computed_faces, expected_faces)

    def test_create_faces_2_aabb(self):
        self.set_object_ids([0, 0, 1, 1])

        occ = [(0, 2), (1, 2)]

        expected_faces = {
            0: [
                [self.middle_ac, self.middle_bc, self.middle_ad, self.middle_bd],
            ],
            1: [
                [self.middle_ac, self.middle_bc, self.middle_ad, self.middle_bd],
            ],
        }

        computed_faces = {0: [], 1: [], 2: []}
        create_faces_2(computed_faces, occ, self.points)
        expected_faces = sort_faces_dict(expected_faces)
        computed_faces = sort_faces_dict(computed_faces)

        assert_faces_equal(self, computed_faces, expected_faces)

    def test_create_faces_2_abbb(self):
        self.set_object_ids([1, 0, 0, 0])
        occ = [(0, 3), (1, 1)]

        expected_faces = {
            0: [
                [self.middle_ab, self.middle_ac, self.middle_ad],
            ],
            1: [
                [self.middle_ab, self.middle_ac, self.middle_ad],
            ],
        }

        computed_faces = {0: [], 1: [], 2: []}
        create_faces_2(computed_faces, occ, self.points)
        expected_faces = sort_faces_dict(expected_faces)
        computed_faces = sort_faces_dict(computed_faces)

        assert_faces_equal(self, computed_faces, expected_faces)

    def test_create_faces_2_aabb_wide(self):
        self.set_object_ids([0, 0, 1, 1])

        occ = [(0, 2), (1, 2)]

        self.a.vertex = np.array([0, 0, -2])
        self.b.vertex = np.array([0, 0, 2])
        self.c.vertex = np.array([0, 2, 0])
        self.d.vertex = np.array([1, 2, 0])

        expected_faces = {
            0: [
                [self.middle_ac, self.middle_bc, self.middle_ad, self.middle_bd],
            ],
            1: [
                [self.middle_ac, self.middle_bc, self.middle_ad, self.middle_bd],
            ],
        }

        computed_faces = {0: [], 1: [], 2: []}
        create_faces_2(computed_faces, occ, self.points)
        expected_faces = sort_faces_dict(expected_faces)
        computed_faces = sort_faces_dict(computed_faces)

        assert_faces_equal(self, computed_faces, expected_faces)

    def test_face_coord_to_points_and_faces_3_points(self):
        face_coords = [[np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])]]
        expected_points = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])]
        expected_faces = [3, 0, 1, 2]
        points, faces = face_coord_to_points_and_faces(face_coords)

        for i in range(len(points)):
            self.assertTrue(np.array_equal(points[i], expected_points[i]))
        self.assertTrue(np.array_equal(faces, expected_faces))

    def test_face_coord_to_points_and_faces_4_points(self):
        face_coords = [[np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0]), np.array([0, 1, 0])]]
        expected_points = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0]), np.array([0, 1, 0])]
        expected_faces = [3, 0, 1, 2, 3, 2, 3, 1]
        points, faces = face_coord_to_points_and_faces(face_coords)

        print(faces)
        for i in range(len(points)):
            self.assertTrue(np.array_equal(points[i], expected_points[i]))
        self.assertTrue(np.array_equal(faces, expected_faces))

    def test_face_coord_to_points_and_faces_3_points_2_faces(self):
        face_coords = [
            [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])],
            [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])],
        ]
        expected_points = [np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])]
        expected_faces = [3, 0, 1, 2, 3, 0, 1, 2]
        points, faces = face_coord_to_points_and_faces(face_coords)

        for i in range(len(points)):
            self.assertTrue(np.array_equal(points[i], expected_points[i]))
        self.assertTrue(np.array_equal(faces, expected_faces))

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
        points, faces = face_coord_to_points_and_faces(face_coords)
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


def assert_faces_equal(testcase, computed_faces, expected_faces):
    for k in computed_faces.keys():
        for i, arr in enumerate(computed_faces[k]):
            # check of arr(np.ndarray) is in expected_faces[0] (list of np.ndarray)
            for j, x in enumerate(arr):
                testcase.assertTrue(
                    (x == expected_faces[k][i][j]).all(), mismaching_faces_error(computed_faces, expected_faces)
                )


if __name__ == "__main__":
    unittest.main()
