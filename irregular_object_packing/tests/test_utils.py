import unittest

from parameterized import parameterized

from irregular_object_packing.packing.utils import split_quadrilateral_to_triangles


def vertices_of_face(face, points):
    return [points[pid] for pid in face]

class TestSplitQuadrilateralToTriangles(unittest.TestCase):
    def setUp(self) -> None:
        self.points = {
            1: (0, 0, 0),
            2: (1, 0, 0),
            3: (1, 1, 0),
            4: (0, 1, 0),
            5: (2, 0, 0),
            6: (2, 1, 0),
            7: (3, 1, 0),
            8: (3, 0, 0),
            9: (4, 0, 0),
            10: (4, 1, 0),
        }
        self.faces = [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1,3,6,9],
            [1, 4, 6, 10],
            [1, 4, 6, 7],
            [1, 2, 5, 6],
        ]
        self.expected_triangles = [
            [[0, 1, 2], [0, 2, 3]],
            [[0, 1, 5], [0, 5, 4]],
            [[1, 2, 6], [1, 6, 5]],
            [[0, 3, 7], [0, 7, 5]],
            [[0, 1, 7], [0, 7, 5]],
        ]

        return super().setUp()

    @parameterized.expand([
        ("square",[1, 2, 3, 4], [[1,2,3],[1,3,4]]),
        ("trapezoid-right",[1, 2, 3, 5], [[1,2,3],[1, 3, 5]]),
        ("trapezoid-acute",[1,3,6,9], [[1,3,6],[1,6,9]]),
        ("trapezoid-obtuse-shift",[1, 4, 6, 10], [[1,4,6],[4,6,10]]),
        ("parallelogram",[1, 4, 6, 7], [[1,4,6],[4,6,7]]),
        ("rectangle",[1, 2, 5, 6], [[1,2,6],[1,5,6]]),
        ("trapezoid-3_side_equal",[1, 3, 6, 8], [[1,3,6],[1,6,8]]),
        ("trapezoid-isosceles",[1, 3, 7, 9], [[1,3,7],[1,7,9]]),
    ])
    def test_correct(self, name, face, expected_triangles):
        triangles = split_quadrilateral_to_triangles(
            face,
            vertices_of_face(face, self.points)
        )
        self.assertEqual(len(triangles), 2)
        for triangle in triangles:
            triangle = sorted(triangle)
            self.assertIn(triangle, expected_triangles)



if __name__ == "__main__":
    unittest.main()
