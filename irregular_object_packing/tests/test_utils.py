import unittest


from irregular_object_packing.packing.utils import *


class TestSplitQuadrilateralToTriangles(unittest.TestCase):
    def test_square(self):
        points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        expected_triangles = [[(0, 0, 0), (1, 1, 0), (1, 0, 0)], [(0, 0, 0), (1, 1, 0), (0, 1, 0)]]
        triangles = split_quadrilateral_to_triangles(points)
        self.assertEqual(len(triangles), 2)
        for triangle in triangles:
            self.assertIn(triangle, expected_triangles)

    def test_square_shifted(self):
        points = [(0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0)]
        expected_triangles = [
            [(0, 1, 0), (1, 0, 0), (0, 0, 0)],
            [(0, 1, 0), (1, 0, 0), (1, 1, 0)],
        ]
        triangles = split_quadrilateral_to_triangles(points)
        self.assertEqual(len(triangles), 2)
        for triangle in triangles:
            self.assertIn(triangle, expected_triangles)

    def test_square_3d(self):
        points = [(0, 0, 0), (0, 1, 0), (1, 1, 1), (1, 0, 1)]
        expected_triangles = [[(0, 0, 0), (1, 1, 1), (0, 1, 0)], [(0, 0, 0), (1, 1, 1), (1, 0, 1)]]
        triangles = split_quadrilateral_to_triangles(points)
        self.assertEqual(len(triangles), 2)
        for triangle in triangles:
            self.assertIn(triangle, expected_triangles)

    def test_rectangle(self):
        points = [(0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0)]
        expected_triangles = [[(0, 0, 0), (2, 3, 0), (2, 0, 0)], [(0, 0, 0), (2, 3, 0), (0, 3, 0)]]
        triangles = split_quadrilateral_to_triangles(points)
        self.assertEqual(len(triangles), 2)
        for triangle in triangles:
            self.assertIn(triangle, expected_triangles)

    def test_rectangle_shifted(self):
        points = [(0, 0, 0), (0, 3, 0), (2, 3, 0), (2, 0, 0)]
        expected_triangles = [[(0, 0, 0), (2, 3, 0), (2, 0, 0)], [(0, 0, 0), (2, 3, 0), (0, 3, 0)]]
        triangles = split_quadrilateral_to_triangles(points)
        self.assertEqual(len(triangles), 2)
        for triangle in triangles:
            self.assertIn(triangle, expected_triangles)

    def test_non_symmetric(self):
        points = [(1, 1, 0), (4, 1, 0), (4, 4, 0), (1, 4, 0)]
        expected_triangles = [[(1, 1, 0), (4, 4, 0), (4, 1, 0)], [(1, 1, 0), (4, 4, 0), (1, 4, 0)]]
        triangles = split_quadrilateral_to_triangles(points)
        self.assertEqual(len(triangles), 2)
        for triangle in triangles:
            self.assertIn(triangle, expected_triangles)

    def test_diamond(self):
        points = [(0, 2, 0), (2, 0, 0), (4, 2, 0), (2, 4, 0)]
        expected_triangles = [[(0, 2, 0), (4, 2, 0), (2, 0, 0)], [(0, 2, 0), (4, 2, 0), (2, 4, 0)]]
        triangles = split_quadrilateral_to_triangles(points)
        self.assertEqual(len(triangles), 2)
        for triangle in triangles:
            self.assertIn(triangle, expected_triangles)


if __name__ == "__main__":
    unittest.main()
