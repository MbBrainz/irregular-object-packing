import unittest

import numpy as np

from irregular_object_packing.cat.tetra_cell import TetraCell, filter_relevant_cells
from irregular_object_packing.cat.utils import n_related_objects, sort_by_occurrance


class InitCell(unittest.TestCase):
    def assert_cell_correct(self, cell, expected_point_ids, expected_object_ids, expected_nobjects, expected_case):
        self.assertListEqual(list(cell.points), expected_point_ids)
        self.assertListEqual(list(cell.objs), expected_object_ids)
        self.assertEqual(cell.nobjs, expected_nobjects)
        self.assertEqual(cell.case, expected_case)

    def trivial(self):
        point_ids = [1, 2, 3, 4]
        obj_ids = [1, 2, 3, 4]
        cell = TetraCell(point_ids, obj_ids, 1)
        expected_point_ids, expected_object_ids , expected_case = sort_by_occurrance(point_ids, obj_ids)
        self.assert_cell_correct(cell, expected_point_ids, expected_object_ids, 1, expected_case)

    def test_3331(self):
        point_ids = [1, 2, 3, 4]
        obj_ids = [6, 6, 6, 1]
        cell = TetraCell(point_ids, obj_ids, 1)
        expected_point_ids, expected_object_ids , expected_case = sort_by_occurrance(point_ids, obj_ids)
        self.assert_cell_correct(cell, expected_point_ids, expected_object_ids, 2, expected_case)

    def test_2222(self):
        point_ids = [1, 2, 3, 4]
        obj_ids = [1, 1, 2, 2]
        cell = TetraCell(point_ids, obj_ids, 1)
        expected_point_ids, expected_object_ids, expected_case = sort_by_occurrance(point_ids, obj_ids)
        self.assert_cell_correct(cell, expected_point_ids, expected_object_ids, 2, expected_case)

    def test_2211(self):
        point_ids = [1, 2, 3, 4]
        obj_ids = [1, 2, 2, 3]
        cell = TetraCell(point_ids, obj_ids, 1)
        expected_point_ids, expected_object_ids , expected_case = sort_by_occurrance(point_ids, obj_ids)
        self.assert_cell_correct(cell, expected_point_ids, expected_object_ids, 3, expected_case)


class FilterRelevantCells(unittest.TestCase):

    def assert_relevant_correct(self, relevant, expected):
        self.assertListEqual(relevant, expected, "Relevant cells are not correct")

    def assert_skipped_correct(self, skipped, expected):
        self.assertListEqual(skipped, expected, "Skipped cells are not correct")

    def test_empty(self):
        # Test with no cells
        cells = np.array([])
        objects_npoints = []
        relevant, skipped = filter_relevant_cells(cells, objects_npoints)
        self.assert_relevant_correct(relevant, [])
        self.assert_skipped_correct(skipped, [])

        # Test with one cell that belongs to a single object
    def test_single_cell(self):
        cells = np.array([[0, 1, 2, 3]])
        objects_npoints = [4]
        relevant, skipped = filter_relevant_cells(cells, objects_npoints)
        self.assertListEqual(
            skipped,
            [TetraCell(cells[0], n_related_objects(objects_npoints, cells[0]), 0)])

    def test_two_cells(self):
        # Test with two cells, one belongs to a single object and one belongs to two objects
        cells = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
        objects_npoints = [5, 4]
        relevant, skipped = filter_relevant_cells(cells, objects_npoints)
        expected_cell = TetraCell(np.array([2, 3, 4, 5]), np.array([0, 0, 1, 1]), 1)

        self.assertTrue(relevant[0], expected_cell)
        self.assertTrue(len(skipped), 1)

class SplitCell(unittest.TestCase):
    # TODO: Implement
    pass

class SplitAndProcess(unittest.TestCase):
    # TODO: Implement
    pass

if __name__ == '__main__':
    unittest.main()
