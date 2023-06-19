import unittest

import numpy as np

from irregular_object_packing.cat.chordal_axis_transform import (
    filter_relevant_cells,
    split_and_process,
)
from irregular_object_packing.cat.tetra_cell import TetraCell
from irregular_object_packing.cat.tetrahedral_split import (
    split_2_2222,
    split_2_3331,
    split_3,
    split_4,
)
from irregular_object_packing.cat.utils import n_related_objects
from irregular_object_packing.tests.helpers import float_array
from irregular_object_packing.tests.test_tetrahedral_splits import (
    SPLIT_2_2222_OUTPUT,
    SPLIT_2_3331_OUTPUT,
    SPLIT_3_OUTPUT,
    SPLIT_4_OUTPUT,
    cell_1111,
    cell_2211,
    cell_2222,
    cell_3331,
    empty_normals_and_cells,
    reorder_split_input,
)


class SplitAndProcess(unittest.TestCase):

    def assert_cat_cells_correct(self, cat_cells, expected):
        pass
    def test_1111(self):
        cell, _, _ = cell_1111()
        normals, cat_cells, normals_pp = empty_normals_and_cells()
        all_tet_points = reorder_split_input(cell.points)
        split_and_process(cell, all_tet_points, normals, cat_cells, normals_pp)
        self.assert_correct_split_and_process(cell,all_tet_points,normals, cat_cells, SPLIT_4_OUTPUT)

    def test_3331(self):
        cell, _, _ = cell_3331()
        normals, cat_cells, normals_pp = empty_normals_and_cells()
        all_tet_points = reorder_split_input(cell.points)
        split_and_process(cell, all_tet_points, normals, cat_cells, normals_pp)

        self.assert_correct_split_and_process( cell, all_tet_points, normals, cat_cells, SPLIT_2_3331_OUTPUT)

    def test_2222(self):
        cell, _, _ = cell_2222()
        normals, cat_cells, normals_pp = empty_normals_and_cells()
        all_tet_points = reorder_split_input(cell.points)
        split_and_process(cell, all_tet_points, normals, cat_cells, normals_pp)
        self.assert_correct_split_and_process(cell, all_tet_points, normals, cat_cells, SPLIT_2_2222_OUTPUT)


    # unittest.skip("resort points doesnt function properly but tests result is correct")
    def test_2211(self):
        cell, _, _ = cell_2211()
        normals, cat_cells, normals_pp = empty_normals_and_cells()
        all_tet_points = reorder_split_input(cell.points)
        split_and_process(cell, all_tet_points, normals, cat_cells, normals_pp)
        self.assert_correct_split_and_process(cell, all_tet_points , normals, cat_cells, SPLIT_3_OUTPUT)


    def assert_correct_split_and_process(self, cell: TetraCell, all_tet_points, res_normals, res_cat_cells, expected_split_output):
        ALL_OBJECTS = [0, 1, 2, 3, 4]
        for _i, obj in enumerate(ALL_OBJECTS):
            if obj not in cell.objs:
                self.assertEqual(res_normals[obj], [])
                self.assertEqual(res_cat_cells[obj], [])

        for _i, obj in enumerate(cell.objs):
            obj_normals = res_normals[obj]
            obj_cell = res_cat_cells[obj]

            for _j, face_normal in enumerate(obj_normals):
                self.assert_cat_cells_correct(obj_cell, expected_split_output)
                self.assertEqual(np.shape(face_normal), (3, 3))

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
    def test_1111(self):
        cell, _, _ = cell_1111()
        self.assertEqual(cell.split_func, split_4)
        result = cell.split(reorder_split_input(cell.points))
        np.testing.assert_array_equal(float_array(result), float_array(SPLIT_4_OUTPUT))

    def test_3331(self):
        cell, _, _ = cell_3331()
        self.assertEqual(cell.split_func, split_2_3331)
        result = cell.split(reorder_split_input(cell.points))
        np.testing.assert_equal(float_array(result), float_array(SPLIT_2_3331_OUTPUT))


    def test_2222(self):
        cell, _, _ = cell_2222()
        self.assertEqual(cell.split_func, split_2_2222)
        result = cell.split(reorder_split_input(cell.points))
        np.testing.assert_array_equal(float_array(result), float_array(SPLIT_2_2222_OUTPUT))

    unittest.skip("resort points doesnt function properly but tests result is correct")
    def test_2211(self):
        cell, _, _ = cell_2211()
        self.assertEqual(cell.split_func, split_3)
        all_tet_points = reorder_split_input(cell.points)
        result = cell.split(all_tet_points)
        for i in range(len(result)):
            for j in range(len(result[i])):
                np.testing.assert_array_equal(float_array(result[i][j]), float_array(SPLIT_3_OUTPUT[i][j]))

        # np.testing.assert_array_equal(float_array(result), float_array(SPLIT_3_OUTPUT[0][0]))
    def test_invalid(self):
        cell, _, _ = cell_1111()
        for case in [(1,), (1, 2), (1, 2, 3, 4, 5)]:
            cell.case = case
            with self.assertRaises(ValueError):
                cell.split(reorder_split_input(cell.points))
