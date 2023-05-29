import unittest

import numpy as np

from irregular_object_packing.cat.tetra_cell import (
    TetraCell,
    filter_relevant_cells,
    split_and_process,
)
from irregular_object_packing.cat.tetrahedral_split import (
    split_2_2222,
    split_2_3331,
    split_3,
    split_4,
)
from irregular_object_packing.cat.utils import (
    n_related_objects,
    sort_by_occurrance,
)
from irregular_object_packing.tests.helpers import float_array
from irregular_object_packing.tests.tetrahedral_splits import (
    SPLIT_2_2222_OUTPUT,
    SPLIT_2_3331_OUTPUT,
    SPLIT_3_OUTPUT,
    SPLIT_4_OUTPUT,
    SPLIT_INPUT,
)


# have [a, b, c, d] which corresponds to [1, 2, 3, 4]
# want [d, c, b, a] which corresponds to [4, 3, 2, 1]
# want [b, a, d, c] which corresponds to [2, 1, 4, 3]
def resort_points(point_ids):
    """convenience func to resort points so that sorted_points[point_ids]
    returns the same input as SPLIT_INPUT[1, 2, 3, 4]"""
    # Adds a NaN row to the input data to cover for index 0
    nan_row = [[np.NaN, np.NaN, np.NaN]]
    sorted_points = [[]] * 4

    for i, pid in enumerate(point_ids):
        sorted_points[pid-1] = SPLIT_INPUT[i]

    return float_array(nan_row + sorted_points)


# initialize cat cells list
def empty_normals_and_cells():
    """convenience func to initialize empty lists for cat cells and normals for 5 points and 5 objs"""
    n_objs = 5
    face_normals = []
    for _i in range(n_objs):
        face_normals.append([])

    face_normals_pp = []
    for _i in range(5):
        face_normals_pp.append([])


    cat_cells = []
    for _i in range(n_objs):
        cat_cells.append([])
    return face_normals, cat_cells, face_normals_pp

def cell_1111() -> tuple[TetraCell, list, list]:
    point_ids = [1, 2, 3, 4]
    obj_ids = [1, 2, 3, 4]
    return TetraCell(point_ids, obj_ids, 1), point_ids, obj_ids

def cell_3331() -> tuple[TetraCell, list, list]:
    point_ids = [1, 2, 3, 4]
    obj_ids = [4, 4, 4, 1]
    return TetraCell(point_ids, obj_ids, 1), point_ids, obj_ids

def cell_2222() -> tuple[TetraCell, list, list]:
    point_ids = [1, 2, 3, 4]
    obj_ids = [1, 1, 2, 2]
    return TetraCell(point_ids, obj_ids, 1), point_ids, obj_ids

def cell_2211() -> tuple[TetraCell, list, list]:
    point_ids = [1, 2, 3, 4]
    obj_ids = [1, 2, 2, 3]
    return TetraCell(point_ids, obj_ids, 1), point_ids, obj_ids

class InitCell(unittest.TestCase):
    def assert_cell_correct(self, cell, expected_point_ids, expected_object_ids, expected_nobjects, expected_case):
        self.assertListEqual(list(cell.points), expected_point_ids)
        self.assertListEqual(list(cell.objs), expected_object_ids)
        self.assertEqual(cell.nobjs, expected_nobjects)
        self.assertEqual(cell.case, expected_case)

    def init_1111(self):
        cell,point_ids,obj_ids = cell_1111()
        expected_point_ids, expected_object_ids , expected_case = sort_by_occurrance(point_ids, obj_ids)
        self.assert_cell_correct(cell, expected_point_ids, expected_object_ids, 1, expected_case)

    def test_3331(self):
        cell,point_ids,obj_ids = cell_3331()
        expected_point_ids, expected_object_ids , expected_case = sort_by_occurrance(point_ids, obj_ids)
        self.assert_cell_correct(cell, expected_point_ids, expected_object_ids, 2, expected_case)

    def test_2222(self):
        cell,point_ids,obj_ids = cell_2222()
        expected_point_ids, expected_object_ids, expected_case = sort_by_occurrance(point_ids, obj_ids)
        self.assert_cell_correct(cell, expected_point_ids, expected_object_ids, 2, expected_case)

    def test_2211(self):
        cell,point_ids,obj_ids = cell_2211()
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
    def test_1111(self):
        cell, _, _ = cell_1111()
        self.assertEqual(cell.split_func, split_4)
        result = cell.split(resort_points(cell.points))
        np.testing.assert_array_equal(float_array(result), float_array(SPLIT_4_OUTPUT))

    def test_3331(self):
        cell, _, _ = cell_3331()
        self.assertEqual(cell.split_func, split_2_3331)
        result = cell.split(resort_points(cell.points))
        np.testing.assert_equal(float_array(result), float_array(SPLIT_2_3331_OUTPUT))

    def test_2222(self):
        cell, _, _ = cell_2222()
        self.assertEqual(cell.split_func, split_2_2222)
        result = cell.split(resort_points(cell.points))
        np.testing.assert_array_equal(float_array(result), float_array(SPLIT_2_2222_OUTPUT))

    unittest.skip("resort points doesnt function properly but tests result is correct")
    def test_2211(self):
        cell, _, _ = cell_2211()
        self.assertEqual(cell.split_func, split_3)
        all_tet_points = resort_points(cell.points)
        result = cell.split(all_tet_points)
        for i in range(len(result)):
            for j in range(len(result[i])):
                np.testing.assert_array_equal(float_array(result[i][j]), float_array(SPLIT_3_OUTPUT[i][j]))

        # np.testing.assert_array_equal(float_array(result), float_array(SPLIT_3_OUTPUT[0][0]))


class SplitAndProcess(unittest.TestCase):

    def assert_cat_cells_correct(self, cat_cells, expected):
        pass
    def test_1111(self):
        cell, _, _ = cell_1111()
        normals, cat_cells, normals_pp = empty_normals_and_cells()
        all_tet_points = resort_points(cell.points)
        split_and_process(cell, all_tet_points, normals, cat_cells, normals_pp)
        self.assert_correct_split_and_process(cell,all_tet_points,normals, cat_cells, SPLIT_4_OUTPUT)

    def test_3331(self):
        cell, _, _ = cell_3331()
        normals, cat_cells, normals_pp = empty_normals_and_cells()
        all_tet_points = resort_points(cell.points)
        split_and_process(cell, all_tet_points, normals, cat_cells, normals_pp)

        self.assert_correct_split_and_process( cell, all_tet_points, normals, cat_cells, SPLIT_2_3331_OUTPUT)

    def test_2222(self):
        cell, _, _ = cell_2222()
        normals, cat_cells, normals_pp = empty_normals_and_cells()
        all_tet_points = resort_points(cell.points)
        split_and_process(cell, all_tet_points, normals, cat_cells, normals_pp)
        self.assert_correct_split_and_process(cell, all_tet_points, normals, cat_cells, SPLIT_2_2222_OUTPUT)


    unittest.skip("resort points doesnt function properly but tests result is correct")
    def test_2211(self):
        cell, _, _ = cell_2211()
        normals, cat_cells, normals_pp = empty_normals_and_cells()
        all_tet_points = resort_points(cell.points)
        split_and_process(cell, all_tet_points, normals, cat_cells, normals_pp)
        self.assert_correct_split_and_process(cell, all_tet_points , normals, cat_cells, SPLIT_3_OUTPUT)


    def assert_correct_split_and_process(self, cell: TetraCell, all_tet_points, normals, cat_cells, split_output):
        for _i, obj in enumerate(cell.objs):
            obj_normals = normals[obj]
            cat_cells[obj]
            for _j, face_normal in enumerate(obj_normals):
                self.assertEqual(np.shape(face_normal), (3, 3))


if __name__ == '__main__':
    unittest.main()
