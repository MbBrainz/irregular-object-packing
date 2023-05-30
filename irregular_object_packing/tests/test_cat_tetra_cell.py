import unittest

import numpy as np

from irregular_object_packing.cat.tetra_cell import (
    TetraCell,
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
    cell_1111,
    cell_2211,
    cell_2222,
    cell_3331,
)


class InitCell(unittest.TestCase):
    def assert_cell_correct(self, cell, expected_point_ids, expected_object_ids, expected_nobjects, expected_case):
        self.assertListEqual(list(cell.points), expected_point_ids)
        self.assertListEqual(list(cell.objs), expected_object_ids)
        self.assertEqual(cell.nobjs, expected_nobjects)
        self.assertEqual(cell.case, expected_case)

    def test_1111(self):
        cell,point_ids,obj_ids = cell_1111()
        expected_point_ids, expected_object_ids , expected_case = sort_by_occurrance(point_ids, obj_ids)
        self.assert_cell_correct(cell, expected_point_ids, expected_object_ids, 4, expected_case)

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


if __name__ == '__main__':
    unittest.main()
