import unittest

import numpy as np

from irregular_object_packing.cat.tetrahedral_split import (
    split_2_2222,
    split_2_3331,
    split_3,
    split_4,
)
from irregular_object_packing.tests.helpers import float_array
from irregular_object_packing.tests.test_tetrahedral_splits import (
    SPLIT_2_2222_OUTPUT,
    SPLIT_2_3331_OUTPUT,
    SPLIT_3_OUTPUT,
    SPLIT_4_OUTPUT,
    SPLIT_INPUT,
)


class ComputeFaces(unittest.TestCase):
    def test_split_4(self):
        result = split_4(float_array(SPLIT_INPUT))
        np.testing.assert_array_equal(float_array(result), float_array(SPLIT_4_OUTPUT))

    def test_split_3(self):
        result = split_3(float_array(SPLIT_INPUT))
        for i in range(len(result)):
            for j in range(len(result[i])):
                np.testing.assert_array_equal(float_array(result[i][j]), float_array(SPLIT_3_OUTPUT[i][j]))

    def test_split_2_3331(self):
        result = split_2_3331(float_array(SPLIT_INPUT))
        np.testing.assert_array_equal(float_array(result), float_array(SPLIT_2_3331_OUTPUT))

    def test_split_2_2222(self):
        result = split_2_2222(float_array(SPLIT_INPUT))
        np.testing.assert_array_equal(float_array(result), float_array(SPLIT_2_2222_OUTPUT))

    def test_raises(self):
        invalid_arrays = [
            np.ones((3, 3)),
            np.ones((4, 2)),
            np.ones((4, 4)),
            np.ones((4, 3, 3)),
            np.ones((4, 4)),
            np.ones(()),
        ]
        for invalid_array in invalid_arrays:
            self.assertRaises(AssertionError, split_4, invalid_array)
            self.assertRaises(AssertionError, split_3, invalid_array)
            self.assertRaises(AssertionError, split_2_3331, invalid_array)
            self.assertRaises(AssertionError, split_2_2222, invalid_array)


if __name__ == '__main__':
    unittest.main()
