import unittest

import numpy as np

from irregular_object_packing.cat.tetra_cell import TetraCell
from irregular_object_packing.tests.helpers import float_array

SPLIT_INPUT = [[0, 0, 0], [36, 0, 0], [0, 36, 0], [0, 0, 36], ]
SPLIT_4_OUTPUT = [
    [
        [[9, 9, 9], [18, 0, 0], [12, 12, 0]],
        [[9, 9, 9], [12, 12, 0], [0, 18, 0]],
        [[9, 9, 9], [0, 18, 0], [0, 12, 12]],
        [[9, 9, 9], [0, 12, 12], [0, 0, 18]],
        [[9, 9, 9], [0, 0, 18], [12, 0, 12]],
        [[9, 9, 9], [12, 0, 12], [18, 0, 0]]
    ],
    [
        [[9, 9, 9], [18, 0, 0], [12, 12, 0]],
        [[9, 9, 9], [12, 12, 0], [18, 18, 0]],
        [[9, 9, 9], [18, 18, 0], [12, 12, 12]],
        [[9, 9, 9], [12, 12, 12], [18, 0, 18]],
        [[9, 9, 9], [18, 0, 18], [12, 0, 12]],
        [[9, 9, 9], [12, 0, 12], [18, 0, 0]]
    ],
    [
        [[9, 9, 9], [0, 18, 0], [12, 12, 0]],
        [[9, 9, 9], [12, 12, 0], [18, 18, 0]],
        [[9, 9, 9], [18, 18, 0], [12, 12, 12]],
        [[9, 9, 9], [12, 12, 12], [0, 18, 18]],
        [[9, 9, 9], [0, 18, 18], [0, 12, 12]],
        [[9, 9, 9], [0, 12, 12], [0, 18, 0]]
     ],
    [
        [[9, 9, 9], [0, 0, 18], [12, 0, 12]],
        [[9, 9, 9], [12, 0, 12], [18, 0, 18]],
        [[9, 9, 9], [18, 0, 18], [12, 12, 12]],
        [[9, 9, 9], [12, 12, 12], [0, 18, 18]],
        [[9, 9, 9], [0, 18, 18], [0, 12, 12]],
        [[9, 9, 9], [0, 12, 12], [0, 0, 18]]
    ]
]


SPLIT_3_OUTPUT = [
    [
        [[0, 12, 12], [12, 12, 12], [18, 18, 0],[0, 18, 0]],
        [[0, 12, 12], [12, 12, 12], [18, 0, 18], [0, 0, 18]],
    ],
    [
        [[0, 12, 12], [12, 12, 12], [18, 18, 0],[0, 18, 0]],
        [[0, 12, 12], [12, 12, 12], [18, 0, 18], [0, 0, 18]],
    ],
    [
        [[0, 12, 12], [12, 12, 12], [18, 18, 0],[0, 18, 0]],
        [[0, 12, 12], [12, 12, 12], [0, 18, 18]],
    ],
    [
        [[0, 12, 12], [12, 12, 12], [18, 0, 18], [0, 0, 18]],
        [[0, 12, 12], [12, 12, 12], [0, 18, 18]],
    ],
]

SPLIT_2_3331_OUTPUT = [
    [
        [[0, 0, 18], [18, 0, 18], [0, 18, 18]],
    ],
    [
        [[0, 0, 18], [18, 0, 18], [0, 18, 18]],
    ],
    [
        [[0, 0, 18], [18, 0, 18], [0, 18, 18]],
    ],
    [
        [[0, 0, 18], [18, 0, 18], [0, 18, 18]],
    ],
]


SPLIT_2_2222_OUTPUT = [
    [
        [[0, 18, 0], [0, 0, 18], [18, 0, 18], [18, 18, 0]],
    ],
    [
        [[0, 18, 0], [0, 0, 18], [18, 0, 18], [18, 18, 0]],
    ],
    [
        [[0, 18, 0], [0, 0, 18], [18, 0, 18], [18, 18, 0]],
    ],
    [
        [[0, 18, 0], [0, 0, 18], [18, 0, 18], [18, 18, 0]],
    ],
]

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

def cell_1111_reverse() -> tuple[TetraCell, list, list]:
    point_ids = [1, 2, 3, 4]
    obj_ids = [4, 3, 2, 1]
    return TetraCell(point_ids, obj_ids, 1), point_ids, obj_ids

def cell_1111_reverse_2() -> tuple[TetraCell, list, list]:
    point_ids = [4,3,2,1]
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

def cell_4444() -> tuple[TetraCell, list, list]:
    point_ids = [1, 2, 3, 4]
    obj_ids = [1, 1, 1, 1]
    return TetraCell(point_ids, obj_ids, 1), point_ids, obj_ids

# have [a, b, c, d] which corresponds to [1, 2, 3, 4]
# want [d, c, b, a] which corresponds to [4, 3, 2, 1]
# want [b, a, d, c] which corresponds to [2, 1, 4, 3]
def reorder_split_input(point_ids, SPLIT_INPUT=SPLIT_INPUT):
    """convenience func to resort points so that sorted_points[point_ids]
    returns the same input as SPLIT_INPUT[1, 2, 3, 4]"""
    # Adds a NaN row to the input data to cover for index 0
    nan_row = [[np.NaN, np.NaN, np.NaN]]
    sorted_points = [[]] * 4

    for i, pid in enumerate(point_ids):
        sorted_points[pid-1] = SPLIT_INPUT[i]

    return float_array(nan_row + sorted_points)



# Set up a mock SPLIT_INPUT and float_array function for testing
RESORT_TEST_INPUT = [[0.0,0.0,0.0], [1.0,1.0,1.0], [2.0,2.0,2.0], [3.0,3.0,3.0]]

class TestResortPoints(unittest.TestCase):
    def setUp(self):
        self.nan_row = [[np.NaN, np.NaN, np.NaN]]
        self.expected_result = lambda lst: float_array(self.nan_row + lst)

    def test_resort_points_ordered(self):
        # Test that the function correctly reorders the points
        # Same Order
        np.testing.assert_array_equal(
            reorder_split_input([1, 2, 3, 4], RESORT_TEST_INPUT),
            float_array(self.nan_row + RESORT_TEST_INPUT)
        )

    def test_resort_points_reverse_order(self):
        # Reversed Order
        np.testing.assert_array_equal(
            reorder_split_input([4, 3, 2, 1], RESORT_TEST_INPUT),
            self.expected_result([RESORT_TEST_INPUT[3], RESORT_TEST_INPUT[2], RESORT_TEST_INPUT[1], RESORT_TEST_INPUT[0]])
        )

        # Mixed Order
    def test_resort_points_mixed_order(self):
        np.testing.assert_array_equal(
            reorder_split_input([2, 1, 4, 3], RESORT_TEST_INPUT),
            self.expected_result([RESORT_TEST_INPUT[1], RESORT_TEST_INPUT[0], RESORT_TEST_INPUT[3], RESORT_TEST_INPUT[2]])
        )

if __name__ == "__main__":
    unittest.main()
