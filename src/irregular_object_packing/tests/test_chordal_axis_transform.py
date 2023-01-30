#%%
import sys
sys.path[0] = '/Users/maurits/code/cls/thesis/irregular-object-packing/irregular-object-packing/src/irregular_object_packing'
from unittest.util import sorted_list_difference
from packing.chordal_axis_transform import *
from packing.utils import *

import unittest
from typing import List

class TestSinglePoint4Faces(unittest.TestCase):
    
    def test_single_point_of_tetrahedron(self):
        self.maxDiff = None
        test_points = [
            TetPoint(Vertex(0.0, 0.0, 0.0, 0), 0),
            TetPoint(Vertex(8.0, 0.0, 0.0, 1), 0), 
            TetPoint(Vertex(0.0, 8.0, 0.0, 2), 0), 
            TetPoint(Vertex(0.0, 0.0 ,8.0, 3), 0)] # initialize other TetPoints
        
        
        # expected output
        center = Vertex(4.0, 4.0, 4.0, 0)
        for other in test_points: center += other.point
        center = center / 4
        
        middle_ab = (test_points[0] + test_points[1]) /2
        middle_ac = (test_points[0] + test_points[2]) /2
        middle_ad = (test_points[0] + test_points[3]) /2
        middle_abc = (test_points[1] + test_points[2]) /3
        middle_acd = (test_points[2] + test_points[3]) /3
        middle_abd = (test_points[1] + test_points[3]) /3
        test_points[0].add_triangle(Triangle([test_points[0].point, test_points[1].point, test_points[2].point], [0,1,2]))
        test_points[0].add_triangle(Triangle([test_points[0].point, test_points[2].point, test_points[3].point], [0,2,3]))
        test_points[0].add_triangle(Triangle([test_points[0].point, test_points[1].point, test_points[3].point], [0,1,3]))
        expected_faces = {0: 
            [
                [center, middle_ab, middle_abc],
                [center, middle_ab, middle_abd],
                [center, middle_ac, middle_abc],
                [center, middle_ac, middle_acd],
                [center, middle_ad, middle_abd],
                [center, middle_ad, middle_acd]
            ]
        }
        
        computed_faces = { 0: single_point_4faces(test_points[0], test_points[1:], center) }
        
        
        # # sort faces so that they can be compared
        computed_faces = sort_faces_dict(computed_faces)
        expected_faces = sort_faces_dict(expected_faces)
        print(mismaching_faces_error(computed_faces, expected_faces))
        
        # self.assertDictEqual(computed_faces, expected_faces, mismaching_faces_error(computed_faces, expected_faces))

    def test_create_faces_3(self):
        test_points = [
            TetPoint(Vertex(0.0, 0.0, 0.0, 0), 0),
            TetPoint(Vertex(0.0, 0.0 ,8.0, 0), 0), # initialize other TetPoints
            TetPoint(Vertex(8.0, 0.0, 0.0, 1), 0), 
            TetPoint(Vertex(0.0, 8.0, 0.0, 2), 0),]

        occ = [(0, 2), (1, 1), (2, 1)]
        # expected output

        middle_a0b = (test_points[0] + test_points[2]) /2
        middle_a0c = (test_points[0] + test_points[3]) /2
        middle_a1b = (test_points[1] + test_points[2]) /2
        middle_a1c = (test_points[1] + test_points[3]) /2
        middle_bc = (test_points[2] + test_points[3]) /2
        middle_a0bc = (test_points[0] + test_points[2] + test_points[3]) /3
        middle_a1bc = (test_points[1] + test_points[2] + test_points[3]) /3


        expected_faces = {
            0: [
                [middle_a0b, middle_a1b, middle_a0bc, middle_a1bc],
                [middle_a0c, middle_a1c, middle_a0bc, middle_a1bc], 
            ],
            1: [
            [middle_bc, middle_a0bc, middle_a1bc],
            [middle_a0b, middle_a1b, middle_a0bc, middle_a1bc], 
            ],
            2: [
            [middle_bc, middle_a0bc, middle_a1bc],
            [middle_a0c, middle_a1c, middle_a0bc, middle_a1bc], 
            ]
        }

        computed_faces = create_faces_3({0:[], 1:[], 2:[]}, occ, test_points)
        expected_faces = sort_faces_dict(expected_faces)
        computed_faces = sort_faces_dict(computed_faces)
        
        self.assertEqual(computed_faces, expected_faces, mismaching_faces_error(computed_faces, expected_faces))

    
def mismaching_faces_error(computed_faces, expected_faces):
    error_message = "computed faces are not equal to expected faces;\n"
    for i in range(len(expected_faces[0])):
        error_message += f"face {i}:\n"
        error_message += f"expected: {expected_faces[0][i]}\n"
        error_message += f"got: {computed_faces[0][i]}\n"
    
    return error_message

if __name__ == '__main__':
    unittest.main()

# %%

test_points = [
    TetPoint(Vertex(0.0, 0.0, 0.0, 0), 0),
    TetPoint(Vertex(8.0, 0.0, 0.0, 1), 0), 
    TetPoint(Vertex(0.0, 8.0, 0.0, 2), 0), 
    TetPoint(Vertex(0.0, 0.0 ,8.0, 3), 0)] # initialize other TetPoints


# expected output
center = Vertex(4.0, 4.0, 4.0, 0)
for other in test_points: center += other.point
center = center / 4

middle_ab = (test_points[0] + test_points[1]) /2
middle_ac = (test_points[0] + test_points[2]) /2
middle_ad = (test_points[0] + test_points[3]) /2
middle_abc = (test_points[1] + test_points[2]) /3
middle_acd = (test_points[2] + test_points[3]) /3
middle_abd = (test_points[1] + test_points[3]) /3
test_points[0].add_triangle(Triangle([test_points[0].point, test_points[1].point, test_points[2].point], [0,1,2]))
test_points[0].add_triangle(Triangle([test_points[0].point, test_points[2].point, test_points[3].point], [0,2,3]))
test_points[0].add_triangle(Triangle([test_points[0].point, test_points[1].point, test_points[3].point], [0,1,3]))
expected_faces = {0: 
    [
        [center, middle_ab, middle_abc],
        [center, middle_ab, middle_abd],
        [center, middle_ac, middle_abc],
        [center, middle_ac, middle_acd],
        [center, middle_ad, middle_abd],
        [center, middle_ad, middle_acd]
    ]
}

computed_faces = { 0: single_point_4faces(test_points[0], test_points[1:], center) }


# # sort faces so that they can be compared
computed_faces = sort_faces_dict(computed_faces)
computed_faces[0].sort()
expected_faces = sort_faces_dict(expected_faces)
expected_faces[0].sort()
print(mismaching_faces_error(computed_faces, expected_faces))
# %%
