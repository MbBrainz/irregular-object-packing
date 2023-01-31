
import sys
sys.path[0] = '/Users/maurits/code/cls/thesis/irregular-object-packing/irregular-object-packing/src/irregular_object_packing'

from unittest.util import sorted_list_difference
from packing.chordal_axis_transform import *
from packing.utils import *

import unittest
from typing import List

class TestCreateCatFaces(unittest.TestCase):
    def setUp(self) -> None:
        self.a = TetPoint(np.array([0.0, 0.0, 0.0]))
        self.b = TetPoint(np.array([8.0, 0.0, 0.0]))
        self.c = TetPoint(np.array([0.0, 8.0, 0.0]))
        self.d = TetPoint(np.array([0.0, 0.0 ,8.0]))
        return super().setUp()

    def set_object_ids(self, object_ids: list[int]):
        self.a.vertex_id = object_ids[0]
        self.b.vertex_id = object_ids[1]
        self.c.vertex_id = object_ids[2]
        self.d.vertex_id = object_ids[3]
        
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

        self.a.add_triangle(Triangle([self.a.vertex, self.b.vertex, self.c.vertex],[], [0,1,2]))
        self.a.add_triangle(Triangle([self.a.vertex, self.c.vertex, self.d.vertex],[], [0,2,3]))
        self.a.add_triangle(Triangle([self.a.vertex, self.b.vertex, self.d.vertex],[], [0,1,3]))
        
        expected_faces = {0: 
            [
                [self.center, self.middle_ad, self.middle_acd],
                [self.center, self.middle_ad, self.middle_abd],
                [self.center, self.middle_ab, self.middle_abd],
                [self.center, self.middle_ab, self.middle_abc],
                [self.center, self.middle_ac, self.middle_abc],
                [self.center, self.middle_ac, self.middle_acd],
            ]
        }
        
        computed_faces = { 0: single_point_4faces(self.points[0], self.points[1:], self.center) }
        
        # # sort faces so that they can be compared
        computed_faces = sort_faces_dict(computed_faces)
        expected_faces = sort_faces_dict(expected_faces)
        print(mismaching_faces_error(computed_faces, expected_faces))
        
        assert_faces_equal(self, computed_faces, expected_faces)
        
    def test_create_faces_4(self):
        
        cat_faces = {0:[], 1:[], 2:[], 3:[]}
        self.a.add_triangle(Triangle([self.a.vertex, self.b.vertex, self.c.vertex],[], [0,1,2]))
        self.a.add_triangle(Triangle([self.a.vertex, self.c.vertex, self.d.vertex],[], [0,2,3]))
        self.a.add_triangle(Triangle([self.a.vertex, self.b.vertex, self.d.vertex],[], [0,1,3]))
        expected_faces = {
            0: [ # from face a
                [self.center, self.middle_ad, self.middle_acd],
                [self.center, self.middle_ad, self.middle_abd],
                [self.center, self.middle_ab, self.middle_abd],
                [self.center, self.middle_ab, self.middle_abc],
                [self.center, self.middle_ac, self.middle_abc],
                [self.center, self.middle_ac, self.middle_acd],
            ],
            1: [ # from face b
                [self.center, self.middle_bc, self.middle_abc],
                [self.center, self.middle_bc, self.middle_bcd], 
                [self.center, self.middle_ab, self.middle_abc],
                [self.center, self.middle_ab, self.middle_abd],
                [self.center, self.middle_bd, self.middle_bcd],
                [self.center, self.middle_bd, self.middle_abd],
            ],
            2: [ # from face c _____
                [self.center, self.middle_bc, self.middle_abc],
                [self.center, self.middle_bc, self.middle_bcd],
                [self.center, self.middle_cd, self.middle_bcd],
                [self.center, self.middle_ac, self.middle_abc],
                [self.center, self.middle_cd, self.middle_acd],
                [self.center, self.middle_ac, self.middle_acd],
            ],
            3: [ # from face d
                [self.center, self.middle_cd, self.middle_acd],
                [self.center, self.middle_bd, self.middle_bcd], 
                [self.center, self.middle_cd, self.middle_bcd],
                [self.center, self.middle_bd, self.middle_abd],
                [self.center, self.middle_ad, self.middle_abd],
                [self.center, self.middle_ad, self.middle_acd]
            ]
        }
        
        pass

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
            ]
        }

        computed_faces = create_faces_3({0:[], 1:[], 2:[]}, occ, self.points)
        expected_faces = sort_faces_dict(expected_faces)
        computed_faces = sort_faces_dict(computed_faces)

        assert_faces_equal(self, computed_faces, expected_faces)
        

    def test_create_faces_2_aabb(self):
        points = [ TetPoint(self.a, 0, 0), TetPoint(self.b, 0, 0), TetPoint(self.c, 1, 0), TetPoint(self.d, 1, 0)]

        occ = [(0, 2), (1, 2)]
        middle_a0b0 = (self.a + self.c) /2
        middle_a0b1 = (self.a + self.d) /2
        middle_a1b0 = (self.b + self.c)/2
        middle_a1b1 = (self.b + self.d) /2

        expected_faces = {
            0: [
                [middle_a0b0, middle_a1b0, middle_a0b1, middle_a1b1], 
            ],
            1: [
                [middle_a0b0, middle_a1b0, middle_a0b1, middle_a1b1],
            ]
        }

        computed_faces = create_faces_2({0:[], 1:[], 2:[]}, occ, points)
        expected_faces = sort_faces_dict(expected_faces)
        computed_faces = sort_faces_dict(computed_faces)

        assert_faces_equal(self, computed_faces, expected_faces)
        
    def test_create_faces_2_abbb(self):
        test_points = [
            TetPoint(np.array([0.0, 0.0, 0.0,]), 0, 0),
            TetPoint(np.array([0.0, 0.0 ,8.0,]), 1, 0), # initialize other TetPoints
            TetPoint(np.array([8.0, 0.0, 0.0,]), 1, 0), 
            TetPoint(np.array([0.0, 8.0, 0.0,]), 1, 0),]
        occ = [(0, 3), (1, 1)]
        middle_a0b0 = (self.a + self.b) /2
        middle_a0b1 = (self.a + self.c) /2
        middle_a0b2 = (self.a + self.d) /2


        expected_faces = {
            0: [
                [middle_a0b0,  middle_a0b1, middle_a0b2], 
            ],
            1: [
                [middle_a0b0,  middle_a0b1, middle_a0b2],
            ]
        }

        computed_faces = create_faces_2({0:[], 1:[], 2:[]}, occ, test_points)
        expected_faces = sort_faces_dict(expected_faces)
        computed_faces = sort_faces_dict(computed_faces)

        assert_faces_equal(self, computed_faces, expected_faces)
        

    
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
                testcase.assertTrue((x == expected_faces[k][i][j]).all(), mismaching_faces_error(computed_faces, expected_faces))

if __name__ == '__main__':
    unittest.main()
