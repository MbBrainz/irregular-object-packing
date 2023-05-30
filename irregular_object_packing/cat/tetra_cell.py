from dataclasses import dataclass

import numpy as np
from pyvista import UnstructuredGrid

from irregular_object_packing.cat.tetrahedral_split import (
    split_2_2222,
    split_2_3331,
    split_3,
    split_4,
)
from irregular_object_packing.cat.utils import (
    create_face_normal,
    get_cell_arrays,
    n_related_objects,
    sort_by_occurrance,
)


@dataclass
class TetraCell:
    points: np.ndarray
    objs: np.ndarray
    nobjs: int
    id: int

    def __init__(self, point_ids, object_ids, id):
        """Create a cell object by sorting the points by occurrance."""
        s_point_ids, s_object_ids, case = sort_by_occurrance(point_ids, object_ids)
        self.points = s_point_ids
        self.objs = s_object_ids
        self.case = case
        self.nobjs = len(self.case)
        self.id = id

    @property
    def split_func(self):
        if self.case == (1, 1, 1, 1,):
            return split_4
        elif self.case == (2, 2,):
            return split_2_2222
        elif self.case == (3, 1,):
            return split_2_3331
        elif self.case == (2, 1, 1,):
            return split_3
        else:
            raise ValueError("The cell case is not recognized.")

    def split(self, all_tet_points: np.ndarray) -> tuple[list[np.ndarray]]:
        return self.split_func(all_tet_points[self.points])

