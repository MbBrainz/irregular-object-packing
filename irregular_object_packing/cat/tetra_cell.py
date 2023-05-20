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

    def has_vertex(self, vertex_id):
        return vertex_id in self.points

    def get_point_object_tuple(self):
        return list(zip(self.points, self.objs, strict=True))

    def belongs_to_obj(self, obj_id):
        return obj_id in self.objs

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


def split_and_process(cell: TetraCell, tetmesh_points: np.ndarray, normals: np.ndarray[list[np.ndarray]], cat_cells: np.ndarray[list[np.ndarray]]):
    """Splits the cell into faces and processes them."""
    # 0. split the cell into faces
    split_faces = cell.split(tetmesh_points)

    for i, faces in enumerate(split_faces):
        p_id = cell.points[i]
        obj_id = cell.objs[i]
        obj_point = tetmesh_points[i]
        for face in faces:
            # tetmesh_points[face]
            face_normal = create_face_normal(face[:3], obj_point)
            normals[p_id].append(face_normal)
            cat_cells[obj_id].append(face)


def filter_relevant_cells(cells: list[TetraCell], objects_npoints: list[int]):
    """Filter out cells that only belong to a single object.

    parameters:
    cells (ndarray): an array of shape (n_cells, 4) with the indices of the points in the cell. shape: [id0, id1, id2, id3]
    objects_npoints (List[int]): A list of the number of points for each object.
    """
    relevant_cells: list[TetraCell] = []
    skipped_cells = []

    for i, cell in enumerate(cells):
        rel_objs = n_related_objects(objects_npoints, cell=cell)
        cell = TetraCell(cell, rel_objs, i)
        if cell.nobjs == 1:
            skipped_cells.append(cell)
        else:
            relevant_cells.append(cell)

    return relevant_cells, skipped_cells


def process_cells_to_normals(tetmesh_points: np.ndarray, rel_cells: list[TetraCell], n_objs: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # initialize face normals list
    face_normals = []
    for _i in range(len(tetmesh_points)):
        face_normals.append([])

    # initialize cat cells list
    cat_cells = []
    for _i in range(n_objs):
        cat_cells.append([])

    for cell in rel_cells:
        # mutates face_normals and cat_cells
        split_and_process(cell, tetmesh_points, face_normals, cat_cells)

    return face_normals, cat_cells


def compute_cat_faces_new(tetmesh: UnstructuredGrid, point_sets, obj_coords: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    assert (tetmesh.celltypes == 10).all(), "Tetmesh must be of type tetrahedron"

    objects_npoints = [len(obj) for obj in point_sets]  # FIXME hacky solution

    # filter tetrahedron mesh to only contain tetrahedrons with points from more than one object
    cells = get_cell_arrays(tetmesh.cells)
    rel_cells, _ = filter_relevant_cells(cells, objects_npoints)

    face_normals, cat_cells = process_cells_to_normals(tetmesh.points, rel_cells, len(point_sets))

    return face_normals, cat_cells



# # Maybe usefull later
# def filter_cells_with_vertex(cells: list[TetraCell], vertex_id: int) -> list[TetraCell]:
#     """Filter out cells that only belong to a specific vertex."""
#     return filter(lambda cell: cell.has_vertex(vertex_id), cells)


# def cell_to_tetpoints(cell: TetraCell, tetmesh: UnstructuredGrid):
#     tet_points: list[TetPoint] = []
#     for i, point in enumerate(cell.points):
#         tet_point = TetPoint(
#             point=tetmesh.points[point],
#             p_id=point,
#             obj_id=cell.objs[i],
#             cell_id=cell.id,
#         )
#         tet_points.append(tet_point)

#     return tet_points


# def cell_to_occ_dict(cell: TetraCell):
#     occ_dict = {}
#     for obj_id in cell.objs:
#         occ_dict[obj_id] = occ_dict.get(obj_id, 0) + 1

#     return occ_dict
