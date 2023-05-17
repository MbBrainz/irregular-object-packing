# %%
from collections import Counter
from dataclasses import dataclass

import numpy as np
from pyvista import UnstructuredGrid

from irregular_object_packing.packing.cat import TetPoint


def sort_face_points_by_length(expected_faces):
    sorted_faces = []
    for face in expected_faces:
        sorted_faces.append(sort_points_by_polar_angles(face))
        # sort_face_points_by_length(face))

    return sorted_faces


def sort_points_by_polar_angles(points):
    points = np.array(points)
    # Calculate theta and phi angles for each point
    theta = np.arctan2(points[:, 1], points[:, 0])
    phi = np.arccos(points[:, 2] / np.linalg.norm(points, axis=1))

    # Combine theta and phi into a single array
    polar_angles = np.column_stack((theta, phi))

    # Sort the points based on the polar angles
    sorted_indices = np.lexsort(np.transpose(polar_angles))
    return points[sorted_indices]


def sort_faces_dict(faces):
    for k, _v in faces.items():
        faces[k] = sort_face_points_by_length(faces[k])

    return faces


def compute_face_unit_normal(points, v_i):
    """Compute the normal vector of a planar face defined by either 3 or 4 points in 3D
    space.

    This function calculates the normal vector of a planar face, ensuring that it points
    in the direction of the reference point v_i.

    Parameters:
    points (List[np.ndarray]): A list of either 3 or 4 numpy arrays representing points in 3D space,
                               with x, y, and z coordinates.
    v_i (np.ndarray): A numpy array representing a point in 3D space, with x, y, and z coordinates,
                      used to determine the direction of the normal vector.

    Returns:
    np.ndarray: The normal vector of the planar face, pointing in the direction of v_i.

    Raises:
    AssertionError: If the number of points in the input list is not 3 or 4.

    Examples
    --------
    >>> compute_face_normal(np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]]), np.array([0, 1, 2]))
    array([0, 1, 0])
    """
    n_points = len(points)
    assert 3 <= n_points <= 4, "The number of points should be either 3 or 4."

    v0 = points[1] - points[0]
    v1 = points[2] - points[0]
    normal = np.cross(v0, v1)
    if np.dot(normal, v_i - points[0]) < 0:
        normal *= -1
    unit_normal = normal / np.linalg.norm(normal)

    return unit_normal


def print_transform_array(array):
    symbols = ["f", "θ_x", "θ_y", "θ_z", "t_x", "t_y", "t_z"]
    header = " ".join([f"{symbol+':':<8}" for symbol in symbols])
    row = " ".join([f"{value:<8.3f}" for value in array])
    print(header)
    print(row + "\n")


def get_tetmesh_cell_arrays(tetmesh: UnstructuredGrid) -> np.ndarray:
    return get_cell_arrays(tetmesh.cells)


def get_cell_arrays(cells: np.ndarray) -> np.ndarray:
    """Get the cell arrays from a pyvista.UnstructuredGrid object.
    This function assumes that the cells are tetrahedrons.

    -> tetmesh.cells will return a numpy array of shape (n_cells, 5),
    where the first column is the number of vertices in the cell,
    and the remaining columns are the indices of the vertices.
    """
    return np.array(np.hsplit(cells, cells.size / 5)).reshape(-1, 5)[:, 1:]


def n_related_objects(objects_npoints, cell) -> np.ndarray:
    """Check if a cell belongs to a single object or not.

    Parameters:
    objects_npoints (List[int]): A list of the number of points for each object.
    cell (ndarray): an array of shape (4,) with the indices of the points in the cell. shape: [id0, id1, id2, id3]"""
    # Get a numpy array of the number of points for each object

    # Calculate the cumulative sum to get the ranges
    ranges = np.cumsum(objects_npoints)

    # Check which objects the points belong to
    point_objects = np.searchsorted(ranges, cell, side='right')

    return point_objects


@dataclass
class Cell:
    points: np.ndarray
    objs: np.ndarray
    nobjs: int
    id: int

    def __init__(self, point_ids, object_ids, id):
        """Create a cell object by sorting the points by occurrance."""
        s_point_ids, s_object_ids = sort_by_occurrance(point_ids, object_ids)
        self.points = s_point_ids
        self.objs = s_object_ids
        self.nobjs = len(np.unique(object_ids))
        self.id = id

    def has_vertex(self, vertex_id):
        return vertex_id in self.points

    def get_point_object_tuple(self):
        return list(zip(self.points, self.objs, strict=True))




def filter_cells_with_vertex(cells: list[Cell], vertex_id: int) -> list[Cell]:
    """Filter out cells that only belong to a specific vertex."""
    return filter(lambda cell: cell.has_vertex(vertex_id), cells)

def filter_relevant_cells(cells, objects_npoints):
    """Filter out cells that only belong to a single object.

    parameters:
    cells (ndarray): an array of shape (n_cells, 4) with the indices of the points in the cell. shape: [id0, id1, id2, id3]
    objects_npoints (List[int]): A list of the number of points for each object.
    """
    relevant_cells: list[Cell] = []
    skipped_cells = []

    for i, cell in enumerate(cells):
        rel_objs = n_related_objects(objects_npoints, cell=cell)
        cell = Cell(cell, rel_objs, i)
        if cell.nobjs == 1:
            skipped_cells.append(cell)
        else:
            relevant_cells.append(cell)

    return relevant_cells, skipped_cells


def cell_to_tetpoints(cell: Cell, tetmesh: UnstructuredGrid):
    tet_points: list[TetPoint] = []
    for i, point in enumerate(cell.points):
        tet_point = TetPoint(tetmesh.points[point], point, cell.objs[i], cell.id)
        tet_points.append(tet_point)

    return tet_points


def cell_to_occ_dict(cell: Cell):
    occ_dict = {}
    for obj_id in cell.objs:
        occ_dict[obj_id] = occ_dict.get(obj_id, 0) + 1

    return occ_dict


def sort_by_occurrance(point_ids: list[int], object_ids: list[int]) -> list[int]:
    """Sort a list of point ids by the number of times they occur in the list of object ids.

    Parameters:
    point_ids (list[int]): A list of point ids.
    object_ids (list[int]): A list of object ids.

    Returns:
    list[int]: A list of point ids sorted by the number of times they occur in the list of object ids.
    """
    if len(point_ids) != len(object_ids) or len(object_ids) == 0:
        raise ValueError("The number of point ids and object ids must be non empty and the same")


    count_dict = Counter(object_ids)
    combined = list(zip(point_ids, object_ids, strict=True))
    combined.sort(key=lambda x: (count_dict[x[1]], x[1]), reverse=True)

    sorted_point_ids, sorted_object_ids = zip(*combined, strict=True)

    return list(sorted_point_ids), list(sorted_object_ids)
