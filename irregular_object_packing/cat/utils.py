from collections import Counter

import numpy as np
from pyvista import UnstructuredGrid


def sort_by_occurrance(point_ids: list[int], object_ids: list[int]) -> list[int]:
    """Sort a list of point ids by the number of times they occur in the list of object ids.

    Parameters:
    point_ids (list[int]): A list of point ids.
    object_ids (list[int]): A list of object ids.

    Returns:
    list[int]: A list of point ids sorted by the number of times they occur in the list of object ids.
    """
    if len(point_ids) != len(object_ids) or len(object_ids) != 4:
        raise ValueError("The number of point ids and object ids must be lenght 4.")

    count_dict = Counter(object_ids)
    combined = list(zip(point_ids, object_ids, strict=True))
    combined.sort(key=lambda x: (count_dict[x[1]], x[1]), reverse=True)

    sorted_point_ids, sorted_object_ids = zip(*combined, strict=True)

    return list(sorted_point_ids), list(sorted_object_ids), tuple(sorted(count_dict.values(), reverse=True))


def get_tetmesh_cell_arrays(tetmesh: UnstructuredGrid) -> np.ndarray:
    return get_cell_arrays(tetmesh.cells)


def get_cell_arrays(cells: np.ndarray) -> np.ndarray:
    """Get the cell arrays from a pyvista.UnstructuredGrid object.
    This function assumes that the cells are tetrahedrons.

    -> tetmesh.cells will return a numpy array of shape (n_cells, 5),
    where the first column is the number of vertices in the cell,
    and the remaining columns are the indices of the vertices.

    returns a numpy array of shape (n_cells, 4), where each row is a cell,
    """
    return np.array(np.hsplit(cells, cells.size / 5)).reshape(-1, 5)[:, 1:]


def n_related_objects(objects_npoints, cell) -> np.ndarray:
    """Check if a cell belongs to a single object or not.

    Parameters:
    objects_npoints (List[int]): A list of the number of points for each object.
    cell (ndarray): an array of shape (4,) with the indices of the points in the cell in any order. shape: [id0, id1, id2, id3]

    Returns:
    np.ndarray: An array of shape (4,) with the number of objects each cell belongs to in the same order as the input ids.
    """
    # Get a numpy array of the number of points for each object
    # assert len(objects_npoints) > 0, "The list of objects should not be empty."
    # assert len(cell) == 4, "The number of points in the objects should be equal to the number of points in the cell."

    # Calculate the cumulative sum to get the ranges
    ranges = np.cumsum(objects_npoints)

    # Check which objects the points belong to
    point_objects = np.searchsorted(ranges, cell, side='right')

    return point_objects


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

    """
    shape = np.shape(points)
    assert shape[1] == 3, "The points should be 3D."
    assert 3 <= shape[0] <= 4, "The number of points should be either 3 or 4."

    v0 = points[1] - points[0]
    v1 = points[2] - points[0]
    normal = np.cross(v0, v1)

    normal *= -2 * (np.dot(normal, v_i - points[0]) < 0) + 1

    unit_normal = normal / np.linalg.norm(normal)

    return unit_normal


def create_face_normal(face_vertices: np.ndarray, obj_point: np.ndarray):
    """Create a face normal from the face vertices and the object point.

    Parameters:
    face_vertices (ndarray): an array of shape (3, 3) with the coordinates of the vertices of the face.
    obj_point (ndarray): an array of shape (3,) with the coordinates of the object point.

    Returns:
    vertex_face_normal (ndarray): an array of shape (2,3) with the coordinates of the related point[0], the first point on the face in [1] and the normal in [2].
    """
    assert np.shape(face_vertices) == (3,3), "The face vertices should be 3D."
    vertex_face_normal = np.empty((3, 3), dtype=np.float64)
    vertex_face_normal[0] = obj_point
    vertex_face_normal[1] = face_vertices[0]
    vertex_face_normal[2] = compute_face_unit_normal(face_vertices, obj_point)
    return vertex_face_normal
