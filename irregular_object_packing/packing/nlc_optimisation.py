# %%
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

import irregular_object_packing.packing.chordal_axis_transform as cat
from irregular_object_packing.packing.chordal_axis_transform import CatData

from importlib import reload


# Define the objective function to be maximized
def objective(x):
    f, theta, t = x[0], x[1:4], x[4:]
    return -f  # maximize f


def compute_face_normal(points, v_i):
    """Compute the normal vector of a face.

    The normal vector is the cross product of two vectors of the face with
    the centroid of the face as the origin.

    Parameters
    ----------
    points : list of tuple of float
        The points of the face.
    v_i : tuple of float
        A point on the plane of the face.

    Returns
    -------
    normal : tuple of float
        The normal vector of the face.

    Examples
    --------
    >>> compute_face_normal([(0, 0, 0), (0, 0, 1)], (0, 1, 2))
    (0.0, 1.0, 0.0)
    """
    n_points = len(points)
    assert 3 <= n_points <= 4, "The number of points should be either 3 or 4."

    v0 = points[1] - points[0]
    v1 = points[2] - points[0]
    normal = np.cross(v0, v1)
    if np.dot(normal, v_i - points[0]) < 0:
        normal *= -1
    return normal


def rotation_matrix(rx, ry, rz):
    """Rotation matrix for rotations around the x-, y-, and z-axis.

    Parameters
    ----------
    rx : float
        Rotation angle around the x-axis [rad].
    ry : float
        Rotation angle around the y-axis [rad].
    rz : float
        Rotation angle around the z-axis [rad].

    Returns
    -------
    R : (3,3) array
        Rotation matrix.

    Examples
    --------
    >>> rotation_matrix(0, 0, 0)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> rotation_matrix(np.pi, 0, 0)
    array([[-1.,  0.,  0.],
           [ 0., -1.,  0.],
           [ 0.,  0.,  1.]])
    >>> rotation_matrix(0, np.pi, 0)
    array([[-1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0., -1.]])
    >>> rotation_matrix(0, 0, np.pi)
    array([[ 1.,  0.,  0.],
           [ 0., -1.,  0.],
           [ 0.,  0., -1.]])
    """
    cx, cy, cz = np.cos(rx), np.cos(ry), np.cos(rz)
    sx, sy, sz = np.sin(rx), np.sin(ry), np.sin(rz)
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return R_x @ R_y @ R_z


def construct_transform_matrix(x, translation=True):
    """Transforms parameters to transformation matrix.

    Parameters
    ----------
    x : (7,) ndarray
        Parameters of the transformation matrix.
    translation : bool, optional
        Whether to include translation in transformation matrix, by
        default True.

    Returns
    -------
    T : (4, 4) ndarray
        Transformation matrix.

    Examples
    --------
    >>> x = np.array([1, 0, 0, 0, 0, 0, 0])
    >>> construct_transform_matrix(x)
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    """
    # Extract parameters
    f, theta, t = x[0], x[1:4], x[4:]
    # Construct identity transformation matrix
    T = np.eye(4)
    # Compute rotation matrix
    R = rotation_matrix(*theta)
    # Scale rotation matrix
    f = f ** (1 / 3)
    S = np.diag([f, f, f])
    # Compute final transformation matrix
    T[:3, :3] = R @ S
    if translation:
        T[:3, 3] = t

    return T


def transform_v(v_i, T: np.ndarray):
    """Transforms vector v_i with transformation matrix T.

    Parameters
    ----------
    v_i : np.ndarray
        vector to be transformed
    T : np.ndarray
        4x4 transformation matrix

    Returns
    -------
    np.ndarray
        transformed vector

    Examples
    --------
    >>> import numpy as np
    >>> T = np.eye(4)  # identity matrix
    >>> v_i = np.array([0, 0, 0])
    >>> transform_v(v_i, T)
    array([0., 0., 0.])
    """
    transformed_v_i = T @ np.hstack((v_i, 1))  # transform v_i

    # Normalize the resulting homogeneous coordinate vector to get the transformed 3D coordinate
    norm_v = transformed_v_i[:-1] / transformed_v_i[-1]

    return norm_v


def constraint_single_point_margin(
    v_i, transform_matrix, facets, points: dict, obj_coord=np.zeros(3), margin=None
):
    """Compute conditions for a single point.

    Parameters
    ----------
    v_i : int
        ID of the point for which the conditions are computed.
    transform_matrix: numpy.ndarray
        Matrix that transforms a point.
    facets: list
        List of lists of point IDs.
    points: dict
        Dictionary of point IDs and their coordinates.
    obj_coord : numpy.ndarray
        Coordinates of the object center.

    Returns
    -------
    list
        List of conditions for a single point.
    """
    v_i = np.array(points[v_i]) - obj_coord

    transformed_v_i = transform_v(v_i, transform_matrix)  # transform v_i

    values = []
    for facet_p_ids in facets:
        facet = [np.array(points[p_id]) - obj_coord for p_id in facet_p_ids]

        n_j = compute_face_normal(facet, v_i)

        # normals = facet[1:]  # remaining points in facet are normals
        # for q_j in facet[:1]:
        q_j = facet[0]
        condition = np.dot(transformed_v_i - q_j, n_j) / np.linalg.norm(n_j)
        if margin is None:
            values.append(condition)

        else:
            dist = abs(condition) - margin
            if dist < 0:
                dist = 0

            # Return negative value if point is inside surface plus margin, positive value otherwise
            if condition < 0:
                values.append(-dist)
            else:
                values.append(dist)

    return values


def constraint_single_point_normal(
    v_i, transform_matrix, facets, normals, points: dict, obj_coord=np.zeros(3), margin=None
):
    """Compute conditions for a single point.

    Parameters
    ----------
    v_i : int
        ID of the point for which the conditions are computed.
    transform_matrix: numpy.ndarray
        Matrix that transforms a point.
    facets: list
        List of lists of point IDs.
    points: dict
        Dictionary of point IDs and their coordinates.
    obj_coord : numpy.ndarray
        Coordinates of the object center.

    Returns
    -------
    list
        List of conditions for a single point.
    """
    v_i = np.array(points[v_i]) - obj_coord

    transformed_v_i = transform_v(v_i, transform_matrix)  # transform v_i

    values = []
    for i, facet_p_ids in enumerate(facets):
        facet = [np.array(points[p_id]) - obj_coord for p_id in facet_p_ids]

        n_j = normals[i]

        # normals = facet[1:]  # remaining points in facet are normals
        # for q_j in facet[:1]:
        q_j = facet[0]
        condition = np.dot(transformed_v_i - q_j, n_j) / np.linalg.norm(n_j)
        if margin is None:
            values.append(condition)

        else:
            dist = abs(condition) - margin
            if dist < 0:
                dist = 0

            # Return negative value if point is inside surface plus margin, positive value otherwise
            if condition < 0:
                values.append(-dist)
            else:
                values.append(dist)

    return values


def constraint_multiple_points(
    tf_arr: list[float],
    v: list[int],
    facets_sets: list[list[int]],
    points: dict,
    obj_coords=np.zeros(3),
    margin=None,
):
    """Compute conditions for a list of point with corresponding facets_sets.

    Parameters
    ----------
    v : list[int]
        ID of the point for which the conditions are computed.
    tf_arr: numpy.ndarray
        array with transformation parameters.
    facets: dict[list]
        list of facets per ID in v defined by lists of point IDs.
    points: dict
        Dictionary of point IDs and their coordinates.
    obj_coord : numpy.ndarray
        Coordinates of the object center.

    Returns
    -------
    list
        List of conditions for all the points.
    """
    transform_matrix = construct_transform_matrix(tf_arr)

    constraints = []  # list of constraints
    for i, v_i in enumerate(v):
        constraints += constraint_single_point_margin(
            v_i, transform_matrix, facets_sets[i], points, obj_coords, margin=margin
        )

    return constraints


def constraints_from_dict(tf_arr: list[float], obj_id: int, cat_data: CatData, margin=None):
    # item will be in the form (vi, [facet1, facet2, ...])
    items = cat_data.cat_faces[obj_id].items()
    # TODO: replace with only keys and then use dict to get faces

    v, facets_sets = [*zip(*items)]
    return constraint_multiple_points(
        tf_arr, v, facets_sets, cat_data.points, cat_data.object_coords[obj_id], margin
    )


def test_nlcp():
    # Define the set of facets and the point v_i
    # facets = [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]
    import numpy as np

    # Define points
    points = {
        # Box of size 2x2x2 centered at the origin
        1: np.array([-1, -1, 1], dtype=np.float64),
        2: np.array([1, -1, 1], dtype=np.float64),
        3: np.array([1, 1, 1], dtype=np.float64),
        4: np.array([-1, 1, 1], dtype=np.float64),
        5: np.array([-1, -1, 0], dtype=np.float64),
        6: np.array([1, -1, 0], dtype=np.float64),
        7: np.array([1, 1, 0], dtype=np.float64),
        8: np.array([-1, 1, 0], dtype=np.float64),
        # points to test
        9: np.array([0, 0, 0.9]),
        10: np.array([0, 0.9, 0.0]),
        11: np.array([0.9, 0.0, 0.0]),
    }

    # Define facets
    facets = [
        np.array([1, 2, 3, 4]),
        np.array([5, 6, 7, 8]),
        np.array([1, 2, 6, 5]),
        np.array([2, 3, 7, 6]),
        np.array([3, 4, 8, 7]),
        np.array([4, 1, 5, 8]),
    ]

    # quick
    def get_face_coords(facet, points):
        return [points[p_id] for p_id in facet]

    x0 = np.array([0.9, 0.01, 0.01, 0.01, 0, 0, 0])

    v = [9, 10, 11]
    facets_sets = [facets, facets, facets]

    r_bound = (-1 / 12 * np.pi, 1 / 12 * np.pi)
    t_bound = (0, 0)
    bounds = [(0.1, None), r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]
    # constraint_dict = {"type": "ineq", "fun": constraint_multiple_points, "args": (v, [facets, facets, facets])}
    constraint_dict = {
        "type": "ineq",
        "fun": constraint_multiple_points,
        "args": (
            v,
            facets_sets,
            points,
            np.array([0, 0, 0]),
            0.1,
        ),
    }

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict)
    T = construct_transform_matrix(res.x)
    ## %%
    # Print the results
    print("Optimal solution:")
    print(res.x)
    print("Maximum scaling factor:")
    print(-res.fun)
    print("resulting vectors:")

    print(transform_v(points[9], T))
    print(transform_v(points[10], T))
    print(transform_v(points[11], T))

    ## %%
    # Create a 3D plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the faces with opacity = 0.5
    for face in facets:
        face = get_face_coords(face, points)

        collection = Poly3DCollection([face], alpha=0.2, facecolor="blue", edgecolor="black")
        ax.add_collection(collection)
    pairs = [
        [points[9], transform_v(points[9], T)],
        [points[10], transform_v(points[10], T)],
        [points[11], transform_v(points[11], T)],
    ]

    # Plot the pairs of points with lines connecting them
    colors = ["r", "g", "b"]  # Different colors for each pair
    for i, pair in enumerate(pairs):
        color = colors[i % len(colors)]  # Cycle through the colors
        ax.plot(*zip(*pair), color=color, marker="o", linestyle="-")

    # Set the plot limits and labels
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.1, 1.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Show the plot
    plt.show()


# test_nlcp()
# %%
