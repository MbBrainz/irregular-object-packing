# %%
import numpy as np
from scipy.optimize import minimize

from irregular_object_packing.packing.chordal_axis_transform import CatData
from irregular_object_packing.packing.utils import (
    print_transform_array,
)


# Define the objective function to be maximized
def objective(x):
    f, _theta, _t = x[0], x[1:4], x[4:]
    return -f  # maximize f


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

    # Normalize the resulting homogeneous vector to get the transformed 3D coordinate
    norm_v = transformed_v_i[:-1] / transformed_v_i[-1]

    return norm_v


def local_constraint_for_point_with_padding(
    v_id, transform_matrix, faces, points: dict, obj_coord=np.zeros(3), padding=0.0
):
    """Compute conditions for a single point.

    Parameters
    ----------
    v_id : int
        ID of the point for which the conditions are computed.
    transform_matrix: numpy.ndarray
        Matrix that transforms a point.
    faces: list
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
    # translate the point to the local coordinate system of the object
    # NOTE: This will be a point that has already been rotated and
    # scaled to according to the last iteration. Therefore the bounds for
    # rotation, scaling and translation are around zero
    v_i = np.array(points[v_id]) - obj_coord
    # apply the transformation matrix to the point
    transformed_v_i = transform_v(v_i, transform_matrix)

    constraints = []
    for _i, (face_p_ids, n_face) in enumerate(faces):
        # translate the points of the faces to the local coordinate system of the object
        face_coords = [np.array(points[p_id]) - obj_coord for p_id in face_p_ids]

        # NOTE: Any point on the surface of the object can be used as q_j
        q_j = face_coords[0] # q_j = np.mean(face_coords, axis=0) -> not necessary

        # NOTE: The normal vector has unit length [./utils.py:196], no need to divide
        condition = np.dot(transformed_v_i - q_j, n_face)


        # Return negative value if point is inside surface plus margin,
        dist = condition - padding
        constraints.append(dist)

        # NOTE: alternative implementation: (currently allows point inside margin only)
        # MARGIN: Compute distance to surface
        # dist = abs(condition) - padding
        # if dist < 0:
        #     dist = 0
        # # positive value otherwise
        # if condition < 0:
        #     constraints.append(-dist)
        # else:
        #     constraints.append(dist)
        # # constraints.append(condition)
    return constraints


def local_constraint_for_point(
    v_id, transform_matrix, faces, points: dict, obj_coord=np.zeros(3), padding=0.0
):
    """Compute conditions for a single point relative to the local coordinate system of
    the object. This is done by translating `v_i` and the points of the facets to the
    local coordinate system before computing the optimization conditions.

    Parameters
    ----------
    v_id : int
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
        List of constraints for a single point.
    """

    # translate the point to the local coordinate system of the object
    # NOTE: This will be a point that has already been rotated and
    # scaled to according to the last iteration. Therefore the bounds for
    # rotation, scaling and translation are around zero
    v_i = np.array(points[v_id]) - obj_coord
    # apply the transformation matrix to the point
    transformed_v_i = transform_v(v_i, transform_matrix)

    constraints = []
    for _i, (face_p_ids, n_face) in enumerate(faces):
        # translate the points of the facet to the local coordinate system of the object
        face_coords = [np.array(points[p_id]) - obj_coord for p_id in face_p_ids]
        q_j = face_coords[
            0
        ]  # first point in facet is q_j (can be any point of the facet)
        q_j = np.mean(face_coords, axis=0)
        condition = np.dot(transformed_v_i - q_j, n_face) / np.linalg.norm(n_face)
        constraints.append(condition)

    return constraints


def local_constraint_multiple_points(
    tf_arr: list[float],
    v: list[int],
    sets_of_faces: list[list[int]],
    points: dict,
    obj_coords,
    padding=0.0,
):
    """Compute conditions for a list of point with corresponding facets_sets.

    Parameters
    ----------
    v : list[int]
        ID of the point for which the conditions are computed.
    tf_arr: numpy.ndarray
        array with global transformation parameters.
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
    constraint_func = local_constraint_for_point
    if padding != 0.0:
        constraint_func = local_constraint_for_point_with_padding
    transform_matrix = construct_transform_matrix(tf_arr)
    constraints = []  # list of constraints
    for i, v_i in enumerate(v):
        constraints += constraint_func(
            v_i, transform_matrix, sets_of_faces[i], points, obj_coords, padding=padding
        )

    return constraints


def local_constraints_from_cat(
    tf_arr: list[float], obj_id: int, cat_data: CatData, padding=0.0
):
    """Compute the conditions for one object."""
    # item will be in the form

    # [(vi, [facet_j, facet_j+1, ...]), (vi+1, [facet_k, facet_k+1, ...)]
    items = cat_data.cat_faces[obj_id].items()

    v, sets_of_faces = [*zip(*items, strict=True)]
    return local_constraint_multiple_points(
        tf_arr,
        v,
        sets_of_faces,
        cat_data.points,
        cat_data.object_coords[obj_id],
        padding,
    )

# -----------------------------------------------------------------------------
# Visual Tests
# -----------------------------------------------------------------------------
def test_nlcp_facets():
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

    # Define facets (face, normal)
    facets = [
        (np.array([1, 2, 3, 4]), np.array([0, 0, -1])),
        (np.array([5, 6, 7, 8]), np.array([0, 0, 1])),
        (np.array([1, 2, 6, 5]), np.array([0, 1, 0])),
        (np.array([2, 3, 7, 6]), np.array([-1, 0, 0])),
        (np.array([3, 4, 8, 7]), np.array([0, -1, 0])),
        (np.array([4, 1, 5, 8]), np.array([+1, 0, 0])),
    ]

    # quick
    def get_face_coords(facet, points):
        return [points[p_id] for p_id in facet[0]]

    x0 = np.array([1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1])

    v = [9, 10, 11]
    facets_sets = [facets, facets, facets]

    r_bound = (-1 / 12 * np.pi, 1 / 12 * np.pi)
    t_bound = (0, None)
    bounds = [(0.1, None), r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]

    constraint_dict = {
        "type": "ineq",
        "fun": local_constraint_multiple_points,
        "args": (
            v,
            facets_sets,
            points,
            np.array([0, 0, 0]),
            0.1,
        ),
    }

    res = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict
    )
    T = construct_transform_matrix(res.x)
    ## %%
    # Print the results
    print("Optimal solution:")
    print_transform_array(res.x)
    # print("resulting vectors:")

    print(transform_v(points[9], T))
    print(transform_v(points[10], T))
    print(transform_v(points[11], T))

    ## %%
    # Create a 3D plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the faces with opacity = 0.5
    for face in facets:
        face = get_face_coords(face, points)

        collection = Poly3DCollection(
            [face], alpha=0.2, facecolor="blue", edgecolor="black"
        )
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
        ax.plot(*zip(*pair, strict=True), color=color, marker="o", linestyle="-")

    # Set the plot limits and labels
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.1, 1.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Show the plot
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_nlcp_facets()
    pass

# %%
# test_nlcp_facets()

# %%
