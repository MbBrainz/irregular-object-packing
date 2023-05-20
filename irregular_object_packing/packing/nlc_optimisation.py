# %%
import numpy as np
from numba import float64, jit, prange
from numpy import ndarray
from scipy.optimize import minimize

# Define the objective function to be maximized
NO_PYTHON = True
DEBUG = False


@jit(nopython=NO_PYTHON, debug=DEBUG)
def objective(x):
    """ Objective function to be minimized.
    returns negative of f(x) so its maximized instead of minimized.
        f, _theta, _t = x[0], x[1:4], x[4:]
    """
    return -x[0]  # maximize f


@jit(float64[:, :](float64[:]), nopython=NO_PYTHON, debug=DEBUG, fastmath=True, cache=True)
def rotation_matrix(theta):
    """Rotation matrix for rotations around the x-, y-, and z-axis.

    Parameters
    ----------
    theta: (3,) array of (rx, ry, rz)
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

    cosine = np.cos(theta, dtype=np.float64)
    sine = np.sin(theta, dtype=np.float64)

    R = np.array([
        [cosine[1] * cosine[2], -cosine[1] * sine[2] * cosine[0] + sine[1] * sine[0], cosine[1] * sine[2] * sine[0] + sine[1] * cosine[0]],
        [sine[2], cosine[2] * cosine[0], -cosine[2] * sine[0]],
        [-sine[1] * cosine[2], sine[1] * sine[2] * cosine[0] + cosine[1] * sine[0], -sine[1] * sine[2] * sine[0] + cosine[1] * cosine[0]]
    ], dtype=np.float64)

    return np.ascontiguousarray(R)


@ jit(float64[: , :](float64, float64[:], float64[:]), nopython=NO_PYTHON, debug=DEBUG, fastmath=True, cache=True)
def construct_transform_matrix(f, theta, t):
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
    # Construct identity transformation matrix
    T = np.eye(4, dtype=np.float64)
    # Compute rotation matrix
    R = rotation_matrix(theta)
    # Scale rotation matrix
    f = f ** (1 / 3)
    S = np.diag(np.full(3, f, dtype=np.float64))
    # Compute final transformation matrix
    T[:3, :3] = R @ S
    T[:3, 3] = t

    return np.ascontiguousarray(T)


def construct_transform_matrix_from_array(tf_array):
    return construct_transform_matrix(tf_array[0], tf_array[1:4], tf_array[4:])


@ jit(float64[:](float64[:], float64[: , :]), nopython=NO_PYTHON, debug=DEBUG, fastmath=True, cache=True)
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
    contiguous_vi = np.ones(4, dtype=np.float64)
    contiguous_vi[: 3] = v_i
    transformed_v_i = T @ contiguous_vi  # transform v_i

    # Normalize the resulting homogeneous vector to get the transformed 3D coordinate
    norm_v = transformed_v_i[: 3] / transformed_v_i[3]

    return norm_v


@ jit(nopython=NO_PYTHON, debug=DEBUG, fastmath=True, cache=True)
def local_constraint_for_vertex(
    vertex, face_point, face_normal, transform_matrix, obj_coord=np.zeros(3), padding=0.0


):
    """Compute conditions for a single point.
    NOTE: the point has already been rotated and
    scaled to according to the last iteration. Therefore the bounds for
    rotation, scaling and translation are around zero

    Parameters
    ----------
    p_vertex : np.ndarray[float]
        Point to be checked.
    face_normals : np.ndarray[float]
        List of face normals.
    transform_matrix : np.ndarray[float]
        Transformation matrix shape (4,4).
    obj_coord : np.ndarray[float], optional
        Object coordinate, by default np.zeros(3)

    Returns
    -------
    list
        List of conditions for a single point.
    """
    # translate the point to the local coordinate system of the object
    v_i = vertex - obj_coord
    # apply the transformation matrix to the point
    transformed_v_i = transform_v(v_i, transform_matrix)

    q_j = face_point - obj_coord  # q_j = np.mean(face_coords, axis=0) -> not necessary

    # NOTE: The normal vector has unit length [./utils.py:196], no need to divide
    condition = np.dot((transformed_v_i - q_j), face_normal)

    # Return negative value if point is inside surface plus margin, cond - padding == distance
    constraint = condition - padding
    return constraint


@ jit(nopython=NO_PYTHON, debug=DEBUG)
def local_constraint_vertices(
    tf_arr: ndarray[float],
    vertex_fpoint_fnormal_arr: ndarray[ndarray[float]],
    obj_coords: ndarray[float],
    padding=0.0,
):
    """Compute conditions for a list of point with corresponding facets_sets.

    Parameters
    ----------
    v : list[int]
        ID of the point for which the conditions are computed.
    tf_arr: numpy.ndarray
        array with global transformation parameters.
    sets_of_faces: dict[list]
        np.array in the shape of
    points: dict
        Dictionary of point IDs and their coordinates.
    obj_coord : numpy.ndarray
        Coordinates of the object center.

    Returns
    -------
    list
        List of conditions for all the points.
    """
    transform_matrix = construct_transform_matrix(tf_arr[0], tf_arr[1:4], tf_arr[4:])
    size = len(vertex_fpoint_fnormal_arr)
    constraints = np.empty(size, dtype=np.float64)
    for i in prange(size):
        constraints[i] = local_constraint_for_vertex(vertex_fpoint_fnormal_arr[i][0], vertex_fpoint_fnormal_arr[i][1], vertex_fpoint_fnormal_arr[i][2], transform_matrix, obj_coords, padding)
    return constraints


def compute_optimal_growth(previous_tf_array,  obj_coord, vertex_fpoint_normal_arr, padding, max_scale, scale_bound, max_angle, max_t,):
    r_bound = (-max_angle, max_angle)
    t_bound = (-max_t if max_t is not None else None, max_t)
    bounds = [scale_bound, r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]
    x0 = np.array([scale_bound[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # randomize initial guess based on max_angle
    x0[1:4] = np.random.uniform(-max_angle, max_angle, size=3)

    constraint_dict = {
        "type": "ineq",
        "fun": local_constraint_vertices,
        "args": (
            vertex_fpoint_normal_arr,
            obj_coord,
            padding,
        ),
    }
    res = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict
    )
    tf_arr = res.x

    new_tf = previous_tf_array + tf_arr
    new_scale = previous_tf_array[0] * tf_arr[0]
    if new_scale > max_scale:
        new_scale = max_scale

    new_tf[0] = new_scale
    return new_tf


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
        9: np.array([0, 0, 0.9], dtype=np.float64),
        10: np.array([0, 0.9, 0.0], dtype=np.float64),
        11: np.array([0.9, 0.0, 0.0], dtype=np.float64),
    }

    print(len(points))

    # Define facets (face, normal)
    np.array([
        [1, 2, 3],
        [5, 6, 7],
        [1, 2, 6],
        [2, 3, 7],
        [3, 4, 8],
        [4, 1, 5],
    ], dtype=np.int64)

    np.array([
        [0, 0, -1],
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [+1, 0, 0],
    ], dtype=np.float64)

    np.array([1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1], dtype=np.float64)

    np.array([9, 10, 11], dtype=np.int64)

    # Define constraints
    vertex = np.array([0, 0, 0.9])
    fpoint = np.array([0, 0, 1], dtype=np.float64)
    fnormal = np.array([0, 0, -1], dtype=np.float64)

    constraint = local_constraint_for_vertex(vertex, fpoint, fnormal, np.eye(4), np.array([0, 0, 0], dtype=np.float64), 0.0)
    print(constraint)

    vertex_fpoint_fnormal_arr = np.array([
        [vertex, fpoint, fnormal],
        [vertex, fpoint, fnormal],
        [vertex, fpoint, fnormal],
    ], dtype=np.float64)

    tf_array = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    constraints = local_constraint_vertices(tf_array, vertex_fpoint_fnormal_arr, np.array([0, 0, 0], dtype=np.float64), 0.0)
    print(constraints)


#     points = make_dict_typed(points)

#     local_constraint_multiple_points(
#         x0,
#         v,
#         faces_sets,
#         face_normal_sets,
#         points,
#         obj_coords=np.array([0, 0, 0], dtype=np.float64),
#         padding=0.0,
#     )

#     constraint_dict = {
#         "type": "ineq",
#         "fun": local_constraint_multiple_points,
#         "args": (
#             v,
#             faces_sets,
#             face_normal_sets,
#             points,
#             np.array([0, 0, 0], dtype=np.float64),
#             0.1,
#         ),
#     }

#     res = minimize(
#         objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict
#     )
#     T = construct_transform_matrix(res.x[0], res.x[1:4], res.x[4:7])

#     # Print the results
#     print("Optimal solution:")
#     print_transform_array(res.x)
#     # print("resulting vectors:")

#     print(transform_v(points[9], T))
#     print(transform_v(points[10], T))
#     print(transform_v(points[11], T))

#     # Create a 3D plot
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     # Plot the faces with opacity = 0.5
#     # quick
#     def get_face_coords(face, points):
#         return [points[p_id] for p_id in face]

#     for face in faces:
#         face = get_face_coords(face, points)

#         collection = Poly3DCollection(
#             [face], alpha=0.2, facecolor="blue", edgecolor="black"
#         )
#         ax.add_collection(collection)
#     pairs = [
#         [points[9], transform_v(points[9], T)],
#         [points[10], transform_v(points[10], T)],
#         [points[11], transform_v(points[11], T)],
#     ]

#     # Plot the pairs of points with lines connecting them
#     colors = ["r", "g", "b"]  # Different colors for each pair
#     for i, pair in enumerate(pairs):
#         color = colors[i % len(colors)]  # Cycle through the colors
#         ax.plot(*zip(*pair, strict=True), color=color, marker="o", linestyle="-")

#     # Set the plot limits and labels
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     ax.set_zlim(-0.1, 1.1)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")

#     # Show the plot
#     # plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     test_nlcp_facets()
#     pass

# # %%
test_nlcp_facets()

# # %%

# %%
