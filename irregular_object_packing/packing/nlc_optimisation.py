# %%
import numba
import numpy as np
from numba import jit
from numba.typed.typeddict import Dict
from numba.typed.typedlist import List
from numpy import ndarray
from scipy.optimize import minimize

from irregular_object_packing.cat.chordal_axis_transform import CatData
from irregular_object_packing.packing.utils import (
    print_transform_array,
)

# Define the objective function to be maximized
NO_PYTHON = True


@jit(nopython=NO_PYTHON, debug=True)
def objective(x):
    """ Objective function to be minimized.
    returns negative of f(x) so its maximized instead of minimized.
        f, _theta, _t = x[0], x[1:4], x[4:]
    """
    return -x[0]  # maximize f


@jit(numba.float64[:, :](numba.float64[:]), nopython=NO_PYTHON, debug=True, fastmath=True)
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

    cosine = np.cos(theta)
    sine = np.sin(theta)

    R = np.array([
        [cosine[1] * cosine[2], -cosine[1] * sine[2] * cosine[0] + sine[1] * sine[0], cosine[1] * sine[2] * sine[0] + sine[1] * cosine[0]],
        [sine[2], cosine[2] * cosine[0], -cosine[2] * sine[0]],
        [-sine[1] * cosine[2], sine[1] * sine[2] * cosine[0] + cosine[1] * sine[0], -sine[1] * sine[2] * sine[0] + cosine[1] * cosine[0]]
    ], dtype=np.float64)

    return R.copy()


@ jit(numba.float64[: , :](numba.float64, numba.float64[:], numba.float64[:]), nopython=NO_PYTHON, debug=True, fastmath=True)
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

    return np.copy(T)


def construct_transform_matrix_from_array(tf_array):
    return construct_transform_matrix(tf_array[0], tf_array[1:4], tf_array[4:])


@ jit(numba.float64[:](numba.float64[:], numba.float64[: , :]), nopython=NO_PYTHON, debug=True)
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


@ jit(nopython=NO_PYTHON, debug=True)
def local_constraint_for_point(
    v_id, transform_matrix, faces: np.ndarray[int], normals: np.ndarray[float], points, obj_coord=np.zeros(3), padding=0.0


):
    """Compute conditions for a single point.
    NOTE: the point has already been rotated and
    scaled to according to the last iteration. Therefore the bounds for
    rotation, scaling and translation are around zero

    Parameters
    ----------
    v_id : int
        ID of the point for which the conditions are computed.
    transform_matrix: numpy.ndarray
        Matrix that transforms a point.
    faces: ndarray[int]
        List of lists of point IDs.
    face_normals: ndarray[float]
        List of face normals.
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
    v_i = points[v_id] - obj_coord
    # apply the transformation matrix to the point
    transformed_v_i = transform_v(v_i, transform_matrix)

    constraints = np.empty(len(faces), dtype=np.float64)
    for i in range(len(faces)):
        face_p_ids = faces[i]
        n_face = normals[i]

        # translate the points of the faces to the local coordinate system of the object
        face_coords = [points[p_id] - obj_coord for p_id in face_p_ids]

        # NOTE: Any point on the surface of the object can be used as q_j
        q_j = face_coords[0]  # q_j = np.mean(face_coords, axis=0) -> not necessary

        # NOTE: The normal vector has unit length [./utils.py:196], no need to divide
        condition = np.dot(transformed_v_i - q_j, n_face)

        # Return negative value if point is inside surface plus margin,
        dist = condition - padding
        constraints[i] = dist

    return constraints


@jit(nopython=NO_PYTHON, debug=True)
def local_constraint_multiple_points(
    tf_arr: ndarray[float],
    v: ndarray[int],
    sets_of_faces: list[ndarray[int]],
    sets_of_normals: list[ndarray[float]],
    points: Dict,
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
    constraints = List()  # list of constraints

    for i, vi in enumerate(v):
        faces = sets_of_faces[i]  # assumes each face has 3 points
        normals = sets_of_normals[i]
        constraints.append(
            local_constraint_for_point(
                vi, transform_matrix, faces, normals, points, obj_coords, padding=padding
            )
        )
    return [item for sublist in constraints for item in sublist]

    # return np.concatenate(constraints).flat


def make_dict_typed(d: dict) -> Dict:
    """Convert a dictionary to a typed dictionary."""
    nd = Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64[:]
    )

    for k, v in d.items():
        nd[k] = np.array(v, dtype='f8')

    return nd


def local_constraints_from_cat(
    tf_arr: list[float], obj_id: int, cat_data: CatData, padding=0.0
):
    """Compute the conditions for one object.
    item will be in the form:
    - [(vi, [facet_j, facet_j+1, ...]), (vi+1, [facet_k, facet_k+1, ...)]

    """

    items = cat_data.cat_faces[obj_id].items()
    points = make_dict_typed(cat_data.points)

    v, faces = [*zip(*items, strict=True)]

    _, face_normals = [*zip(*cat_data.cat_normals[obj_id].items(), strict=True)]
    assert len(v) == len(faces) == len(face_normals)

    # Convert to numba compatible datatypes
    point_list = List()
    face_normals_list = List()
    faces_list = List()

    for i in range(len(faces)):
        if len(faces[i]) == 0:
            continue
        faces_3 = [f[:3] for f in faces[i]]
        point_list.append(v[i])
        faces_list.append(np.array(faces_3, dtype=np.int64))
        face_normals_list.append(np.array(face_normals[i], dtype=np.float64))

    return local_constraint_multiple_points(
        np.array(tf_arr, dtype=np.float64),
        np.array(point_list, dtype=np.int64),
        faces_list,
        face_normals_list,
        points,
        cat_data.object_coords[obj_id],
        padding,
    )


def optimal_local_transform(
    obj_id,
    cat_data,
    scale_bound=(0.1, None),
    max_angle=1 / 12 * np.pi,
    max_t=None,
    padding=0.0,
):
    """Computes the optimal local transform for a given object id.

    This will return the transformation parameters that maximises scale with
    respect to a local coordinate system of the object. This is possible due to
    the `obj_coords`.
    """

    r_bound = (-max_angle, max_angle)
    t_bound = (-max_t if max_t is not None else None, max_t)
    bounds = [scale_bound, r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]
    x0 = np.array([scale_bound[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # randomize initial guess based on max_angle
    x0[1:4] = np.random.uniform(-max_angle, max_angle, size=3)

    constraint_dict = {
        "type": "ineq",
        "fun": local_constraints_from_cat,
        "args": (
            obj_id,
            cat_data,
            padding,
        ),
    }
    res = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict
    )
    return res.x


def compute_optimal_growth(obj_id, previous_tf_array, max_scale, scale_bound, max_angle, max_t, padding, cat_data):
    tf_arr = optimal_local_transform(
        obj_id=obj_id,
        cat_data=cat_data,
        # scale_bound=scale_bound,
        max_angle=max_angle,
        max_t=max_t,
        padding=padding,
    )

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

    # Define facets (face, normal)
    faces = np.array([
        [1, 2, 3],
        [5, 6, 7],
        [1, 2, 6],
        [2, 3, 7],
        [3, 4, 8],
        [4, 1, 5],
    ], dtype=np.int64)

    face_normals = np.array([
        [0, 0, -1],
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
        [+1, 0, 0],
    ], dtype=np.float64)

    x0 = np.array([1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1], dtype=np.float64)

    v = np.array([9, 10, 11], dtype=np.int64)
    faces_sets = List([faces, faces, faces])
    face_normal_sets = List([face_normals, face_normals, face_normals])

    r_bound = (-1 / 12 * np.pi, 1 / 12 * np.pi)
    t_bound = (0, None)
    bounds = [(0.1, None), r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]

    points = make_dict_typed(points)

    local_constraint_multiple_points(
        x0,
        v,
        faces_sets,
        face_normal_sets,
        points,
        obj_coords=np.array([0, 0, 0], dtype=np.float64),
        padding=0.0,
    )

    constraint_dict = {
        "type": "ineq",
        "fun": local_constraint_multiple_points,
        "args": (
            v,
            faces_sets,
            face_normal_sets,
            points,
            np.array([0, 0, 0], dtype=np.float64),
            0.1,
        ),
    }

    res = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict
    )
    T = construct_transform_matrix(res.x[0], res.x[1:4], res.x[4:7])

    # Print the results
    print("Optimal solution:")
    print_transform_array(res.x)
    # print("resulting vectors:")

    print(transform_v(points[9], T))
    print(transform_v(points[10], T))
    print(transform_v(points[11], T))

    # Create a 3D plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the faces with opacity = 0.5
    # quick
    def get_face_coords(face, points):
        return [points[p_id] for p_id in face]

    for face in faces:
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
