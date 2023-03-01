# %%
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

import irregular_object_packing.packing.chordal_axis_transform as cat
from importlib import reload

# reload(cat)

IropData = cat.IropData


# Define the objective function to be maximized
def objective(x):
    f, theta, t = x[0], x[1:4], x[4:]
    return -f  # maximize f


def compute_face_normal(points, v_i):
    # Compute the cross product of two vectors that lie on the face
    # Compute the cross product of two vectors that lie on the face
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Calculate the vectors between the centroid and all other points
    vectors = np.array(points) - centroid

    # Calculate the cross product of two of the vectors to get the normal vector
    normal = np.cross(vectors[0], vectors[1])

    # Check if the other point is on the same side of the plane as the normal vector
    if np.dot(normal, np.array(v_i) - centroid) < 0:
        normal *= -1

    return normal


def construct_transform_matrix(x, translation=True):
    f, theta, t = x[0], x[1:4], x[4:]
    T = np.eye(4)  # identity transformation matrix
    R = rotation_matrix(*theta)
    f = f ** (1 / 3)
    S = np.diag([f, f, f])
    T[:3, :3] = R @ S  # compute rotation matrix and transpose
    if translation:
        T[:3, 3] = t

    # R = rotation_matrix(*theta)
    # T = np.eye(4)  # identity transformation matrix
    # T[3, 3] = f  # ** (1 / 3) # TODO: THIS is wrong. The matrices can be multiplyed, but are not commutative
    # T[:3, :3] = R  # compute rotation matrix
    # # T[0, 0], T[1, 1], T[2, 2], T[3, 3] = f, f, f, f  # **1/3  # set scaling factor
    # T[:3, 3] = t  # set translation vector
    return T


def transform_v(v_i, T: np.ndarray):
    transformed_v_i = T @ np.hstack((v_i, 1))  # transform v_i

    # Normalize the resulting homogeneous coordinate vector to get the transformed 3D coordinate
    norm_v = transformed_v_i[:-1] / transformed_v_i[-1]

    return norm_v


# Define the constraint function
def constraint_single_point(v_i, transform_matrix, facets, points: dict, obj_coord=np.zeros(3)):
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
        values.append(condition)

    values

    return values


def constraint_multiple_points(
    tf_arr: list[float],
    v: list[int],
    facets_sets: dict[list[int]],
    points: dict,
    obj_coords=np.zeros(3),
    pbar=False,
):
    transform_matrix = construct_transform_matrix(tf_arr)

    constraints = []  # list of constraints
    for i, v_i in tqdm(enumerate(v), disable=not pbar):
        constraints += constraint_single_point(v_i, transform_matrix, facets_sets[i], points, obj_coords)

    return constraints


def constraints_from_dict(tf_arr: list[float], obj_id: int, irop_data: IropData):
    # item will be in the form (vi, [facet1, facet2, ...])
    items = irop_data.cat_faces[obj_id].items()

    v, facets_sets = [*zip(*items)]
    return constraint_multiple_points(tf_arr, v, facets_sets, irop_data.points, irop_data.object_coords[obj_id])


# Define a function to compute the rotation matrix from a rotational vector
def rotation_matrix(rx, ry, rz):
    cx, cy, cz = np.cos(rx), np.cos(ry), np.cos(rz)
    sx, sy, sz = np.sin(rx), np.sin(ry), np.sin(rz)
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return R_x @ R_y @ R_z


def test_nlcp():
    # Define the set of facets and the point v_i
    # facets = [np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]
    import numpy as np

    # Define points
    points = {
        1: np.array([-1, -1, 1], dtype=np.float64),
        2: np.array([1, -1, 1], dtype=np.float64),
        3: np.array([1, 1, 1], dtype=np.float64),
        4: np.array([-1, 1, 1], dtype=np.float64),
        5: np.array([-1, -1, 0], dtype=np.float64),
        6: np.array([1, -1, 0], dtype=np.float64),
        7: np.array([1, 1, 0], dtype=np.float64),
        8: np.array([-1, 1, 0], dtype=np.float64),
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

    # Define vectors
    vectors = {}

    # Define the initial guess for the variables
    # x0 = np.array([0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    x0 = np.array([0.9, 0.01, 0.01, 0.01, 0, 0, 0])

    v = [9, 10, 11]

    data = IropData([])
    data.points = {k: tuple(v) for k, v in points.items()}
    data.point_ids = {tuple(v): k for k, v in points.items()}
    data.cat_cells = {0: facets}
    data.cat_faces = {0: {v_i: facets for v_i in v}}

    # Define the bounds for the variables
    r_bound = (-1 / 12 * np.pi, 1 / 12 * np.pi)
    t_bound = (0, 0)
    bounds = [(0.1, None), r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]
    # constraint_dict = {"type": "ineq", "fun": constraint_multiple_points, "args": (v, [facets, facets, facets])}
    constraint_dict = {
        "type": "ineq",
        "fun": constraints_from_dict,
        "args": (
            0,
            data,
        ),
    }
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict)
    ## %%
    v_1 = data.point(v[0])
    v_2 = data.point(v[1])
    v_3 = data.point(v[2])
    # Print the results
    print("Optimal solution:")
    print(res.x)
    print("Maximum scaling factor:")
    print(-res.fun)
    print("resulting vectors:")
    print(transform_v(v_1, res.x))
    print(transform_v(v_2, res.x))
    print(transform_v(v_3, res.x))

    ## %%
    # Create a 3D plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the faces with opacity = 0.5
    for face in facets:
        face = data.get_face(face)

        collection = Poly3DCollection([face], alpha=0.5, facecolor="blue", edgecolor="black")
        ax.add_collection(collection)
    pairs = [
        [v_1, transform_v(v_1, res.x)],
        [v_2, transform_v(v_2, res.x)],
        [v_3, transform_v(v_3, res.x)],
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
