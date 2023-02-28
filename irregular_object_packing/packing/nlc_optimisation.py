# %%
import numpy as np
from scipy.optimize import minimize


# Define the objective function to be maximized
def objective(x):
    f, theta, t = x[0], x[1:4], x[4:]
    return -f  # maximize f


def compute_face_normal(points, v_i):
    # Compute the cross product of two vectors that lie on the face
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    # Normalize the normal vector
    normal /= np.linalg.norm(normal)

    if np.dot(normal, v_i - v1) < 0:
        normal *= -1

    return normal


def transform_v(v_i, x):
    f, theta, t = x[0], x[1:4], x[4:]
    R = rotation_matrix(*theta)
    T = np.eye(4)  # identity transformation matrix
    T[3, 3] = f  # ** (1 / 3)
    T[:3, :3] = R  # compute rotation matrix
    # T[0, 0], T[1, 1], T[2, 2], T[3, 3] = f, f, f, f  # **1/3  # set scaling factor
    T[:3, 3] = t  # set translation vector

    transformed_v_i = T @ np.hstack((v_i, 1))  # transform v_i

    # Normalize the resulting homogeneous coordinate vector to get the transformed 3D coordinate
    norm_v = transformed_v_i[:-1] / transformed_v_i[-1]

    return norm_v


# Define the constraint function
def constraint_single_point(x, facets, v_i):
    transformed_v_i = transform_v(v_i, x)  # transform v_i
    values = []
    for facet in facets:
        n_j = compute_face_normal(facet, v_i)

        # normals = facet[1:]  # remaining points in facet are normals
        for q_j in facet[:1]:
            condition = np.dot(transformed_v_i - q_j, n_j) / np.linalg.norm(n_j)
            values.append(condition)

    values

    return np.array(values)


def constraint_multiple_points(x, v, facets_sets):
    constr = []  # list of constraints
    for i, v_i in enumerate(v):
        constr.append(constraint_single_point(x, facets_sets[i], v_i))

    return np.array(constr).flatten()


def constraints_from_dict(x, cat_faces):
    v, facets_sets = [*zip(*cat_faces.items())]
    return constraint_multiple_points(x, v, facets_sets)


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
    facets = [
        np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64),
        np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float64),
        np.array([[-1, -1, 1], [1, -1, 1], [-1, -1, 0], [1, -1, 0]], dtype=np.float64),
        np.array([[1, -1, 0], [1, -1, 1], [1, 1, 1], [1, 1, 0]], dtype=np.float64),
        np.array([[-1, -1, 1], [-1, -1, 0], [-1, 1, 0], [-1, 1, 1]], dtype=np.float64),
        np.array([[-1, 1, 0], [1, 1, 0], [1, 1, 1], [-1, 1, 1]], dtype=np.float64),
    ]

    # Define the initial guess for the variables
    x0 = np.array([0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    v_1 = np.array([0, 0, 0.9])

    transform_v(v_1, x0)

    # Define the bounds for the variables
    r_bound = (-1 / 4 * np.pi, 1 / 4 * np.pi)
    t_bound = (0, 1)
    bounds = [(0.1, None), r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]

    # Define the constraints for the optimization problem
    constraint_dict = {"type": "ineq", "fun": constraint_single_point, "args": (facets, v_1)}

    # initial guess
    init_res = constraint_single_point(x0, facets, v_1)

    # Solve the optimization problem using the SQP method
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict)

    # Print the results
    print("Optimal solution:")
    print(res.x)
    print("Maximum scaling factor:")
    print(-res.fun)
    print("resulting vector:")
    print(transform_v(v_1, res.x))

    # %%
    # Define the bounds for the variables
    r_bound = (-1 / 12 * np.pi, 1 / 4 * np.pi)
    t_bound = (0, 1)
    bounds = [(0.1, None), r_bound, r_bound, r_bound, t_bound, t_bound, t_bound]

    # %%
    v_2 = np.array([0, 0.9, 0.0])
    v_3 = np.array([0.9, 0.0, 0.0])
    cat_faces = {tuple(v_1): facets, tuple(v_2): facets, tuple(v_3): facets}
    v = [v_1, v_2, v_3]

    # constraint_dict = {"type": "ineq", "fun": constraint_multiple_points, "args": (v, [facets, facets, facets])}
    constraint_dict = {"type": "ineq", "fun": constraints_from_dict, "args": (cat_faces,)}
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraint_dict)
    # %%
    # Print the results
    print("Optimal solution:")
    print(res.x)
    print("Maximum scaling factor:")
    print(-res.fun)
    print("resulting vectors:")
    print(transform_v(v_1, res.x))
    print(transform_v(v_2, res.x))
    print(transform_v(v_3, res.x))

    # %%
    # Create a 3D plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the faces with opacity = 0.5
    for face in facets:
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


# %%
