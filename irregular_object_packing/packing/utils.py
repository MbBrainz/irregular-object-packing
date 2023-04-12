# %%
import math

import numpy as np
import pyvista as pv


def sort_points_clockwise(points, start, end):
    # Create normal vector from line start and end points
    start = np.array(start)
    end = np.array(end)

    vector = np.array([end[0] - start[0], end[1] - start[1], end[2] - start[2]])
    norm = np.linalg.norm(vector)
    n = vector / norm

    p = (
        points[0] - points[0].dot(n) * n
    )  # take the first point to compute the first orthogonal vector
    q = np.cross(n, p)

    angles = []
    for point in points:
        t = np.dot(n, np.cross((point - start), p))
        u = np.dot(n, np.cross((point - start), q))
        angles.append(math.atan2(u, t))

    sorted_points = [
        x for _, x in sorted(zip(angles, points, strict=True), key=lambda pair: pair[0])
    ]
    return sorted_points


def test_sort_points_clockwise():
    points = np.array(
        [
            [-1, -1, -16],  # 225º
            [-1, 2, 23],  # 135º
            [1, 1, 1],  # 45º
            [1, -1, 2],  # 315º
        ]
    )

    # represents the normal vector of the x-y plane
    start = [0, 0, 0]
    end = [0, 0, 1]

    expected_points = [
        (1, 1, 1),  # 45º
        (-1, 2, 23),  # 135º
        (-1, -1, -16),  # 225º
        (1, -1, 2),  # 315º
    ]
    sorted_points = sort_points_clockwise(points, start, end)
    print(sorted_points)
    print("-------")
    print(expected_points)


def sort_face_points_by_length(expected_faces):
    sorted_faces = []
    for face in expected_faces:
        sorted_faces.append(sort_points_by_polar_angles(face))
        # sort_face_points_by_length(face))

    return sorted_faces


def sort_facepoints_by_length(face):
    return sorted(face, key=lambda point: point[0] ** 2 + point[1] ** 2 + point[2] ** 2)


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


def plot_shapes(shape1, shape2, shape3, shape4):
    # Create a plotter object
    plotter = pv.Plotter()

    # Define the camera position and view angle for each shape
    camera_pos1 = [0, 0, -1]
    view_up1 = [0, 1, 0]
    camera_pos2 = [1, 0, 0]
    view_up2 = [0, 1, 0]
    camera_pos3 = [0, -1, 0]
    view_up3 = [0, 0, 1]
    camera_pos4 = [-1, 0, 0]
    view_up4 = [0, 1, 0]

    # Add the shapes to the plotter
    plotter.add_mesh(
        shape1,
        show_edges=True,
        cmap="cool",
        camera_position=camera_pos1,
        view_up=view_up1,
    )
    plotter.add_mesh(
        shape2,
        show_edges=True,
        cmap="hot",
        camera_position=camera_pos2,
        view_up=view_up2,
    )
    plotter.add_mesh(
        shape3,
        show_edges=True,
        cmap="inferno",
        camera_position=camera_pos3,
        view_up=view_up3,
    )
    plotter.add_mesh(
        shape4,
        show_edges=True,
        cmap="viridis",
        camera_position=camera_pos4,
        view_up=view_up4,
    )

    # Divide the plot into 4 subplots and display each shape in its own subplot
    plotter.subplot(2, 2, 0)
    plotter.display_mesh(shape1)
    plotter.subplot(2, 2, 1)
    plotter.display_mesh(shape2)
    plotter.subplot(2, 2, 2)
    plotter.display_mesh(shape3)
    plotter.subplot(2, 2, 3)
    plotter.display_mesh(shape4)

    # Show the plot
    plotter.show()


def translation_matrix(x0, x1):
    return np.array(
        [
            [1, 0, 0, x1[0] - x0[0]],
            [0, 1, 0, x1[1] - x0[1]],
            [0, 0, 1, x1[2] - x0[2]],
            [0, 0, 0, 1],
        ]
    )


def distance_squared(p1, p2):
    return np.sum((np.array(p1) - np.array(p2)) ** 2)


def get_max_bounds(bounds):
    x_size, y_size, z_size = [bounds[i + 1] - bounds[i] for i in range(0, 6, 2)]
    return max(x_size, y_size, z_size)


def has_duplicates(points):
    """
    Check if the list of points has duplicates.

    :param points: A list of points (tuples)
    :return: True if there are duplicates, False otherwise
    """
    unique_points = set(points)
    return len(unique_points) != len(points)


def split_quadrilateral_to_triangles(points: list[int], vertices: np.ndarray):
    if len(vertices) != 4 | len(points) != 4:
        raise ValueError("Expected a list of 4 points")
    for p in vertices:
        if not isinstance(p, (list, tuple, np.ndarray)):
            raise ValueError(f"Expected vertex type, but got {p} in {vertices}")
        if len(p) != 3:
            raise ValueError(f"Expected a list of 3D points, but got {p} in {vertices}")
    if has_duplicates(points):
        raise ValueError(f"The list of points contains duplicates {points}")

    # points = [Point(*p) for p in list(zip(point_ids, vertices, strict=True))]
    # points = sort_vertices_clockwise(vertices, points)

    # Compute distances between all pairs of points
    distances = [
        (points[0], points[2], distance_squared(vertices[0], vertices[2])),
        (points[1], points[3], distance_squared(vertices[1], vertices[3])),
    ]

    # Find the pair of points with the longest distance
    diagonal = min(distances, key=lambda x: x[2])
    diagonal = (diagonal[0], diagonal[1])

    # Get the two remaining points
    remaining_points = [p for p in points if p not in diagonal]

    # Form two triangles by connecting the endpoints of the diagonal with the remaining points
    triangle1 = [diagonal[0], diagonal[1], remaining_points[0]]
    triangle2 = [diagonal[0], diagonal[1], remaining_points[1]]

    return [triangle1, triangle2]


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


# %%

# compute_face_normal(np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]]), [0, 1, 2])
# %%


def center_point(points):
    return np.mean(points, axis=0)


def plane_normal(points):
    v1 = np.array(points[1]) - np.array(points[0])
    v2 = np.array(points[2]) - np.array(points[0])
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)


def angle(v1, v2, normal):
    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)
    angle = np.arctan2(np.linalg.norm(cross_product), dot_product)

    if np.dot(normal, cross_product) < 0:
        angle = 2 * np.pi - angle

    return angle


def sort_vertices_clockwise(points, point_ids):
    if len(points) != 4:
        raise ValueError("Expected a list of 4 points")

    center = center_point(points)
    normal = plane_normal(points)

    # Compute the reference direction
    ref_direction = np.array(points[0]) - center

    # Compute the angles between the reference direction and each vertex
    angles = [angle(ref_direction, np.array(p) - center, normal) for p in points]

    # Sort the vertices based on their angles
    sorted_points = [(point, point_id) for (_angle, point, point_id) in sorted(zip(angles, points, point_ids, strict=True), key=lambda x: x[0])]

    return sorted_points
