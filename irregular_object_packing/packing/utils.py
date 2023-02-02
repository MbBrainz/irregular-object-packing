# %%
import math

import numpy as np
import pyvista as pv


def angle_between(point, line_start, line_end):
    """DEPRECATED"""
    line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1], line_end[2] - line_start[2])

    point_vector = (point[0] - line_start[0], point[1] - line_start[1], point[2] - line_start[2])
    dot_product = (
        point_vector[0] * line_vector[0] + point_vector[1] * line_vector[1] + point_vector[2] * line_vector[2]
    )
    point_magnitude = math.sqrt(point_vector[0] ** 2 + point_vector[1] ** 2 + point_vector[2] ** 2)
    line_magnitude = math.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2 + line_vector[2] ** 2)
    return math.acos(dot_product / (point_magnitude * line_magnitude))


def sort_points_clockwise(points, start, end):
    # Create normal vector from line start and end points
    start = np.array(start)
    end = np.array(end)

    vector = np.array([end[0] - start[0], end[1] - start[1], end[2] - start[2]])
    norm = np.linalg.norm(vector)
    n = vector / norm

    p = points[0] - points[0].dot(n) * n  # take the first point to compute the first orthogonal vector
    q = np.cross(n, p)

    angles = []
    for point in points:
        t = np.dot(n, np.cross((point - start), p))
        u = np.dot(n, np.cross((point - start), q))
        angles.append(math.atan2(u, t))

    sorted_points = [x for _, x in sorted(zip(angles, points), key=lambda pair: pair[0])]
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


# test_sort_points_clockwise()


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
    for k, v in faces.items():
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
    plotter.add_mesh(shape1, show_edges=True, cmap="cool", camera_position=camera_pos1, view_up=view_up1)
    plotter.add_mesh(shape2, show_edges=True, cmap="hot", camera_position=camera_pos2, view_up=view_up2)
    plotter.add_mesh(shape3, show_edges=True, cmap="inferno", camera_position=camera_pos3, view_up=view_up3)
    plotter.add_mesh(shape4, show_edges=True, cmap="viridis", camera_position=camera_pos4, view_up=view_up4)

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
    return np.array([[1, 0, 0, x1[0] - x0[0]], [0, 1, 0, x1[1] - x0[1]], [0, 0, 1, x1[2] - x0[2]], [0, 0, 0, 1]])