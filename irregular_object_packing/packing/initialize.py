"""
Initialization phase of the packing algorithm.

"""
# %%
from typing import List

import numpy as np
import plotly.graph_objs as go
from pyvista import PolyData
from scipy.spatial import Voronoi


def random_coordinate_within_bounds(bounding_box: np.ndarray) -> np.ndarray:
    """generates a random coordinate within the bounds of the bounding box"""
    x = np.random.uniform(bounding_box[0][0], bounding_box[1][0])
    y = np.random.uniform(bounding_box[0][1], bounding_box[1][1])
    z = np.random.uniform(bounding_box[0][2], bounding_box[1][2])
    random_positions = np.array((x, y, z))
    return random_positions


# def random_coordinate_within_bounds(bounding_box: np.ndarray, N=1000) -> np.ndarray:
#     """generates a random coordinate within the bounds of the bounding box"""
#     x = np.random.uniform(bounding_box[0][0], bounding_box[1][0], N)
#     y = np.random.uniform(bounding_box[0][1], bounding_box[1][1], N)
#     z = np.random.uniform(bounding_box[0][2], bounding_box[1][2], N)
#     random_positions = np.array((x, y, z)).reshape(N, 3)
#     return random_positions


def get_min_bounding_mesh(mesh: PolyData) -> PolyData:
    """Selects one of 'Box', 'Sphere' or 'Cylinder' bounding mesh of mesh that has the smallest volume

    Args:
        mesh (PolyData): original mesh

    Returns:
        PolyData: bounding mesh
    """
    options = [mesh.bounding_box_oriented, mesh.bounding_sphere, mesh.bounding_cylinder]
    volume_min = np.argmin([i.volume for i in options])
    bounding_mesh = options[volume_min]
    return bounding_mesh


def get_max_radius(mesh: PolyData) -> float:
    """Returns the maximum distence from the center of mass to the mesh points

    Args:
        mesh (PolyData): original mesh

    Returns:
        float: maximum dimension
    """
    distances = np.linalg.norm(mesh.points - mesh.center_of_mass(), axis=1)
    max_distance = np.max(distances)
    return max_distance


def init_coordinates(
    container: PolyData,
    mesh: PolyData,
    coverage_rate: float = 0.3,
    f_init: float = 0.1,
) -> tuple[np.ndarray, int]:
    """Places the objects inside the container at initial location.

    Args:
        container (PolyData): container mesh
        mesh (PolyData): mesh of the objects
        coverage_rate (float): percentage of the container volume that should be filled

    returns:
        tuple[np.ndarray, int]: coordinates of the objects and number of skipped objects
    """
    # TODO: Make sure the container is a closed surface mesh
    # max_dim_mesh = max(np.abs(mesh.bounds)) * 2 # for sphere this is the same, but quicker. for other shapes might be different
    max_dim_mesh = get_max_radius(mesh) * 2

    min_distance_between_meshes = f_init ** (1 / 3) * max_dim_mesh
    max_volume = container.volume * coverage_rate

    objects_coords = []
    acc_vol, skipped = 0, 0
    while acc_vol < max_volume:
        coord = random_coordinate_within_bounds(np.reshape(container.bounds, (2, 3)))
        if coord_is_correct(
            coord, container, objects_coords, min_distance_between_meshes
        ):
            objects_coords.append(coord)
            acc_vol += mesh.volume
        else:
            skipped += 1

    return objects_coords, skipped


def trimesh_variant():
    return
    while acc_vol < max_volume:
        coord = random_coordinate_within_bounds(scaled_container.bounds)
        if scaled_container.contains([coord]):
            distance_arr = [
                np.linalg.norm(coord - i) > min_distance_between_meshes
                for i in objects_coords
            ]
            distance_to_container = trimesh.proximity.signed_distance(
                scaled_container, [coord]
            )[0]
            distance_arr.append(distance_to_container > min_distance_between_meshes / 2)


def coord_is_correct(
    coord,
    container: PolyData,
    object_coords: list[np.ndarray],
    min_distance_between_meshes: float,
):
    is_inside = PolyData([coord]).select_enclosed_points(container)["SelectedPoints"][0]
    if is_inside == 1:
        distance_arr = [
            np.linalg.norm(coord - i) > min_distance_between_meshes
            for i in object_coords
        ]
        # distance_to_container = trimesh.proximity.signed_distance(container, [coord])[0]
        point = container.find_closest_point(coord)
        distance_to_container = np.linalg.norm(coord - container.points[point])
        distance_arr.append(distance_to_container > min_distance_between_meshes / 2)

        if np.alltrue(distance_arr):
            return True
    return False


def filter_coords(
    container: PolyData, mesh_volume, coverage_rate, min_distance, coords
):
    max_volume = container.volume * coverage_rate
    acc_vol = 0
    skipped = 0
    objects_coords = []
    # object is centered at the origin

    points_inside = PolyData(coords).select_enclosed_points(container)

    i = -1
    while acc_vol < max_volume:
        i += 1
        coord = points_inside.points[i]
        distance_arr = [True] + [
            np.linalg.norm(coord - i) > min_distance for i in objects_coords
        ]

        if np.alltrue(distance_arr):
            point = container.find_closest_point(coord)
            distance_to_container = np.linalg.norm(coord - point)
            if distance_to_container > min_distance / 2:
                objects_coords.append(coord)
                acc_vol += mesh_volume
                continue

            skipped += 1
    return skipped, objects_coords


# NOT IN USE CURRENTLY
def dynamic_plot(points: np.ndarray, power_cells: List[np.ndarray]):
    """Create a dynamic 3D plot of the power cells and the input points."""
    polygons = []
    for _i, cell in enumerate(power_cells):
        polygons.append(
            go.Mesh3d(
                x=[x[0] for x in cell],
                y=[x[1] for x in cell],
                z=[x[2] for x in cell],
                #   color=plt.cm.jet(i/len(power_cells)),
                opacity=0.5,
            )
        )

    # Create the input points as a scatter plot
    if points is not None:
        scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker={"size": 3, "color": "red"},
        )

    # Combine the polygons and the scatter plot into a single figure
    fig = go.Figure(data=polygons + [scatter])

    # Set the axis labels and the title
    fig.update_layout(
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z"},
        title="Power Cells",
    )

    fig.show()


# NOT IN USE CURRENTLY
class PartitionBuilder:
    vor: Voronoi
    container: PolyData
    points: np.ndarray
    seed_points: np.ndarray = []
    threshold: float = 0.01
    power_cells: List[np.ndarray] = []

    def __init__(self, container: PolyData, points: np.ndarray):
        self.container = container
        self.points = points
        self.seed_points = np.random.rand(len(points), 3) * 4
        self.vor = Voronoi(points)

    def power_cell_step(self):
        self.power_cells = []
        for i in range(len(self.points)):
            region = self.vor.regions[self.vor.point_region[i]]
            if len(region) > 0:
                vertices = self.vor.vertices[region]
                # power_cell = PolyData(vertices=vertices, faces=self.vor.ridge_vertices)
                # power_cell = power_cell.intersection(self.container)
                power_cell = vertices
                self.power_cells.append(power_cell)

        # centroids = []
        # for power_cell in self.power_cells:
        # # Use a library such as numpy to compute the centroid of the cell
        #     centroid = np.mean(power_cell, axis=0)
        #     centroids.append(centroid)

        # self.seed_points = centroids

    def plot_power_cells(self):
        dynamic_plot(self.points, self.power_cells)

    def run(self):
        for _i in range(100):
            self.power_cell_step()
