"""
Initialization phase of the packing algorithm.

"""
# %%
import random
from typing import List
from scipy.spatial import Voronoi

import numpy as np
import plotly.graph_objs as go
import trimesh


def random_coordinate_within_bounds(bounding_box: np.ndarray) -> np.ndarray:
    """generates a random coordinate within the bounds of the bounding box"""
    x = random.uniform(bounding_box[0][0], bounding_box[1][0])
    y = random.uniform(bounding_box[0][1], bounding_box[1][1])
    z = random.uniform(bounding_box[0][2], bounding_box[1][2])
    random_position = np.array((x, y, z))
    return random_position


def get_min_bounding_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """gets the bounding mesh of mesh that has the smallest volume

    Args:
        mesh (trimesh.Trimesh): original mesh

    Returns:
        trimesh.Trimesh: bounding mesh
    """
    options = [mesh.bounding_box_oriented, mesh.bounding_sphere, mesh.bounding_cylinder]
    volume_min = np.argmin([i.volume for i in options])
    bounding_mesh = options[volume_min]
    return bounding_mesh


def init_coordinates(
    container: trimesh.Trimesh,
    mesh: trimesh.Trimesh,
    coverage_rate: float = 0.3,
    c_scale: float = 1.0,
) -> np.ndarray:
    """Places the objects inside the container at initial location.

    Args:
        container (trimesh.Trimesh): container mesh
        mesh (trimesh.Trimesh): mesh of the objects
        coverage_rate (float): percentage of the container volume that should be filled
    """
    if c_scale != 1.0:
        scaled_container = container.copy().apply_scale(c_scale)
    else:
        scaled_container = container

    # container_bound = get_min_bounding_mesh(container.apply_scale(0.8))
    max_volume = container.volume * coverage_rate
    acc_vol = 0
    skipped = 0
    objects_coords = []
    while acc_vol < max_volume:
        coord = random_coordinate_within_bounds(scaled_container.bounds)
        if scaled_container.contains([coord]):
            distance_arr = [np.linalg.norm(coord - i) > mesh.volume ** (1 / 3) for i in objects_coords]
            distance_to_container = trimesh.proximity.signed_distance(scaled_container, [coord])[0]

            distance_arr.append(distance_to_container > 1 / 5 * mesh.volume ** (1 / 3))

            if np.alltrue(distance_arr):
                objects_coords.append(coord)
                acc_vol += mesh.volume
            else:
                skipped += 1

    print(f"Skipped {skipped} points for total of {len(objects_coords)} points")
    return objects_coords


# NOT IN USE CURRENTLY
def dynamic_plot(points: np.ndarray, power_cells: List[np.ndarray]):
    """Create a dynamic 3D plot of the power cells and the input points."""
    polygons = []
    for i, cell in enumerate(power_cells):
        polygons.append(
            go.Mesh3d(
                x=list(map(lambda x: x[0], cell)),
                y=list(map(lambda x: x[1], cell)),
                z=list(map(lambda x: x[2], cell)),
                #   color=plt.cm.jet(i/len(power_cells)),
                opacity=0.5,
            )
        )

    # Create the input points as a scatter plot
    if points is not None:
        scatter = go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers", marker=dict(size=3, color="red")
        )

    # Combine the polygons and the scatter plot into a single figure
    fig = go.Figure(data=polygons + [scatter])

    # Set the axis labels and the title
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), title="Power Cells")

    fig.show()


# NOT IN USE CURRENTLY
class PartitionBuilder:
    vor: Voronoi
    container: trimesh.Trimesh
    points: np.ndarray
    seed_points: np.ndarray = []
    threshold: float = 0.01
    power_cells: List[np.ndarray] = []

    def __init__(self, container: trimesh.Trimesh, points: np.ndarray):
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
                # power_cell = trimesh.Trimesh(vertices=vertices, faces=self.vor.ridge_vertices)
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
        for i in range(100):
            self.power_cell_step()
