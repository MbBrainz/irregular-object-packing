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


# TODO: Move plot func to a separate file
# Create the power cells as a list of polygons
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


def place_objects(
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
    objects_coords = []
    while acc_vol < max_volume:
        coord = random_coordinate_within_bounds(scaled_container.bounds)
        if scaled_container.contains([coord]):
            distance_arr = [np.linalg.norm(coord - i) > mesh.volume ** (1 / 3) for i in objects_coords]
            if np.alltrue(distance_arr):
                objects_coords.append(coord)
                acc_vol += mesh.volume
    return objects_coords


def create_packed_scene(
    container: trimesh.Trimesh,
    objects_coords: List[np.ndarray],
    mesh: trimesh.Trimesh,
    mesh_scale: float = 1,
    rotate: bool = False,
):
    """make a trimesh scene with the container and the objects inside.

    Args:
        container (trimesh.Trimesh): container mesh
        objects_coords (List[np.ndarray]): list of coordinates of the objects
        mesh (trimesh.Trimesh): mesh of the objects
        mesh_scale (float, optional): scale of the objects. Defaults to 1.
    """
    objects = []
    for coord in objects_coords:
        new_mesh = mesh.copy()
        if rotate:
            new_mesh = new_mesh.apply_transform(trimesh.transformations.random_rotation_matrix())

        new_mesh.apply_scale(mesh_scale).apply_translation(coord)

        new_mesh.visual.vertex_colors = trimesh.visual.random_color()
        objects.append(new_mesh)

    container.visual.vertex_colors = [250, 255, 255, 100]

    objects.append(container)
    scene = trimesh.Scene(objects)

    return scene


def save_image(scene: trimesh.Scene, path: str):
    png = scene.save_image()
    # Write the bytes to file
    with open(path, "wb") as f:
        f.write(png)


# %%
# if __name__ == '__main__':

#     DATA_FOLDER = './../../../data/mesh/'
#     mesh = trimesh.load_mesh(DATA_FOLDER + 'yog.obj')
#     container = trimesh.primitives.Cylinder(radius=1, height=1)
#     # print the volume of the mesh and of the container

#     container = container.apply_scale(40)

#     objects_coords = pack_objects(container, mesh, 0.5)
#     scene = create_packed_scene(container, objects_coords, mesh)
#     scene.show()
#     save_image(scene, 'packing.png')

# %%


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


# #%%
# container = trimesh.primitives.Box(extends=[4,4,4])
# # point in each corner of the box
# points = np.array([[-1, -1, -1], [ 1, -1, -1], [-1,  1, -1], [ 1,  1, -1], [-1, -1,  1], [ 1, -1,  1], [-1,  1,  1], [ 1,  1,  1],
#                    [0,0,0]])

# # add a list of boundary points
# # points = np.concatenate((points, [[0,0,0], [0,0,4], [0,4,0], [0,4,4], [4,0,0], [4,0,4], [4,4,0], [4,4,4]]))
# # points = np.array([[1, 2, 3], [1, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1] , [2, 1, 3], [4,1,2]])
# # seed_points = np.random.rand(len(points), 3)*4
# seed_points = points
# threshold = 0.001

# partition = PartitionBuilder(container, points)
# partition.power_cell_step()
# partition.plot_power_cells()
# # %%
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D, art3d


# # %%
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # ax.set(xlim=(0, 4), ylim=(0, 4), zlim=(0, 4))
# # Plot the power cells
# for i, cell in enumerate(power_cells):
#     ax.add_collection(Poly3DCollection([cell], alpha=0.25, facecolor=plt.cm.jet(i/len(power_cells))))

# # Plot the input points
# ax.scatter(points[:,0], points[:,1], points[:,2], c='r')

# plt.show()

# # %%

# dynamic_plot(points, power_cells)

# # %%
# container = trimesh.primitives.Cylinder(radius=4, height=4)
# from scipy.spatial import Delaunay
# tri = Delaunay(container.vertices)

# vor = Voronoi(tri.points)

# for i, seed in enumerate(seed_points):
# \region = vor.regions[vor.point_region[i]]
#     if len(region) > 0:
#         verts = vor.vertices[region]
#         # check if the seed point is contained within the container mesh
#         if not container.contains(verts):
#             seed_points.remove(seed)

# vor = Voronoi(seed_points)

# power_cells = []
# for i, region in enumerate(vor.regions):
#     if len(region) > 0:
#         verts = vor.vertices[region]
#         power_cells.append(verts)


# dynamic_plot(None, power_cells)
# # %%
# import pyvista as pv
# import trimesh

# # Compute the Voronoi diagram of the seed points
# vor = Voronoi(points)


# # Convert the container mesh to a trimesh object
# container_mesh = trimesh.primitives.Cylinder(radius=4, height=4)

# # Create a list to store the clipped Voronoi cells
# clipped_cells = []

# # Iterate through the Voronoi cells
# for i, cell in enumerate(vor.regions):
#     # Convert the cell to a trimesh object
#     cell_mesh = trimesh.Trimesh(vertices=cell, faces=np.arange(len(cell)).reshape(-1, 3))
#     # Compute the intersection between the cell and the container mesh
#     intersection = container_mesh.intersection(cell_mesh)
#     if intersection.is_empty:
#         continue
#     # Append the intersection to the list of clipped cells
#     clipped_cells.append(intersection)

# # Plot the clipped Voronoi cells
# p = pv.Plotter()
# for cell in clipped_cells:
#     p.add_mesh(cell.triangles, color='blue', opacity=0.5)
# p.add_mesh(container)
# p.show()

# # %%

# # %%
