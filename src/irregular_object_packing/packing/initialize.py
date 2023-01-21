"""
Initialization phase of the packing algorithm.

"""
#%%
import random
from typing import List
import numpy as np
import trimesh

def random_coordinate_within_bounds(bounding_box: np.ndarray) -> np.ndarray:
    """generates a random coordinate within the bounds of the bounding box

    """
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
    options = [mesh.bounding_box_oriented,
                mesh.bounding_sphere,
                mesh.bounding_cylinder]
    volume_min = np.argmin([i.volume for i in options])
    bounding_mesh = options[volume_min]
    return bounding_mesh

def pack_objects(container: trimesh.Trimesh, mesh: trimesh.Trimesh, coverage_rate: float) -> np.ndarray:
    """packs the objects inside the container
    
    Args:
        container (trimesh.Trimesh): container mesh
        mesh (trimesh.Trimesh): mesh of the objects
        coverage_rate (float): percentage of the container volume that should be filled
    """
    container_bound = get_min_bounding_mesh(container)
    max_volume = container_bound.volume * coverage_rate
    acc_vol = 0 
    objects_coords = []
    while acc_vol < max_volume:
        coord = random_coordinate_within_bounds(container_bound.bounds)
        if container.contains([coord]):
            objects_coords.append(coord)
            acc_vol += mesh.volume
    return objects_coords

def create_packed_scene(container: trimesh.Trimesh, objects_coords: List[np.ndarray], mesh: trimesh.Trimesh, mesh_scale: float = 1):
    """make a trimesh scene with the container and the objects inside. 
    
    Args:
        container (trimesh.Trimesh): container mesh
        objects_coords (List[np.ndarray]): list of coordinates of the objects
        mesh (trimesh.Trimesh): mesh of the objects
        mesh_scale (float, optional): scale of the objects. Defaults to 1.
    """
    nodes = []
    for coord in objects_coords:
        new_mesh = mesh.copy().apply_scale(mesh_scale).apply_translation(coord).apply_transform(trimesh.transformations.random_rotation_matrix())
        new_mesh.visual.vertex_colors = trimesh.visual.random_color()
        nodes.append(new_mesh)

    container.visual.vertex_colors = [250, 255, 255, 100]
    nodes.append(container) 
    scene = trimesh.Scene(nodes)
    return scene

def save_image(scene: trimesh.Scene, path: str):
    png = scene.save_image()
    # Write the bytes to file
    with open(path, "wb") as f:
        f.write(png)

#%%
if __name__ == '__main__':

    DATA_FOLDER = './../../../data/mesh/'
    mesh = trimesh.load_mesh(DATA_FOLDER + 'yog.obj')
    container = trimesh.primitives.Cylinder(radius=1, height=1)
    # print the volume of the mesh and of the container

    container = container.apply_scale(40)

    objects_coords = pack_objects(container, mesh, 0.5)
    scene = create_packed_scene(container, objects_coords, mesh)
    scene.show()
    save_image(scene, 'packing.png')

# %%
