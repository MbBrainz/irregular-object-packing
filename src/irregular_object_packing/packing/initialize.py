"""
Initialization phase of the packing algorithm.

"""
import random
import numpy as np
import trimesh

# %%
DATA_FOLDER = './../../../data/mesh/'
mesh = trimesh.load_mesh(DATA_FOLDER + 'yog.obj')
mesh.show()


# %%
# print(f"\t We are looking for the smallest bounding box volume:\
# mesh.bounding_box_oriented.volume: {mesh.bounding_box_oriented.volume} \n\
# mesh.bounding_cylinder.volume: {mesh.bounding_cylinder.volume} \n\
# mesh.bounding_sphere.volume: {mesh.bounding_sphere.volume}")

def random_coordinate_within_bounds(bounding_box: np.ndarray) -> np.ndarray:
    """generates a random coordinate within the bounds of the bounding box

    """
    x = random.uniform(bounding_box[0][0], bounding_box[1][0])
    y = random.uniform(bounding_box[0][1], bounding_box[1][1])
    z = random.uniform(bounding_box[0][2], bounding_box[1][2])
    random_position = np.array((x, y, z))
    return random_position

# get the bounding mesh of mesh that has the smallest volunme
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

#%%

container = trimesh.primitives.Sphere().apply_scale(50)
container_bound = get_min_bounding_mesh(container)
cov_rate = 0.3
max_volume = container_bound.volume * cov_rate
acc_vol = 0 
objects_coords = []
# fill in the container with the objects until the coverage rate is reached
while acc_vol < max_volume:
      coord = random_coordinate_within_bounds(container_bound.bounds)
      
      if container.contains([coord]):
            objects_coords.append(coord)
            acc_vol += mesh.volume



# %%
# make a scene with the meshes and the bounding box
nodes = []
for coord in objects_coords:
      new_mesh = mesh.copy().apply_scale(0.5).apply_translation(coord).apply_transform(trimesh.transformations.random_rotation_matrix())
      new_mesh.visual.vertex_colors = trimesh.visual.random_color()
      nodes.append(new_mesh)

# set transparant color
container.visual.vertex_colors = [250, 255, 255, 100]
nodes.append(container) 
scene = trimesh.Scene(nodes)
scene.show()

#%%
png = scene.save_image()
# Write the bytes to file
with open('./test_small.png', "wb") as f:
    f.write(png)
    f.close()
# %%

