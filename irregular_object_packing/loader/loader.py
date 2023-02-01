# %%
import meshtaichi_patcher as Patcher
import taichi as ti
from taichi import Mesh, MeshInstance

# import meshio
# import pygmsh as pg
# import pymesh
# DATA_FOLDER = './../../../data/mesh/'

# #%%
# ti.init(arch=ti.gpu)
# mesh = Patcher.load_mesh(DATA_FOLDER + 'yog.obj', relations=['FV'])
# mesh.verts.place({'x' : ti.math.vec3,
#                   'c' : ti.f32, #center
#                   'scale' : ti.f16})

# mesh.x.from_numpy(mesh.get_position_as_numpy())


# #%%
# for f in mesh.faces:
#     print(f)
# %% Import container
