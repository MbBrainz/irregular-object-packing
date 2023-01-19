#%%
import taichi as ti 
import meshtaichi_patcher as Patcher
# import meshio 
# import pygmsh as pg
# import pymesh 
DATA_FOLDER = './../../../data/mesh/'

#%%
ti.init(arch=ti.gpu)
mesh = Patcher.load_mesh(DATA_FOLDER + 'yog.obj', relations=['FV'])
mesh.verts.place({'x' : ti.math.vec3, 
                  'level' : ti.i32, 
                  'd' : ti.f32, 
                  'new_d' : ti.f32})

#%%
for f in mesh.faces:
    print(f)
# %% Import container 

