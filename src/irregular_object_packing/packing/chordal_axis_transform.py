""" In step 4.2.1 of the algorithm, The CAT is computed by using the following steps:
1. Create a Tetrahedron Mesh from all the points on the surface mesh of both the objects and the container as the input points.
2. Use onlty those tetrahedrons that constructed of points from multiple objects.
3. Compute the chordal axis of each tetrahedron. 
4. Compute the chordal axis of the whole object by taking the union of all the chordal axis of the tetrahedrons.(SORT OF)

"""
#%%
# 1. create a tetrahedron mesh from the points set
# 2. go over each tetrahedron and see if it has points from multiple objects
# 3. compute the chordal axis of each tetrahedron in that case
# %%

from collections import Counter
from typing import TypeAlias
import pyvista as pv
import numpy as np


points = np.array([[1, -1, 1], [1, 1, 1], [-1, 1, 1], [0,0,2]])
# %% Check how the points are connected
pc = pv.PolyData(points)
tetmesh = pc.delaunay_3d()
edges = tetmesh.extract_all_edges()
# edges.plot()
tetmesh.explode().plot()


# %%
for i in range(tetmesh.n_cells):
    print('---')
    for cell_points in tetmesh.cell_points(i):
        print(cell_points)
# %%
points = np.array([[1, -1, 1], [1, 1, 1], [-1, 1, 1], [0,0,2]])#, [1, 1, 3]])#, [2, 3, 1], [3, 1, 2], [3, 2, 1] , [2, 1, 3], [4,1,2]], dtype=float)
container = pv.Pyramid([
    [4, -4, 0],
    [-4, -4, 0],
    [-4, 4, 0],
    [4, 4, 0],
    [0, 0, 4],
])

pc = pv.PolyData(np.concatenate([points, container.points]))
tetmesh = pc.delaunay_3d()
tetmesh.explode().plot()

# %%

# %% Filter out the cells that have points from only one object
point_sets = [set(map(tuple, points)), set(map(tuple,container.points))]
cat_faces = {}
for obj_id in range(len(point_sets)):
    cat_faces[obj_id] = []

selected_cells = []
types = []

class TetPoint():
    def __init__(self, point, obj_id):
        self.point = point
        self.obj_id = obj_id
        
    def __eq__(self, other):
        return self.point == other.point
    
    def same_obj(self, other: 'TetPoint'):
        return self.obj_id == other.obj_id
    
    def same(self, obj_id):
        return self.obj_id == obj_id

    def distance(self, other: 'TetPoint'):
        return np.linalg.norm(self.point - other.point)
    
    def center(self, other: 'TetPoint'):
        return (self.point + other.point) / 2

TetPoints: TypeAlias = list[TetPoint]

    

for cell in range(tetmesh.n_cells):
    occ = [0,0,0,0]
    tet_points: TetPoints = []

    for i, obj in enumerate(point_sets):  
        for cell_point in tetmesh.cell_points(cell):
        # check if the cell has points from more than one point set
            if tuple(cell_point) in obj:
                occ[i] += 1
                tet_points.append(TetPoint(cell_point, i))
                
        
    # for cell_point in tetmesh.cell_points(cell):
            
    assert(len(tet_points) == 4)
    
    if occ == [1,1,1,1]:
        pass

    if occ == [2,1,1,0]:
        pass
    
    if occ == [1,2,1,0]:
        pass
    
    if occ == [1,1,2,0]:
        pass
    
    if occ == [2,2,0,0]:
        pass
    
    if occ == [1,3,0,0]:
        multi = [p for p in tet_points if p.obj_id == 1]
        single = [p for p in tet_points if p.obj_id == 0][0]
        
        face = [single.center(p) for p in multi]
        # this currently only works for 2 objects
        cat_faces[0].append(face)
        cat_faces[1].append(face)

    if occ == [3,1,0,0]:
        multi = [p for p in tet_points if p.obj_id == 0]
        single = [p for p in tet_points if p.obj_id == 1][0]
        
        face = [single.center(p) for p in multi]
        # this currently only works for 2 objects
        cat_faces[0].append(face)
        cat_faces[1].append(face)

print(cat_faces)

#%% [markdown] 
# ## Extract facets of the tetrahedrons that have points from more than one object
# %%
p_a = []

 


    
# tetmesh.extract_cells(isolated_cells).plot()
# %%
