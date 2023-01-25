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
from dataclasses import dataclass
from typing import TypeAlias
import pyvista as pv
import numpy as np


cat_points = np.array([[1, -1, 1], [1, 1, 1], [-1, 1, 1], [0,0,2]])
# %% Check how the points are connected
pc = pv.PolyData(cat_points)
tetmesh = pc.delaunay_3d()
edges = tetmesh.extract_all_edges()
# edges.plot()
# tetmesh.explode().plot()


# %%
for i in range(tetmesh.n_cells):
    print('---')
    for cell_points in tetmesh.cell_points(i):
        print(cell_points)
# %%
cat_points = np.array([[1, -1, 1], [1, 1, 1], [-1, 1, 1], [0,0,2]])#, [1, 1, 3]])#, [2, 3, 1], [3, 1, 2], [3, 2, 1] , [2, 1, 3], [4,1,2]], dtype=float)
container = pv.Pyramid([
    [4, -4, 0],
    [-4, -4, 0],
    [-4, 4, 0],
    [4, 4, 0],
    [0, 0, 4],
])

pc = pv.PolyData(np.concatenate([cat_points, container.points]))
tetmesh = pc.delaunay_3d()
tetmesh.explode().plot()

# %%

# %% 
@dataclass(unsafe_hash=True)
class Vertex():
    x: float
    y: float
    z: float
        
    def __add__(self, __o: object) -> 'Vertex':
        return Vertex(self.x + __o.x, self.y + __o.y, self.z + __o.z)
    
    def __div__(self, __o: object) -> 'Vertex':
        return Vertex(self.x / __o, self.y / __o, self.z / __o)
    
# Vertex: TypeAlias = tuple[float, float, float]
class TetPoint():
    point: Vertex
    obj_id: int
    def __init__(self, point: Vertex, obj_id):
        self.point = point
        self.obj_id = obj_id
        
    def __eq__(self, other):
        return self.point == other.point
    
    def same_obj(self, other: 'TetPoint'):
        return self.obj_id == other.obj_id

    def distance(self, other: 'TetPoint'):
        return np.linalg.norm(self.point - other.point) 
    
    def center(self, other: 'TetPoint') -> Vertex:
        return (self.point + other.point) /2

TetPoints: TypeAlias = list[TetPoint]

    
#%% 
# ## Extract facets of the tetrahedrons that have points from more than one object

point_sets = [set(map(tuple,cat_points)), set(map(tuple,container.points))]
cat_faces = {}
for obj_id in range(len(point_sets)):
    cat_faces[obj_id] = []

selected_cells = []
types = []

# go over each cell in the tethraheron mesh and check if it has points from more than one object, if so, comput the CAT facet
for cell in range(tetmesh.n_cells):
    occ = {}
    tet_points: TetPoints = []

    for i, obj in enumerate(point_sets):  
        
        for cell_point in tetmesh.cell_points(cell):
        # check if the cell has points from more than one point set
            if tuple(cell_point) in obj:
                occ[i] = occ.get(i, 0) + 1
                # occ[i] += 1
                tet_points.append(TetPoint(cell_point, i))
     
    #sort occ on value
    occ = sorted(occ.items(), key=lambda x: x[1], reverse=True)
    
        
    # for cell_point in tetmesh.cell_points(cell):
            
    assert(len(tet_points) == 4)
    n_objs = len(occ)
    if n_objs == 1:
        continue # skip cells that have points from only one object
    
    if n_objs == 4:
        pass
    
    if n_objs == 3:
        # if occ == [2,1,1,0]:
        # if occ == [1,2,1,0]:
        # if occ == [1,1,2,0]:
        pass
    
    if n_objs == 2:
        # if occ == [2,2,0,0]:
        # if occ == [1,3,0,0]:
    
        most = [p for p in tet_points if p.obj_id == occ[0][0]]
        least = [p for p in tet_points if p.obj_id == occ[1][0]]
        
        face: list[Vertex] = []
        for pa in least:
            face += [pa.center(pb) for pb in most]
            # Add face to each object cat cell
            for (k, f) in occ:
                cat_faces[k].append(face)

for faces in cat_faces[0]:
    assert len(faces) >= 3, f"Not enough faces for a triangle: {faces}"

# %%

cat_faces0 = cat_faces[0][:4]
def face_coord_to_points_and_faces(cat_faces0):
    """Convert a list of faces represented by points with coordinates 
    to a list of points and a list of faces represented by the number of points and point ids.
    
    This function is used to convert the data so that it can be used by the pyvista.PolyData class.
    
    Note: Currently this function assumes that the indices of the points are not global with respect to other meshes.
    
    >>> cat_faces0 = [[[1, -1, 1], [1, 1, 1], [-1, 1, 1], [0,0,2]], [[1, -1, 1], [1, 1, 1], [-1, 1, 1], [0,0,2]]]
    
    """
    cat_points = []
    poly_faces = []

    points = {}
    # points: list[Vertex] = []
    # point_ids: list[int] = []
    counter = 0
    for i, face in enumerate(cat_faces0):
        poly_face = [len(face)]
    
        for point in face:
            point = Vertex(*point)
            if point not in points.values():
            
            # if any(old_point.all()point for old_point in points.values()):
                points[point] = counter
                cat_points.append(point)
                counter += 1
                
            poly_face.append(points[point])
        poly_faces.append(poly_face)
    return cat_points, poly_faces

cat_points, poly_faces = face_coord_to_points_and_faces(cat_faces0)
    
print(f"cat faces: {cat_faces0}")
print(f"all_points: {[a for a in enumerate(cat_points)]}")
print(f"poly_faces: {poly_faces}")

# for one list of faces make a polydata object

 


    
# tetmesh.extract_cells(isolated_cells).plot()
# %%
