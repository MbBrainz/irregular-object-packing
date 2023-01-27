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

from dataclasses import dataclass, field
from typing import TypeAlias
import pyvista as pv
import numpy as np

# from packing.utils import angle_between
from utils import angle_between

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
cat_points = np.array([[1, -1, 1], [1, 1, 1], [-1, 1, 1], [0,0,2]], dtype=float)#, [1, 1, 3]])#, [2, 3, 1], [3, 1, 2], [3, 2, 1] , [2, 1, 3], [4,1,2]], dtype=float)
container = pv.Pyramid(
    np.array([
    [4, -4, 0],
    [-4, -4, 0],
    [-4, 4, 0],
    [4, 4, 0],
    [0, 0, 4],
], dtype=float))

pc = pv.PolyData(np.concatenate([cat_points, container.points]))
tetmesh = pc.delaunay_3d()
# tetmesh.explode().plot()

# %%

# %% 
@dataclass(unsafe_hash=True)
class Vertex():
    x: float
    y: float
    z: float
    
    obj_id: int = field(default=-1)
        
    def __add__(self, __o: object) -> 'Vertex':
        return Vertex(self.x + __o.x, self.y + __o.y, self.z + __o.z)
    
    def __truediv__(self, o: object) -> 'Vertex':
        return Vertex(self.x / o, self.y / o, self.z / o)

    def to_numpy(self):
        return np.array([self.x, self.y, self.z])

    @property
    def tuple(self):
        return (self.x, self.y, self.z)
    
    
    
@dataclass
class CatFace():
    face: list[Vertex]
    related_objects: list[int]
    
@dataclass
class Triangle(CatFace):
    def __init__(self, face, related_objects):
        if len(face) != 3:
            raise ValueError(f"Triangle must have 3 points, got {len(face)}")
        super().__init__(face, related_objects)
    
    def center(self) -> Vertex:
        center = self.face[0] + self.face[1] + self.face[2]
        return center / 3
    
    @property
    def area(self):
        a, b, c = self.face
        return 0.5 * np.linalg.norm(np.cross(b-a, c-a))
    
    def midpoints(self) -> list[Vertex]:
        a, b, c = self.face
        return [(a+b)/2, (b+c)/2, (c+a)/2]
    
# Vertex: TypeAlias = tuple[float, float, float]
class TetPoint():
    point: Vertex
    tet_id: int
    triangles: list[Triangle]
    
    def __init__(self, point: Vertex, tet_id):
        self.point = point
        self.tet_id = tet_id
        self.triangles = []
        
    def __eq__(self, other):
        return self.point == other.point
    
    def __add__(self, o: object) -> 'TetPoint':
        return TetPoint(self.point + o.point, self.tet_id)
    
    def __div__(self, o: object) -> Vertex:
        return self.point / o
    
    def __truediv__(self, other):
        return self.__div__(other)
    
    def same_obj(self, other: 'TetPoint'):
        return self.obj_id == other.obj_id

    def distance(self, other: 'TetPoint'):
        return np.linalg.norm(self.point - other.point) 
    
    def center(self, other: 'TetPoint') -> Vertex:
        return (self.point + other.point) /2
    
    def add_triangle(self, triangle: Triangle):
        if len(self.triangles) >= 3:
            raise ValueError(f"TetPoint {self} already has 3 triangles")
        self.triangles.append(triangle)

TetPoints: TypeAlias = list[TetPoint]

class Tetrahedron():
    points: list[Vertex]
    
    
#%% go over each cell in the tethraheron mesh and check if it has points from more than one object, if so, comput the CAT facet
def create_faces_2(cat_faces, occ, tet_points: list[TetPoint]):
    most = [p for p in tet_points if p.point.obj_id == occ[0][0]]
    least = [p for p in tet_points if p.point.obj_id == occ[1][0]]
        
    face: list[Vertex] = []
    for pa in least:
        face += [pa.center(pb) for pb in most]
            # Add face to each object cat cell
        for (k, f) in occ:
            cat_faces[k].append(face)
    return face

    
def single_point_4faces(tet_point: TetPoint, others: list[TetPoint], tet_center: Vertex):
    # this should result in 6 faces
    # first 6 points: center bc, center bca, center ba, center bca, center bd, center bcd
    if len(tet_point.triangles) != 3:
        raise ValueError(f"tet_point {tet_point} must have 3 triangles")
    if len(others) != 3:
        raise ValueError(f"others {others} should have len 3")
    
    v0 = tet_point.point
    points: list[TetPoint] = []
    for triangle in tet_point.triangles:
        points.append(triangle.center())
    
    for other in others:
        midpoint = (v0 + other.point) / 2
        points.append(midpoint)
        
    
    sorted_points = sorted(points, key=lambda point: angle_between(point.tuple, v0.tuple, tet_center.tuple))
    cat_faces = []
    for i in range(len(sorted_points)):
        cat_faces.append([tet_center, sorted_points[i], sorted_points[(i+1)%len(sorted_points)]])
    
    return cat_faces
    
     
def create_faces_4(tet_points: list[TetPoint], cat_faces):
    # tet points are the 4 points of the tetrahedron
    # for each comination of 3 points create a triangle
    # a triangle is Triangle([point1, point2, point3], [])
    # and add it to the list of faces of each object
    triangles = []
    i = 0
    combinations = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    for (i, j, k) in combinations:
        triangle = Triangle([tet_points[i].point, tet_points[j].point, tet_points[k].point], 
                                  [tet_points[i].obj_id, tet_points[j].obj_id, tet_points[k].obj_id])
        triangles.append(triangle)
        tet_points[i].add_triangle(triangle)
        tet_points[j].add_triangle(triangle)
        tet_points[k].add_triangle(triangle)
        
    # for each point
    tet_center = sum([point.point for point in tet_points]) / 4
    for point in tet_points:
        others = [other for other in tet_points if other != point]
        cat = single_point_4faces(point, others, tet_center)
        cat_faces[point.obj_id].append(cat)
    

test_points = [
    TetPoint(Vertex(0.0, 0.0, 0.0, 0), 0),
    TetPoint(Vertex(8.0, 0.0, 0.0, 1), 0), 
    TetPoint(Vertex(0.0, 8.0, 0.0, 2), 0), 
    TetPoint(Vertex(0.0, 0.0 ,8.0, 3), 0)] # initialize other TetPoints
center = Vertex(4.0, 4.0, 4.0, 0)
# print(angle_between(test_points[0].point.tuple, test_points[1].point.tuple, center.tuple))
for other in test_points:
    center  += other.point
    
center = center / 4
 # initialize tet_center
# expected output

middle_ab = (test_points[0] + test_points[1]) /2
middle_ac = (test_points[0] + test_points[2]) /2
middle_ad = (test_points[0] + test_points[3]) /2
middle_abc = (test_points[1] + test_points[2]) /3
middle_acd = (test_points[2] + test_points[3]) /3
middle_abd = (test_points[1] + test_points[3]) /3
test_points[0].add_triangle(Triangle([test_points[0].point, test_points[1].point, test_points[2].point], [0,1,2]))
test_points[0].add_triangle(Triangle([test_points[0].point, test_points[2].point, test_points[3].point], [0,2,3]))
test_points[0].add_triangle(Triangle([test_points[0].point, test_points[1].point, test_points[3].point], [0,1,3]))

expected_faces = [
    [center, middle_ab, middle_abc],
    [center, middle_ab, middle_abd],
    [center, middle_ac, middle_abc],
    [center, middle_ac, middle_acd],
    [center, middle_ad, middle_abd],
    [center, middle_ad, middle_acd]
]
print(single_point_4faces(test_points[0], test_points[1:], center))


#%% 
# ## Extract facets of the tetrahedrons that have points from more than one object
point_sets = [set(map(tuple,cat_points)), set(map(tuple,container.points))]
cat_faces = {}
for obj_id in range(len(point_sets)):
    cat_faces[obj_id] = []

selected_cells = []
types = []


for cell in range(tetmesh.n_cells):
    occ = {}
    tet_points: TetPoints = []

    for i, obj in enumerate(point_sets):  
        
        for cell_point in tetmesh.cell_points(cell):
        # check if the cell has points from more than one point set
            if tuple(cell_point) in obj:
                occ[i] = occ.get(i, 0) + 1
                tet_points.append(TetPoint(Vertex(*cell_point, i), tet_id=cell))
    
                
    # cell_data = tetmesh.cell_data(cell) 
    #sort occ on value
    assert len(tet_points) == 4, f'tet_points: {tet_points}' # lil check
    occ = sorted(occ.items(), key=lambda x: x[1], reverse=True)
    n_objs = len(occ)
    
    
    if n_objs == 1:
        continue # skip cells that have points from only one object
    
    if n_objs == 4: # [1,1,1,1]
        create_faces_4(cat_faces, tet_points)
    
    if n_objs == 3: # [2,1,1,0], [1,2,1,0], [1,1,2,0]:
        pass
    
    if n_objs == 2: # [2,2,0,0], [1,3,0,0], [3,1,0,0]
    
        face = create_faces_2(cat_faces, occ, tet_points)

# for faces in cat_faces[0]:
#     assert len(faces) >= 3, f"Not enough faces for a triangle: {faces}"

# %%

cat_faces0 = cat_faces[0]
def face_coord_to_points_and_faces(cat_faces0):
    """Convert a list of faces represented by points with coordinates 
    to a list of points and a list of faces represented by the number of points and point ids.
    
    This function is used to convert the data so that it can be used by the pyvista.PolyData class.
    
    Note: Currently this function assumes that the indices of the points are not global with respect to other meshes.
    
    Face with 3 points:
    >>> face_coord_to_points_and_faces([[[0,0,0], [1,0,0], [1,1,0]]])
    ([Vertex(x=0, y=0, z=0), Vertex(x=1, y=0, z=0), Vertex(x=1, y=1, z=0)], [[3, 0, 1, 2]])

    face with 4 points:
    >>> face_coord_to_points_and_faces([[[0,0,0], [1,0,0], [1,1,0], [0,1,0]]])
    ([Vertex(x=0, y=0, z=0), Vertex(x=1, y=0, z=0), Vertex(x=1, y=1, z=0), Vertex(x=0, y=1, z=0)], [[4, 0, 1, 2, 3]])
        
    faces with 3 and 4 points including overlapping points:
    >>> face_coord_to_points_and_faces([[[1, -1, 1], [1, 1, 1], [-1, 1, 1], [0,0,2]], [[1, -1, 1], [1, 1, 1], [-1, -1, 1]]])
    ([Vertex(x=1, y=-1, z=1), Vertex(x=1, y=1, z=1), Vertex(x=-1, y=1, z=1), Vertex(x=0, y=0, z=2), Vertex(x=-1, y=-1, z=1)], [[4, 0, 1, 2, 3], [3, 0, 1, 4]])
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
            if point not in points:
            
            # if any(old_point.all()point for old_point in points.values()):
                points[point] = counter
                cat_points.append(point.to_numpy())
                counter += 1
                
            poly_face.append(points[point])
        poly_faces += poly_face
    return cat_points, poly_faces

cat_points, poly_faces = face_coord_to_points_and_faces(cat_faces0)
    
# print(f"cat faces: \n{[f'{str(face)} \n' for face in cat_faces0]}")
for face in cat_faces0: print(f"face: {face} \n")
print(f"all_points: {[a for a in enumerate(cat_points)]}")
print(f"poly_faces: {poly_faces}")

# for one list of faces make a polydata object
polydata = pv.PolyData(cat_points, poly_faces)

# %%

pl = pv.Plotter()
pl.add_mesh(polydata.explode(), color="red")
# pl.add_mesh(tetmesh.explode(), color="blue", opacity=0.5)
# pl.show()
# %%

occ = [(0, 1), (1, 1), (2, 1), (3, 1)]


    




    
    

    

def create_faces_3(tet_points, cat_faces):
    for i in range(3):
        for j in range(i+1,3):
            face = [tet_points[i].center, tet_points[j].center]
            # Add face to each object cat cell
            for (k, f) in occ:
                cat_faces[k].append(face)

import pyvista as pv

# # create a tetrahedron
# tet = pv.Tetrahedron()

# # create a list of TetPoint objects
# tet_points = [TetPoint(p, 0) for p in tet.points]
# last = tet_points.pop()
# last.obj_id = 1
# tet_points.append(last)

# # create a dictionary of cat_faces
# cat_faces = {0:[], 1:[], 2:[], 3:[]}
# occ = [(0, 1), (1, 1), (2, 1), (3, 1)]

# # create faces for n_objs == 4
# create_faces_4(tet_points, cat_faces, occ)

# # create faces for n_objs == 3
# # create_faces_3(tet_points, cat_faces)

# # create a polydata object to visualize the tetrahedron


# # create a polydata object to visualize the faces
# cat_points, poly_faces = face_coord_to_points_and_faces(cat_faces[0])

# face_pd = pv.PolyData(cat_points, poly_faces)

# # visualize the tetrahedron and the faces
# p = pv.Plotter()
# p.add_mesh(tet, color='blue')
# p.add_mesh(face_pd, color='red')
# p.show()

# # %%
# # TODO: from 4 points you can construct each triangle and then perform the clapboards. You dont need triangle data beforehand
# %%
