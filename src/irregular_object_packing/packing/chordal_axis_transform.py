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

from dataclasses import dataclass
import pyvista as pv
import numpy as np

# from utils import angle_between, sort_points_clockwise
from packing.utils import angle_between, sort_points_clockwise, sort_faces_dict

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
    
@dataclass
class CatFace():
    vertices: list[np.ndarray]
    v_ids: list[int]
    related_objects: list[int]
    
@dataclass
class Triangle(CatFace):
    def __init__(self, face, v_ids, related_objects):
        if len(face) != 3:
            raise ValueError(f"Triangle must have 3 points, got {len(face)}")
        super().__init__(face, v_ids, related_objects)
    
    def center(self) -> np.ndarray:
        center = self.vertices[0] + self.vertices[1] + self.vertices[2]
        return center / 3
    
    @property
    def area(self):
        a, b, c = self.vertices
        return 0.5 * np.linalg.norm(np.cross(b-a, c-a))
    
    def midpoints(self) -> list[np.ndarray]:
        (a, b, c) = self.vertices
        return [(a+b)/2, (b+c)/2, (c+a)/2]
    
# Vertex: TypeAlias = tuple[float, float, float]
class TetPoint():
    vertex: np.ndarray
    obj_id: int
    tet_id: int
    triangles: list[Triangle]
    
    def __init__(self, point: np.ndarray, obj_id=-1, tet_id=-1):
        self.vertex = point
        self.obj_id = obj_id
        self.tet_id = tet_id
        self.triangles = []
        
    def __eq__(self, other):
        return self.vertex == other.vertex
    
    def __add__(self, o: object) -> 'TetPoint':
        return TetPoint(self.vertex + o.vertex, self.tet_id)
    
    def __div__(self, o: object) -> np.ndarray:
        return self.vertex / o
    
    def __truediv__(self, other):
        return self.__div__(other)
    
    def same_obj(self, other: 'TetPoint'):
        return self.obj_id == other.obj_id

    def distance(self, other: 'TetPoint'):
        return np.linalg.norm(self.vertex - other.vertex) 
    
    def center(self, other: 'TetPoint') -> np.ndarray:
        return (self.vertex + other.vertex) / 2
    
    def add_triangle(self, triangle: Triangle):
        if len(self.triangles) >= 3:
            raise ValueError(f"TetPoint {self} already has 3 triangles")
        self.triangles.append(triangle)

    
#%%
    
def create_faces_3(cat_faces, occ, tet_points: list[TetPoint]):
    most = [p for p in tet_points if p.obj_id == occ[0][0]]
    least = [p for p in tet_points if (p.obj_id == occ[1][0] or p.obj_id == occ[2][0])]
    
    assert len(most) == 2
    assert len(least) == 2
    
    face: list[np.ndarray] = []
    
    # find center point of 2 triangles
    triangles: list[Triangle] = []
    for pa in most:
        triangles.append(Triangle([pa.vertex, least[0].vertex, least[1].vertex], [], [pa.obj_id, least[0].obj_id]))
    
    bc_face = [least[0].center(least[1]), triangles[0].center(), triangles[1].center()]
    aab_face = [least[0].center(most[1]), least[0].center(most[0]), triangles[0].center(), triangles[1].center()]
    aac_face = [least[1].center(most[0]), least[1].center(most[1]), triangles[0].center(), triangles[1].center()]
    
    # Add face to each object cat cell
    cat_faces[most[0].obj_id].append(aab_face)
    cat_faces[most[0].obj_id].append(aac_face)
    
    cat_faces[least[0].obj_id].append(bc_face)
    cat_faces[least[0].obj_id].append(aab_face)
    
    cat_faces[least[1].obj_id].append(bc_face)
    cat_faces[least[1].obj_id].append(aac_face)

    return cat_faces



#%% go over each cell in the tethraheron mesh and check if it has points from more than one object, if so, comput the CAT facet
def create_faces_2(cat_faces, occ, tet_points: list[TetPoint]):
    assert len(occ) == 2
    
    most = [p for p in tet_points if p.obj_id == occ[0][0]]
    least = [p for p in tet_points if p.obj_id == occ[1][0]]
        
    face: list[np.ndarray] = []
    for pa in least:
        face += [pa.center(pb) for pb in most]
        
        # Add face to each object cat cell
    for (k, f) in occ:
        cat_faces[k].append(face)

    return cat_faces
    
def single_point_4faces(tet_point: TetPoint, others: list[TetPoint], tet_center: np.ndarray):
    # this should result in 6 faces
    # first 6 points: center bc, center bca, center ba, center bca, center bd, center bcd
    if len(tet_point.triangles) != 3:
        raise ValueError(f"tet_point {tet_point} must have 3 triangles")
    if len(others) != 3:
        raise ValueError(f"others {others} should have len 3")
    
    v0 = tet_point.vertex
    points: list[np.ndarray] = []
    
    for triangle in tet_point.triangles:
        points.append(triangle.center())
    
    for other in others:
        midpoint = (v0 + other.vertex) / 2
        points.append(midpoint)
        
    # sorted_points = sorted(points, key=lambda point: angle_between(point.tuple, v0.tuple, tet_center.tuple))
    sorted_points = sort_points_clockwise(points, v0, tet_center)
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
    tet_center = sum([point.vertex for point in tet_points]) / 4
    for point in tet_points:
        others = [other for other in tet_points if other != point]
        cat = single_point_4faces(point, others, tet_center)
        cat_faces[point.obj_id].append(cat)
        
    return cat_faces
    


#%% 
# ## Extract facets of the tetrahedrons that have points from more than one object
point_sets = [set(map(tuple,cat_points)), set(map(tuple,container.points))]
cat_faces = {}
for obj_id in range(len(point_sets)):
    cat_faces[obj_id] = []

selected_cells = []
# types = []


for cell in range(tetmesh.n_cells):
    occ = {}
    tet_points: list[TetPoint] = []

    for i, obj in enumerate(point_sets):  
        
        for cell_point in tetmesh.cell_points(cell):
        # check if the cell has points from more than one point set
            if tuple(cell_point) in obj:
                occ[i] = occ.get(i, 0) + 1
                tet_points.append(TetPoint(cell_point, i, tet_id=cell))
    
                
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
        create_faces_3(cat_faces, occ, tet_points)
    
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
    >>> face_coord_to_points_and_faces([[np.array([0,0,0]), np.array([1,0,0]), np.array([1,1,0])]])
    ([array([0, 0, 0]), array([1, 0, 0]), array([1, 1, 0])], [3, 0, 1, 2])

    face with 4 points:
    >>> face_coord_to_points_and_faces([[np.array([0,0,0]), np.array([1,0,0]), np.array([1,1,0]), np.array([0,1,0])]])
    ([array([0, 0, 0]), array([1, 0, 0]), array([1, 1, 0]), array([0, 1, 0])], [4, 0, 1, 2, 3])
        
    faces with 3 and 4 points including overlapping points:
    >>> face_coord_to_points_and_faces([[np.array([1, -1, 1]), np.array([1, 1, 1]), np.array([-1, 1, 1]), np.array([0,0,2])], [np.array([1, -1, 1]), np.array([1, 1, 1]), np.array([-1, -1, 1])]])
    ([array([ 1, -1,  1]), array([1, 1, 1]), array([-1,  1,  1]), array([0, 0, 2]), array([-1, -1,  1])], [4, 0, 1, 2, 3, 3, 0, 1, 4])

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
            # if point not in points:
            true_array = [(old_point == point).all() for old_point in points.keys()]
            
            if not np.any(true_array):
                points[tuple(point)] = counter
                cat_points.append(point)
                counter += 1
                
            poly_face.append(points[tuple(point)])
        assert len(poly_face) >= 3, f"Not enough points for a triangle: {poly_face}"
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


    




    
    

    

# def create_faces_3(tet_points, cat_faces):
#     for i in range(3):
#         for j in range(i+1,3):
#             face = [tet_points[i].center, tet_points[j].center]
#             # Add face to each object cat cell
#             for (k, f) in occ:
#                 cat_faces[k].append(face)

# import pyvista as pv

# # # create a tetrahedron
# # tet = pv.Tetrahedron()

# # # create a list of TetPoint objects
# # tet_points = [TetPoint(p, 0) for p in tet.points]
# # last = tet_points.pop()
# # last.obj_id = 1
# # tet_points.append(last)

# # # create a dictionary of cat_faces
# # cat_faces = {0:[], 1:[], 2:[], 3:[]}
# # occ = [(0, 1), (1, 1), (2, 1), (3, 1)]

# # # create faces for n_objs == 4
# # create_faces_4(tet_points, cat_faces, occ)

# # # create faces for n_objs == 3
# # # create_faces_3(tet_points, cat_faces)

# # # create a polydata object to visualize the tetrahedron


# # # create a polydata object to visualize the faces
# # cat_points, poly_faces = face_coord_to_points_and_faces(cat_faces[0])

# # face_pd = pv.PolyData(cat_points, poly_faces)

# # # visualize the tetrahedron and the faces
# # p = pv.Plotter()
# # p.add_mesh(tet, color='blue')
# # p.add_mesh(face_pd, color='red')
# # p.show()

# # # %%
# # # TODO: from 4 points you can construct each triangle and then perform the clapboards. You dont need triangle data beforehand
# # %%
