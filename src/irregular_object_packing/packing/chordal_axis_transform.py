""" In step 4.2.1 of the algorithm, The CAT is computed by using the following steps:
1. Create a Tetrahedron Mesh from all the points on the surface mesh of both the objects and the container as the input points.
2. Use onlty those tetrahedrons that constructed of points from multiple objects.
3. Compute the chordal axis of each tetrahedron. 
4. Compute the chordal axis of the whole object by taking the union of all the chordal axis of the tetrahedrons.
"""
#%%
from dataclasses import dataclass
import pyvista as pv
import numpy as np
from tqdm import tqdm

# from utils import angle_between, sort_points_clockwise
from packing.utils import angle_between, sort_points_clockwise, sort_faces_dict
    
@dataclass
class CatFace():
    vertices: list[np.ndarray]
    v_ids: list[int]
    related_objects: list[int]
    
@dataclass
class Triangle(CatFace):
    """A triangle face in the CAT. Used to keep track of objects of vertices"""
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
        
    
def create_faces_3(cat_faces, occ, tet_points: list[TetPoint]):
    """Create the faces of a tetrahedron with 3 different objects.
    
    Args:
        - cat_faces: dict of faces for each object
        - occ: list of tuples of the object ids and the number of times they appear in the tetrahedron
        - tet_points: list of points in the tetrahedron
    
    """
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

def create_faces_2(cat_faces, occ, tet_points: list[TetPoint]):
    """Create the faces of a tetrahedron with 2 different objects. 
    This function serves both for the case of 2 and 2 points for object a and b resp., as for 3 and 1 points for object a and b resp.
    
    Args:
        - cat_faces: the dictionary of faces for each object
        - occ: the occurences of each object in the tetrahedron
        - tet_points: the points in the tetrahedron
    """
    assert len(occ) == 2
    
    most = [p for p in tet_points if p.obj_id == occ[0][0]]
    least = [p for p in tet_points if p.obj_id == occ[1][0]]
        
    face: list[np.ndarray] = []
    for pa in least:
        face += [pa.center(pb) for pb in most]
        
        # Add face to each object cat cell
    if len(np.shape(face)) > 2:
        raise ValueError(f"face {face} has more than 2 dimensions")
    for (k, f) in occ:
        cat_faces[k].append(face)
    
def single_point_4faces(tet_point: TetPoint, others: list[TetPoint], tet_center: np.ndarray):
    """Create the faces of one of the points for a tetrahedron with 4 different objects
    
    Args:
        - tet_point: the point to compute the faces for
        - others: the other 3 points in the tetrahedron
        - tet_center: the center of the tetrahedron
        
    Returns:
        - list of faces, each face is a list of points
    """
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
        
    sorted_points = sort_points_clockwise(points, v0, tet_center)
    cat_faces = []
    for i in range(len(sorted_points)):
        cat_faces.append([tet_center, sorted_points[i], sorted_points[(i+1)%len(sorted_points)]])
    
    return cat_faces
     
def create_faces_4(cat_faces,tet_points: list[TetPoint]):
    """Create the faces of the CAT mesh for the case of 4 objects in the tetrahedron.
    adds to the cat_faces dictionary the faces of the CAT mesh for each object.
    
    Args:
        - tet_points (list[TetPoint]): list of the 4 points of the tetrahedron
        - cat_faces (dict): dictionary of the faces of the CAT mesh for each object
    """
    # tet points are the 4 points of the tetrahedron
    # for each comination of 3 points create a triangle
    # and add it to the list of faces of each object
    triangles = []
    for point in tet_points: point.triangles = []
    i = 0
    combinations = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    for (i, j, k) in combinations:
        triangle = Triangle([tet_points[i].vertex, tet_points[j].vertex, tet_points[k].vertex], [],
                                  [tet_points[i].obj_id, tet_points[j].obj_id, tet_points[k].obj_id])
        triangles.append(triangle)
        # might come from another place where this triangle var is not cleared.
        tet_points[i].add_triangle(triangle)
        tet_points[j].add_triangle(triangle)
        tet_points[k].add_triangle(triangle)
        
    tet_center = sum([point.vertex for point in tet_points]) / 4
    for point in tet_points:
        others = [other for other in tet_points if not (other == point).all()]
        cat = single_point_4faces(point, others, tet_center)
        cat_faces[point.obj_id] += cat
        
    
def compute_cat_cells(object_points_list: list[np.ndarray], container_points: np.ndarray):
    """Compute the CAT cells of the objects in the list and the container.
    First a Tetrahedral mesh is create from the pointcloud of all the objects points and the container poins.
    
    Args:
        - object_points_list: a list of point clouds which define the surface meshes of the objects
        - container_points: a point cloud of surface mesh of the container
    """
    pc = pv.PolyData(np.concatenate((object_points_list + [container_points])))
    tetmesh = pc.delaunay_3d()
    
    # The point sets are sets(uniques) of tuples (x,y,z) for each object, for quick lookup
    # NOTE: Each set in the list might contain points from different objects.
    obj_point_sets = [
        set(map(tuple, obj)) for obj in object_points_list] + [
        set(map(tuple, container_points))
    ]
    
    # Each cat cell is a list of faces, each face is a list of points
    cat_cells = compute_cat_faces(tetmesh, obj_point_sets)
    
    return cat_cells
    
def compute_cat_faces(tetmesh, point_sets: list[set[tuple]]):
    """Compute the CAT faces of the tetrahedron mesh. 
    
    args:
        - tetmesh: a tetrahedron mesh of the container and objects
        - point_sets: a list of sets of points, each set contains points from a single object
    """
    cat_faces = {}
    for obj_id in range(len(point_sets)):
        cat_faces[obj_id] = []

    for cell in range(tetmesh.n_cells):
        occ = {}
        tet_points: list[TetPoint] = []

        for i, obj in enumerate(point_sets):  
            for cell_point in tetmesh.cell_points(cell):
        # check if the cell has points from more than one point set
                if tuple(cell_point) in obj:
                    occ[i] = occ.get(i, 0) + 1
                    tet_points.append(TetPoint(cell_point, i, tet_id=cell))
                

        # sort occ on value
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
            create_faces_2(cat_faces, occ, tet_points)
    return cat_faces

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
    n_entries = 0
    for face in cat_faces0:
        n_entries += len(face) + 1
    
    poly_faces = np.empty(n_entries, dtype=np.int32)

    points = {}
    # points: list[Vertex] = []
    # point_ids: list[int] = []
    counter = 0
    face_len = 0
    idx = 0
    for i, face in tqdm(enumerate(cat_faces0)):
        # poly_face = [len(face)]]
        face_len = len(face)
        poly_faces[idx] = face_len
        # poly_face = np.empty(face_len + 1, dtype=np.float32)
        # poly_face[0] = face_len
        # face_point = 0
        
        idx += 1
        for point in face:
            # if point not in points:
            # true_array = [(old_point == point).all() for old_point in points.keys()]
            if tuple(point) not in points.keys():
            
            # if not np.any(true_array):
                points[tuple(point)] = counter
                cat_points.append(point)
                counter += 1
                
            poly_faces[idx] = points[tuple(point)]
            idx += 1
            
        
        # assert len(poly_face) >= 3, f"Not enough points for a triangle: {poly_face}"
        # poly_faces += poly_face
    return cat_points, poly_faces

# ------------------ #
# Showcase functions
# ------------------ #
def plot_shapes(shape1, shape2, shape3, shape4, rotate, filename=None):
    # Create a plotter object
    plotter = pv.Plotter(shape='2|2')

    # Add the shapes to the plotter
    plotter.subplot(0)
    plotter.add_text("object")
    shape1_r = shape1.copy().rotate_x(rotate[0]).rotate_y(rotate[1]).rotate_z(rotate[2])
    plotter.add_mesh(shape1_r, show_edges=True, color='r',)

    plotter.subplot(1)
    plotter.add_text("delaunay tetrahedra")
    shape2_r = shape2.copy().rotate_x(rotate[0]).rotate_y(rotate[1]).rotate_z(rotate[2])
    plotter.add_mesh(shape2_r, show_edges=True, opacity=0.7)

    plotter.subplot(2)
    plotter.add_text("container")
    shape3_r = shape3.copy().rotate_x(rotate[0]).rotate_y(rotate[1]).rotate_z(rotate[2])
    plotter.add_mesh(shape3_r, show_edges=True, )

    plotter.subplot(3)
    plotter.add_text("CAT faces")
    shape4_r = shape4.copy().rotate_x(rotate[0]).rotate_y(rotate[1]).rotate_z(rotate[2])
    plotter.add_mesh(shape4_r, color='r',)
    plotter.add_mesh(shape2_r, opacity=0.3, show_edges=True, edge_color='b')

    plotter.show()
    if filename:
        plotter.save_graphic(filename)

def main():
    cat_points = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) # . rotates_cube_points
    obj_shape = pv.PolyData(cat_points).delaunay_3d()
    container = pv.Cube(center=(0,0,0), x_length=4, y_length=4, z_length=4)

    pc = pv.PolyData(np.concatenate([cat_points, container.points]))
    tetmesh = pc.delaunay_3d()
    cat_faces = compute_cat_cells([cat_points], container.points)

    cat_points, poly_faces = face_coord_to_points_and_faces(cat_faces[0])
    polydata = pv.PolyData(cat_points, poly_faces)

    plot_shapes(obj_shape, tetmesh.explode(), container, polydata.explode(), (0,0,10))
    
if __name__ == "__main__":
    print("This is an example of the CAT algorithm.")
    main()



# %%
