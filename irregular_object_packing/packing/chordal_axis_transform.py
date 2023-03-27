""" In step 4.2.1 of the algorithm in [Ma et al. 2018], The CAT is computed by using the following steps:
1. Create a Tetrahedron Mesh from all the points on the surface mesh of both the objects and the container as the input points.
2. Use onlty those tetrahedrons that constructed of points from multiple objects.
3. Compute the chordal axis of each tetrahedron.
4. Compute the chordal axis of the whole object by taking the union of all the chordal axis of the tetrahedrons.
"""
# %%
from dataclasses import dataclass

import pickle

import numpy as np
import pyvista as pv

# from utils import angle_between, sort_points_clockwise
from irregular_object_packing.packing.utils import (
    sort_points_clockwise,
    split_quadrilateral_to_triangles,
    compute_face_normal,
)


class TetPoint:
    vertex: np.ndarray
    obj_id: int
    p_id: int
    tet_id: int

    def __init__(self, point: np.ndarray, p_id, obj_id=-1, tet_id=-1):
        self.vertex = point
        self.p_id = p_id
        self.obj_id = obj_id
        self.tet_id = tet_id

    def __eq__(self, other):
        return self.vertex == other.vertex

    def __add__(self, o: object) -> "TetPoint":
        return TetPoint(self.vertex + o.vertex, self.tet_id)

    def __div__(self, o: object) -> np.ndarray:
        return self.vertex / o

    def __truediv__(self, other):
        return self.__div__(other)

    def same_obj(self, other: "TetPoint"):
        return self.obj_id == other.obj_id

    def distance(self, other: "TetPoint"):
        return np.linalg.norm(self.vertex - other.vertex)

    def center(self, other: "TetPoint") -> np.ndarray:
        return (self.vertex + other.vertex) / 2


class CatData:
    """A class to hold the data for the CAT algorithm."""

    points: dict
    point_ids: dict
    cat_faces: dict
    """shape: {[obj_id]: {[point_id]: list[face]} } A dictionary of the faces of the CAT. The keys are the object ids and the values are dictionaries with the keys being point ids and the values being a list of all faces relevant to that point."""
    cat_cells: dict
    """shape: {[obj_id]: list[face]} A dictionary of the cells of the CAT. The keys are the object ids and the values are a list of all faces of the cat cell belonging to that object."""
    object_coords: np.ndarray

    def __init__(self, point_sets: list[set[tuple]], object_coords: np.ndarray):
        self.points = {}
        self.point_ids = {}
        self.cat_faces = {}
        self.cat_cells = {}
        self.object_coords = object_coords

        for obj_id in range(len(point_sets)):
            self.cat_cells[obj_id] = []
            self.cat_faces[obj_id] = {}

            for point in point_sets[obj_id]:
                self.add_obj_point(obj_id, point)

    def point_id(self, point: np.ndarray) -> int:
        point = tuple(point)
        n_points = len(self.points)
        p_id = self.point_ids.get(point, n_points)
        if p_id == n_points:
            self.add_point(point, p_id)

        return p_id

    def point(self, p_id: int) -> tuple:
        return self.points[p_id]

    def new_point(self, point: tuple) -> int:
        p_id = len(self.points)
        self.add_point(point, p_id)
        return p_id

    def add_obj_point(self, obj_id: int, point: tuple) -> None:
        p_id = self.new_point(point)
        self.cat_faces[obj_id][p_id] = []

    def add_point(self, point: tuple, p_id: int) -> None:
        self.points[p_id] = point
        self.point_ids[point] = p_id

    def add_cat_face_to_cell(self, obj_id: int, face: list[int]) -> None:
        self.cat_cells[obj_id].append(face)

    def add_cat_face_to_point(self, point: TetPoint, face: list[int]) -> None:
        self.cat_faces[point.obj_id][point.p_id].append(face)

    def add_cat_faces_to_cell(self, obj_id: int, faces: list[list[int]]) -> None:
        for face in faces:
            assert isinstance(face, tuple), f"face is not a tuple: {face}"
            self.add_cat_face_to_cell(obj_id, face)

    def add_cat_faces_to_point(self, point: TetPoint, faces: list[tuple[list[int], np.ndarray]]) -> None:
        for face in faces:
            assert isinstance(face, tuple), f"face is not a tuple: {face}"

            self.add_cat_face_to_point(point, face)

    def get_face(self, face: list[int]) -> list[np.ndarray]:
        return [np.array(self.point(p_id)) for p_id in face]

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> "CatData":
        with open(filename, "rb") as file:
            return pickle.load(file)


def extract_points_of_interest(data: CatData, obj_id: int, translation: np.ndarray):
    point_ids = set(np.concat(data.cat_faces[obj_id]))
    points = {p_id: data.points[p_id] - translation for p_id in point_ids}
    return points


def cat_mesh_from_data(cat_data: CatData, k: int) -> pv.PolyData:
    cat_points, poly_faces = face_coord_to_points_and_faces(cat_data, k)
    return pv.PolyData(cat_points, poly_faces)


def create_faces_3(data: CatData, occ, tet_points: list[TetPoint]):
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

    # The naming assumes that point a and b are most[0] and most[1] respectively
    # and point c and d are least[0] and least[1] respectively
    acd_point = data.point_id((most[0].vertex + least[0].vertex + least[1].vertex) / 3)
    bcd_point = data.point_id((most[1].vertex + least[0].vertex + least[1].vertex) / 3)

    ac_point = data.point_id((most[0].vertex + least[0].vertex) / 2)
    ad_point = data.point_id((most[0].vertex + least[1].vertex) / 2)
    cd_point = data.point_id((least[0].vertex + least[1].vertex) / 2)
    bc_point = data.point_id((most[1].vertex + least[0].vertex) / 2)
    bd_point = data.point_id((most[1].vertex + least[1].vertex) / 2)

    most_face_c = [acd_point, ac_point, bcd_point, bc_point]
    most_face_d = [acd_point, ad_point, bcd_point, bd_point]
    n_mfc = compute_face_normal(data.get_face(most_face_c), most[0].vertex)  # can be both most[0] and most[1]
    n_mfd = compute_face_normal(data.get_face(most_face_d), most[0].vertex)
    mfc_1, mfc_2 = split_quadrilateral_to_triangles(most_face_c)
    mfd_1, mfd_2 = split_quadrilateral_to_triangles(most_face_d)
    most_faces = [(mfc_1, n_mfc), (mfc_2, n_mfc), (mfd_1, n_mfd), (mfd_2, n_mfd)]

    l_face = [cd_point, acd_point, bcd_point]
    n_l0_face = compute_face_normal(data.get_face(l_face), least[0].vertex)
    n_l1_face = n_l0_face * -1
    l0_faces = [(l_face, n_l0_face), (mfc_1, n_mfc * -1), (mfc_2, n_mfc * -1)]
    l1_faces = [(l_face, n_l1_face), (mfd_1, n_mfd * -1), (mfd_2, n_mfd * -1)]

    # Add faces to cells and to BOTH the points of max
    data.add_cat_faces_to_cell(most[0].obj_id, most_faces)
    data.add_cat_faces_to_point(most[0], most_faces)
    data.add_cat_faces_to_point(most[1], most_faces)

    data.add_cat_faces_to_cell(least[0].obj_id, l0_faces)
    data.add_cat_faces_to_point(least[0], l0_faces)

    data.add_cat_faces_to_cell(least[1].obj_id, l1_faces)
    data.add_cat_faces_to_point(least[1], l1_faces)


def create_faces_2(data: CatData, occ, tet_points: list[TetPoint]):
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

    if len(most) == 2:
        ab_point = data.point_id((most[0].vertex + least[0].vertex) / 2)
        ac_point = data.point_id((most[0].vertex + least[1].vertex) / 2)
        bd_point = data.point_id((most[1].vertex + least[0].vertex) / 2)
        cd_point = data.point_id((most[1].vertex + least[1].vertex) / 2)

        face = [ab_point, ac_point, cd_point, bd_point]
        n_mface = compute_face_normal(data.get_face(face), most[0].vertex)
        n_lface = n_mface * -1
        face_0, face_1 = split_quadrilateral_to_triangles(face)
        # faces = [[ab_point, ac_point, cd_point], [cd_point, bd_point, ab_point]]
        m_faces = [(face_0, n_mface), (face_1, n_mface)]
        l_faces = [(face_0, n_lface), (face_1, n_lface)]

        data.add_cat_faces_to_cell(most[0].obj_id, m_faces)
        data.add_cat_faces_to_point(most[0], m_faces)
        data.add_cat_faces_to_point(most[1], m_faces)

        data.add_cat_faces_to_cell(least[0].obj_id, l_faces)
        data.add_cat_faces_to_point(least[0], l_faces)
        data.add_cat_faces_to_point(least[1], l_faces)

    if len(most) == 3:
        ab_point = data.point_id((least[0].vertex + most[0].vertex) / 2)
        ac_point = data.point_id((least[0].vertex + most[1].vertex) / 2)
        ad_point = data.point_id((least[0].vertex + most[2].vertex) / 2)

        face = [ab_point, ac_point, ad_point]
        n_mface = compute_face_normal(data.get_face(face), most[0].vertex)
        n_lface = n_mface * -1
        face = (face, n_mface)

        data.add_cat_face_to_cell(least[0].obj_id, face)
        data.add_cat_face_to_point(least[0], face)

        data.add_cat_face_to_cell(most[0].obj_id, face)
        data.add_cat_face_to_point(most[0], face)
        data.add_cat_face_to_point(most[1], face)
        data.add_cat_face_to_point(most[2], face)


def compute_cat_cells(
    object_points_list: list[np.ndarray], container_points: np.ndarray, obj_coords: list[np.ndarray]
):
    """Compute the CAT cells of the objects in the list and the container.
    First a Tetrahedral mesh is created from the pointcloud of all the objects points and the container points.
    Then, for each tetrahedron that has points from at least 2 different objects, the faces of the CAT mesh are computed.


    Args:
        - object_points_list: a list of point clouds which define the surface meshes of the objects
        - container_points: a point cloud of surface mesh of the container

    Returns:
        - dictionary of the CAT cells for each object.
    """
    pc = pv.PolyData(np.concatenate((object_points_list + [container_points])))
    tetmesh = pc.delaunay_3d()

    # The point sets are sets(uniques) of tuples (x,y,z) for each object, for quick lookup
    # NOTE: Each set in the list might contain points from different objects.
    obj_point_sets = [set(map(tuple, obj)) for obj in object_points_list] + [set(map(tuple, container_points))]

    # Each cat cell is a list of faces, each face is a list of points
    cat_cells = compute_cat_faces(tetmesh, obj_point_sets, obj_coords)

    return cat_cells


def create_faces_4(data: CatData, tet_points: list[TetPoint]):
    center_point = data.point_id(sum([point.vertex for point in tet_points]) / 4)
    ab_point = data.point_id((tet_points[0].vertex + tet_points[1].vertex) / 2)
    ac_point = data.point_id((tet_points[0].vertex + tet_points[2].vertex) / 2)
    ad_point = data.point_id((tet_points[0].vertex + tet_points[3].vertex) / 2)
    bc_point = data.point_id((tet_points[1].vertex + tet_points[2].vertex) / 2)
    cd_point = data.point_id((tet_points[2].vertex + tet_points[3].vertex) / 2)
    bd_point = data.point_id((tet_points[1].vertex + tet_points[3].vertex) / 2)

    abc_point = data.point_id((tet_points[0].vertex + tet_points[1].vertex + tet_points[2].vertex) / 3)
    abd_point = data.point_id((tet_points[0].vertex + tet_points[1].vertex + tet_points[3].vertex) / 3)
    acd_point = data.point_id((tet_points[0].vertex + tet_points[2].vertex + tet_points[3].vertex) / 3)
    bcd_point = data.point_id((tet_points[1].vertex + tet_points[2].vertex + tet_points[3].vertex) / 3)

    faces_a = [
        [center_point, ab_point, abc_point],
        [center_point, abc_point, ac_point],
        [center_point, ac_point, acd_point],
        [center_point, acd_point, ad_point],
        [center_point, ad_point, abd_point],
        [center_point, abd_point, ab_point],
    ]
    faces_a = [(face, compute_face_normal(data.get_face(face), tet_points[0].vertex)) for face in faces_a]

    faces_b = [
        [center_point, ab_point, abc_point],
        [center_point, abc_point, bc_point],
        [center_point, bc_point, bcd_point],
        [center_point, bcd_point, bd_point],
        [center_point, bd_point, abd_point],
        [center_point, abd_point, ab_point],
    ]
    faces_b = [(face, compute_face_normal(data.get_face(face), tet_points[1].vertex)) for face in faces_b]

    faces_c = [
        [center_point, ac_point, abc_point],
        [center_point, abc_point, bc_point],
        [center_point, bc_point, bcd_point],
        [center_point, bcd_point, cd_point],
        [center_point, cd_point, acd_point],
        [center_point, acd_point, ac_point],
    ]
    faces_c = [(face, compute_face_normal(data.get_face(face), tet_points[2].vertex)) for face in faces_c]

    faces_d = [
        [center_point, ad_point, abd_point],
        [center_point, abd_point, bd_point],
        [center_point, bd_point, bcd_point],
        [center_point, bcd_point, cd_point],
        [center_point, cd_point, acd_point],
        [center_point, acd_point, ad_point],
    ]
    faces_d = [(face, compute_face_normal(data.get_face(face), tet_points[3].vertex)) for face in faces_d]

    # Adds the faces the list of each ovbject
    data.add_cat_faces_to_cell(tet_points[0].obj_id, faces_a)
    data.add_cat_faces_to_cell(tet_points[1].obj_id, faces_b)
    data.add_cat_faces_to_cell(tet_points[2].obj_id, faces_c)
    data.add_cat_faces_to_cell(tet_points[3].obj_id, faces_d)

    data.add_cat_faces_to_point(tet_points[0], faces_a)
    data.add_cat_faces_to_point(tet_points[1], faces_b)
    data.add_cat_faces_to_point(tet_points[2], faces_c)
    data.add_cat_faces_to_point(tet_points[3], faces_d)


def compute_cat_faces(tetmesh, point_sets: list[set[tuple]], obj_coords: list[np.ndarray]):
    """Compute the CAT faces of the tetrahedron mesh, by checking which tetrahedrons
    have points from more than one object and splitting those according to figure 2 from the main paper.

    args:
        - tetmesh: a tetrahedron mesh of the container and objects
        - point_sets: a list of sets of points, each set contains points from a single object
    """
    data = CatData(point_sets, obj_coords)

    # FOR EACH TETRAHEDRON
    for cell in range(tetmesh.n_cells):
        occ = {}
        tet_points: list[TetPoint] = []

        for i, obj in enumerate(point_sets):
            for cell_point in tetmesh.cell_points(cell):
                # check if the cell has points from more than one point set
                if tuple(cell_point) in obj:
                    occ[i] = occ.get(i, 0) + 1
                    tet_points.append(
                        TetPoint(cell_point, data.point_ids[tuple(cell_point)], obj_id=i, tet_id=cell)
                    )

        # sort occ on value
        assert len(tet_points) == 4, f"tet_points: {tet_points}"  # lil check
        occ = sorted(occ.items(), key=lambda x: x[1], reverse=True)
        n_objs = len(occ)

        if n_objs == 1:
            continue  # skip cells that have points from only one object

        if n_objs == 4:  # [1,1,1,1]
            create_faces_4(data, tet_points)

        if n_objs == 3:  # [2,1,1,0], [1,2,1,0], [1,1,2,0]:
            create_faces_3(data, occ, tet_points)

        if n_objs == 2:  # [2,2,0,0], [1,3,0,0], [3,1,0,0]
            create_faces_2(data, occ, tet_points)

    return data


def face_coord_to_points_and_faces(data: CatData, obj_id: int):
    # FIXME: This function is not functional anymore. ill fix it later
    """Convert a list of triangular only faces represented by points with coordinates
    to a list of points and a list of faces represented by the number of points and point ids.

    This function is used to convert the data so that it can be used by the pyvista.PolyData class.

    Note: Currently this function assumes that the indices of the points are not global with respect to other meshes.
    """
    cat_faces = data.cat_cells[obj_id]
    cat_points = []
    poly_faces = []
    n_entries = 0
    for face, _n_face in cat_faces:
        len_face = len(face)
        assert len_face == 3, f"len_face: {len_face}"
        if len_face == 3:
            n_entries += 4
        if len_face == 4:
            n_entries += 5
            # n_entries += 8

    poly_faces = np.empty(n_entries, dtype=np.int32)

    points = {}

    counter = 0
    face_len = 0
    idx = 0

    new_faces = []

    for i, (face, _n_face) in enumerate(cat_faces):
        face_len = len(face)
        new_face = np.zeros(face_len)
        for i in range(len(face)):
            if face[i] not in points.keys():
                points[data.point(face[i])] = counter
                cat_points.append(data.point(face[i]))
                counter += 1

            new_face[i] = points[data.point(face[i])]

        # For a face with 4 points, we create 2 triangles,
        # Because pyvista does not support quads correctly, while it says it does.
        # The issue is that when you supply a quad, it will create 2 triangles,
        # but the triangles will overlap by half, like an open envelope shape.
        if face_len == 3:
            poly_faces[idx] = 3
            poly_faces[idx + 1] = new_face[0]
            poly_faces[idx + 2] = new_face[1]
            poly_faces[idx + 3] = new_face[2]
            idx += 4

        elif face_len == 4:  # make a quadrillateral
            poly_faces[idx] = 4
            poly_faces[idx + 1] = new_face[0]
            poly_faces[idx + 2] = new_face[1]
            poly_faces[idx + 3] = new_face[2]
            poly_faces[idx + 4] = new_face[3]
            idx += 5

        # elif face_len == 4:  # make 2 triangles for each quad
        #     poly_faces[idx] = 3
        #     poly_faces[idx + 1] = face[0]
        #     poly_faces[idx + 2] = face[1]
        #     poly_faces[idx + 3] = face[2]
        #     poly_faces[idx + 4] = 3
        #     poly_faces[idx + 5] = face[2]
        #     poly_faces[idx + 6] = face[3]
        #     poly_faces[idx + 7] = face[1]
        #     idx += 8

    return cat_points, poly_faces


# ------------------ #
# Showcase functions
# ------------------ #
def plot_shapes(
    objects: list[pv.PolyData],
    container: pv.PolyData,
    tetrahedra: pv.PolyData,
    cat_meshes: list[pv.PolyData],
    rotate,
    filename=None,
):
    # Create a plotter object
    plotter = pv.Plotter(shape="3|1")

    # Add the shapes to the plotter
    plotter.subplot(0)
    plotter.add_text("object")
    for obj in objects:
        plotter.add_mesh(obj, show_edges=True, color="r")

    plotter.subplot(1)
    plotter.add_text("container")
    plotter.add_mesh(container, show_edges=True, color="b")

    plotter.subplot(2)
    plotter.add_text("delaunay tetrahedra")
    plotter.add_mesh(tetrahedra, show_edges=True, color="g", opacity=0.5)

    plotter.subplot(3)
    plotter.add_text("CAT faces")
    for mesh in cat_meshes:
        plotter.add_mesh(mesh, show_edges=True, color="y", opacity=0.8)

    for obj in objects:
        plotter.add_mesh(obj, show_edges=True, color="r", opacity=0.5)

    plotter.add_mesh(container, show_edges=True, color="gray", opacity=0.1)

    plotter.show()
    if filename:
        plotter.save_graphic(filename)


def main():
    box1 = pv.Cube(center=(-1, -1, 0), x_length=1, y_length=1, z_length=1)
    box2 = pv.Cube(center=(1, 1, 0), x_length=1, y_length=1, z_length=1)
    box3 = pv.Cube(center=(1, -1, 0), x_length=1, y_length=1, z_length=1)
    box4 = pv.Cube(center=(-1, 1, 0), x_length=1, y_length=1, z_length=1)
    boxes = [box1, box2, box3, box4]

    obj_shape = pv.PolyData(box1.points).delaunay_3d()
    container = pv.Cube(center=(0, 0, 0), x_length=4, y_length=4, z_length=3)
    # container = pv.Pyramid([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

    pc = pv.PolyData(np.concatenate([box1.points, box2.points, box3.points, box4.points, container.points]))
    tetmesh = pc.delaunay_3d()
    data = compute_cat_cells([box1.points, box2.points, box3.points, box4.points], container.points, [0, 0, 0])

    # cat_4_faces = [face for face in cat_faces[0] if len(face) == 4]
    cat_box1 = pv.PolyData(*face_coord_to_points_and_faces(data, 0))
    cat_box2 = pv.PolyData(*face_coord_to_points_and_faces(data, 1))
    cat_box3 = pv.PolyData(*face_coord_to_points_and_faces(data, 2))
    cat_box4 = pv.PolyData(*face_coord_to_points_and_faces(data, 3))
    cat_boxes = [cat_box1, cat_box2, cat_box3, cat_box4]

    plotter = pv.Plotter()

    plotter.add_mesh(cat_box1.explode(), show_edges=True, color="r", opacity=0.7)
    plotter.add_mesh(box1, show_edges=True, color="b")
    # plotter.add_mesh(tetmesh.explode(), show_edges=True, color="w", opacity=0.2)
    # plotter.set_focus(box1.center)

    plotter.show()
    plot_shapes(boxes, container, tetmesh.explode(), cat_boxes, (0, 0, 10))


if __name__ == "__main__":
    print("This is an example of the CAT algorithm.")
    main()

# %%
