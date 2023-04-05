import pickle

import numpy as np


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
    """"""
    cat_faces: dict
    """shape: {[obj_id]: {[point_id]: list[face]} } A dictionary of the faces of the CAT
    The keys are the object ids and the values are dictionaries with the keys being
    point ids and the values being a list of all faces relevant to that point."""
    cat_cells: dict
    """shape: {[obj_id]: list[face]} A dictionary of the cells of the CAT.
    The keys are the object ids and the values are a list of all faces of the cat cell
    belonging to that object."""
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
        """A cell is defined by a list of faces which together make up the CAT cell."""
        for face in faces:
            assert isinstance(face, tuple), f"face is not a tuple: {face}"
            self.add_cat_face_to_cell(obj_id, face)

    def add_cat_faces_to_point(
        self, point: TetPoint, faces: list[tuple[list[int], np.ndarray]]
    ) -> None:
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
