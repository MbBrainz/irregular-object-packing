from copy import copy
from dataclasses import dataclass

from numpy import ndarray
from pandas import DataFrame
from pyvista import PolyData

from irregular_object_packing.packing.chordal_axis_transform import (
    CatData,
    face_coord_to_points_and_faces,
)
from irregular_object_packing.packing.nlc_optimisation import construct_transform_matrix


@dataclass
class IterationData:
    i: int
    """The iteration step."""
    i_b: int
    """The scale iteration step."""
    starting_f: float
    """The starting value of the scale."""
    max_f: float
    """The maximum value of the scale."""
    n_succes_scale: int
    """The number of objects that have succesfully been scaled to the current limit."""


class OptimizerData:
    """Data structure for conveniently getting per-step meshes from the data generated
    by the optimizer."""

    _data = {}
    _index = -1

    def __init__(self):
        pass

    def __getitem__(self, key):
        return self._data[key]

    def add(self, tf_arrays: ndarray, cat_data: None | CatData, iteration_data: IterationData):
        self._data[self._index] = {
            "tf_arrays": tf_arrays.copy(),
            "cat_data": copy(cat_data),
            "iterationData": iteration_data,
        }
        # self._data[ref] = self._data[self._index]
        self._index += 1

    def _tf_arrays(self, index: int):
        return self._data[index]["tf_arrays"]

    def _cat_data(self, index: int) -> CatData:
        return self._data[index]["cat_data"]

    def _get_mesh(self, index: int, obj_id: int, mesh: PolyData) -> PolyData:
        return mesh.transform(
            construct_transform_matrix(self._tf_arrays(index)[obj_id]),
            inplace=False,
        )

    def _get_meshes(self, index: int, mesh: PolyData) -> list[PolyData]:
        return [
            mesh.transform(
                construct_transform_matrix(tf_array),
                inplace=False,
            )
            for tf_array in self._tf_arrays(index)
        ]

    @property
    def idx(self):
        return self._index - 1

    # ------------------- Public methods -------------------
    def mesh_before(self, iteration: int, obj_id: int, mesh: PolyData):
        """Get the mesh of the object at the given iteration, before the
        optimisation."""
        return self._get_mesh(iteration - 1, obj_id, mesh)

    def mesh_after(self, iteration: int, obj_id: int, mesh: PolyData):
        """Get the mesh of the object at the given iteration, after the optimisation."""
        return self._get_mesh(iteration, obj_id, mesh)

    def cat_mesh(self, iteration: int, obj_id: int) -> PolyData:
        """Get the mesh of the cat cell that corresponds to the object from the given
        iteration."""
        return PolyData(
            *face_coord_to_points_and_faces(self._cat_data(iteration), obj_id)
        )

    def meshes_before(self, iteration: int, mesh: PolyData):
        """Get the meshes of all objects at the given iteration, before the
        optimisation."""
        if iteration < 0:
            return ValueError("No meshes before iteration 0")
        return self._get_meshes(iteration - 1, mesh)

    def meshes_after(self, iteration: int, mesh: PolyData):
        """Get the meshes of all objects at the given iteration, after the
        optimisation."""
        return self._get_meshes(iteration, mesh)

    def cat_meshes(self, iteration: int) -> list[PolyData]:
        """Get the meshes of all cat cells that correspond to the objects from the given
        iteration."""
        if self._cat_data(iteration) is None:
            raise ValueError("No cat data stored yet for iteration " + str(iteration))
        return [
            PolyData(*face_coord_to_points_and_faces(self._cat_data(iteration), obj_id))
            for obj_id in range(len(self._tf_arrays(iteration)))
        ]

    def final_meshes_after(self, mesh: PolyData):
        """Get the meshes of all objects with the most recent transformation."""
        if self._index < 0:
            ValueError("No data stored yet")

        if self._index == 0:
            return self._get_meshes(-1, mesh)

        return self._get_meshes(self.idx, mesh)

    def final_meshes_before(self, mesh: PolyData):
        """Get the meshes of all objects at the final iteration, before the
        optimisation."""
        return self._get_meshes(self.idx - 1, mesh)

    def final_cat_meshes(self):
        """Get the meshes of all cat cells that correspond to the objects from the final
        iteration."""
        if self._index <= 0:
            ValueError("No cat data stored yet")
        return self.cat_meshes(self.idx)

    def before_and_after_meshes(self, iteration: int, mesh: PolyData):
        """Get the meshes of all objects at the given iteration, before and after the
        optimisation."""
        return (
            self.meshes_before(iteration, mesh),
            self.meshes_after(iteration, mesh),
            self.cat_meshes(iteration),
        )

    def _report(self, iteration=None):
        if iteration is None:
            iteration = self.idx
        df = DataFrame(
            data=[self._tf_arrays(i) for i in range(self.idx)],
            columns=["scale", "r_x", "ry", "rz", "t_x", "t_y", "t_z"],
        )
        return df
