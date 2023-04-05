import numpy as np
import pyvista as pv

# def test_nlcp_facets():

# Define cube object
cube = pv.Cube()
small_cube = pv.Cube().scale(0.5 ** (1 / 3))
smallest_cube = pv.Cube().scale(0.1 ** (1 / 3))
points = cube.points.tolist() + small_cube.points.tolist() + smallest_cube.points.tolist()
pc = pv.PolyData(points)

from meshpy.tet import MeshInfo, Options, build

tetra = MeshInfo()
tetra.set_points(points)
opts = Options("Q")
res = build(tetra, diagnose=True, options=opts)

vertices = np.array(res.points)
faces = np.hstack([[4, *face] for face in res.elements])
pv_mesh = pv.UnstructuredGrid(faces, [pv.CellType.TETRA] * len(res.elements), vertices)

# Visualize the mesh
plotter = pv.Plotter()
plotter.add_mesh(pv_mesh, show_edges=True, color="white", opacity=0.7)
plotter.add_mesh(pc, color="red", point_size=10)
plotter.show_grid()
plotter.show()
