# %%

    # import numpy as np
    # import pyvista as pv


    # optimizer = Optimizer.default_config()
    # object_meshes = optimizer.current_meshes()
    # optimizer.resample_meshes()
    # normals, cat_cells, normals_pp, tetmesh = optimizer.compute_cat_cells()
    # cells = get_cell_arrays(tetmesh.cells)
    # faceless_points = [i for i, n in enumerate(normals_pp) if len(n) == 0]
    # faceless_vertices = tetmesh.points[np.array(faceless_points)]
    # objects_npoints = [x * optimizer.shape.n_points for x in range(optimizer.n_objs)]
    # rel_cells, _ = filter_relevant_cells(cells, objects_npoints)
    # # %%
    # extracted = tetmesh.extract_cells([cell.id for cell in rel_cells])
    # faceless_rel_cells = tetmesh.extract_points(faceless_points, include_cells=True)
    # faceless_rel_cells_centers = faceless_rel_cells.cell_centers()

    # relevant_cells = []
    # for i, cell_center in enumerate(extracted.cell_centers().points):
    #     # if any of faceless_rel_cells_centers is inside cell, then cell is relevant
    #     if np.isclose(faceless_rel_cells_centers.points, cell_center, atol=1E-5).any():
    #         relevant_cells.append(extracted.cells[i])

    # relevant_cells = tetmesh.extract_cells(np.array(relevant_cells))
    # faceless_cells = extracted.extract_points(faceless_points, include_cells=True)
    # plotter = pv.Plotter()
    # plotter.add_points(faceless_vertices, color="green", point_size=10, render_points_as_spheres=True)
    # plotter.add_mesh(relevant_cells, color="green", opacity=0.5, show_edges=True)
    # # plotter.add_mesh_clip_plane(extracted, crinkle=True, show_edges=True)
    # # plotter.add_mesh_clip_plane(tetmesh, crinkle=False, show_edges=True, color="red", opacity=0.5)
    # for mesh in object_meshes:
    #     plotter.add_mesh(mesh, show_edges=True, opacity=0.5)
    # plotter.show()

# https://stackoverflow.com/questions/73490842/find-cells-from-vertices-in-pyvista-polydata-mesh

# %%
