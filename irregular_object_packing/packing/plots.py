import pyvista as pv


def create_plot(object_locations, object_meshes, object_cells, container_mesh):
    # Create a PyVista plotter object
    plotter = pv.Plotter()

    # Create a container mesh with specified opacity
    container = pv.wrap(container_mesh)
    plotter.add_title("objects with CAT cells in Container", font_size=24, shadow=True)
    plotter.add_mesh(container, opacity=0.2)

    # Loop over objects and create a PyVista mesh for each object
    for i in range(len(object_locations)):
        # object_mesh = pv.PolyData(object_meshes[i])
        object_mesh = pv.wrap(object_meshes[i])
        # object_mesh.transform(np.eye(4), object_locations[i])
        plotter.add_mesh(object_mesh, color="r", opacity=0.7)
        plotter.add_mesh(object_cells[i], color="y", opacity=0.3)

    # Set background color and show the plot
    # plotter.background_color = "black"
    plotter.show()
