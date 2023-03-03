import pyvista as pv
from irregular_object_packing.packing import nlc_optimisation
from irregular_object_packing.packing.chordal_axis_transform import face_coord_to_points_and_faces
from irregular_object_packing.packing.growth_based_optimisation import Optimizer


def create_plot(object_locations, object_meshes, object_cells, container_mesh, plotter=None):
    # Create a PyVista plotter object
    if plotter is None:
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


def plot_step_comparison(original_mesh, tf_arrs, cat_cell_mesh_1, cat_cell_mesh_2=None):
    tf_init, tf_fin = tf_arrs
    if cat_cell_mesh_2 is None:
        cat_cell_mesh_2 = cat_cell_mesh_1

    object_mesh = original_mesh.copy()
    post_mesh = original_mesh.copy()

    original_tranform = nlc_optimisation.construct_transform_matrix(tf_init)
    modified_transform = nlc_optimisation.construct_transform_matrix(tf_fin)

    init_mesh = object_mesh.apply_transform(original_tranform)
    post_mesh = post_mesh.apply_transform(modified_transform)

    plotter = pv.Plotter(shape="1|1", notebook=True)  # replace with the filename/path of your first mesh
    plotter.subplot(0)
    plotter.add_title("Initial Placement")
    plotter.add_mesh(init_mesh, color="red", opacity=0.8)
    plotter.add_mesh(cat_cell_mesh_1, color="yellow", opacity=0.4)

    # create the second plot
    # plot2 = pv.Plotter()
    plotter.subplot(1)
    plotter.add_title("Optimized Placement")
    plotter.add_mesh(post_mesh, color="red", opacity=0.8)
    plotter.add_mesh(cat_cell_mesh_2, color="yellow", opacity=0.4)
    plotter.show()

    return plotter


def plot_state(optimizer: Optimizer):
    object_meshes = []
    cat_meshes = []

    data = optimizer.cat_data

    lim = len(data.cat_faces.keys())
    for k, v in data.cat_faces.items():
        if k >= lim - 1:
            break
        cat_points, poly_faces = face_coord_to_points_and_faces(data, k)
        polydata = pv.PolyData(cat_points, poly_faces)
        cat_meshes.append(polydata)

        object_mesh = optimizer.meshes[0].copy()

        transform_matrix = nlc_optimisation.construct_transform_matrix(optimizer.transform_data[k])
        object_mesh = object_mesh.apply_transform(transform_matrix)

        object_meshes.append(object_mesh)

    create_plot(optimizer.transform_data, object_meshes, cat_meshes, optimizer.container.to_mesh())
