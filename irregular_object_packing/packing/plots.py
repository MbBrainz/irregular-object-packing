from numpy import ndarray
import pyvista as pv
from trimesh import Trimesh
import trimesh
from irregular_object_packing.packing import nlc_optimisation
from irregular_object_packing.packing.chordal_axis_transform import face_coord_to_points_and_faces

# from irregular_object_packing.packing.growth_based_optimisation import Optimizer


def create_plot(object_locations, object_meshes: list[pv.PolyData], object_cells, container_mesh, plotter=None):
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
        plotter.add_mesh(object_mesh.decimate(0.1), color="r", opacity=0.7)

        plotter.add_mesh(object_cells[i], color="y", opacity=0.3)

    # Set background color and show the plot
    # plotter.background_color = "black"
    plotter.show()


def plot_full_comparison(
    meshes_before,
    meshes_after,
    cat_cell_meshes,
    container,
    plotter=None,
    title_left="Initial Placement",
    title_right="Improved Placement",
):
    if plotter is None:
        plotter = pv.Plotter()

    plotter = pv.Plotter(shape="1|1", notebook=True)

    colors = generate_tinted_colors(len(meshes_before))
    plotter.subplot(0)
    plotter.add_title(title_left)
    plotter.add_mesh(container, opacity=0.3)
    for i, mesh in enumerate(meshes_before):
        plotter.add_mesh(mesh, color=colors[1][i], opacity=0.9)

    for i, cat_mesh in enumerate(cat_cell_meshes):
        plotter.add_mesh(cat_mesh, color=colors[0][i], opacity=0.5)
        open_edges = cat_mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        if open_edges.n_points > 0:
            plotter.add_mesh(open_edges, color="black", line_width=1)

    plotter.subplot(1)
    plotter.add_title(title_right)
    plotter.add_mesh(container, opacity=0.3)
    for i, mesh in enumerate(meshes_after):
        plotter.add_mesh(mesh, color=colors[1][i], opacity=0.9)

    for i, cat_mesh in enumerate(cat_cell_meshes):
        plotter.add_mesh(cat_mesh, color=colors[0][i], opacity=0.5)
    plotter.show()

    return plotter


# def plot_step_comparison(original_mesh: Trimesh, tf_arrs, cat_cell_mesh_1, cat_cell_mesh_2=None):
def plot_step_comparison(
    mesh_before,
    mesh_after,
    cat_cell_mesh_1,
    cat_cell_mesh_2=None,
    plotter=None,
    title_left="Initial Placement",
    title_right="Improved Placement",
):
    if plotter is None:
        plotter = pv.Plotter()

    if cat_cell_mesh_2 is None:
        cat_cell_mesh_2 = cat_cell_mesh_1

    plotter = pv.Plotter(shape="1|1", notebook=True)  # replace with the filename/path of your first mesh
    plotter.subplot(0)
    plotter.add_title(title_left)
    plotter.add_mesh(mesh_before, color="red", opacity=0.8)
    open_edges = cat_cell_mesh_1.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
    plotter.add_mesh(open_edges, color="black", line_width=1, opacity=0.8)
    plotter.add_mesh(cat_cell_mesh_1, color="yellow", opacity=0.4)

    # create the second plot
    # plot2 = pv.Plotter()
    plotter.subplot(1)
    plotter.add_title(title_right)
    plotter.add_mesh(mesh_after, color="red", opacity=0.8)
    plotter.add_mesh(cat_cell_mesh_2, color="yellow", opacity=0.4)
    plotter.show()

    return plotter


def generate_tinted_colors(num_tints, base_color_1="FFFF00", base_color_2="FF0000"):
    """
    Generates two lists of hex colors with corresponding tints.
    """
    # Convert the base colors to RGB format
    base_color_1_rgb = tuple(int(base_color_1[i : i + 2], 16) for i in (0, 2, 4))
    base_color_2_rgb = tuple(int(base_color_2[i : i + 2], 16) for i in (0, 2, 4))

    # Calculate the step size for the tints
    step_size = 255 // (num_tints + 1)

    # Initialize the lists of tinted colors
    tinted_colors_1 = []
    tinted_colors_2 = []

    # Generate the tinted colors
    for i in range(1, num_tints + 1):
        tint_1_rgb = tuple(min(base_color_1_rgb[j] + i * step_size, 255) for j in range(3))
        tint_2_rgb = tuple(min(base_color_2_rgb[j] + i * step_size, 255) for j in range(3))

        # Convert the tinted colors back to hex format and add them to the lists
        tinted_color_1_hex = f"#{''.join(hex(c)[2:].zfill(2) for c in tint_1_rgb)}"
        tinted_color_2_hex = f"#{''.join(hex(c)[2:].zfill(2) for c in tint_2_rgb)}"

        tinted_colors_1.append(tinted_color_1_hex)
        tinted_colors_2.append(tinted_color_2_hex)

    return tinted_colors_1, tinted_colors_2


def create_packed_scene(
    container: trimesh.Trimesh,
    objects_coords: list[ndarray],
    mesh: trimesh.Trimesh,
    mesh_scale: float = 1,
    rotate: bool = False,
):
    """make a trimesh scene with the container and the objects inside.

    Args:
        container (trimesh.Trimesh): container mesh
        objects_coords (List[np.ndarray]): list of coordinates of the objects
        mesh (trimesh.Trimesh): mesh of the objects
        mesh_scale (float, optional): scale of the objects. Defaults to 1.
    """
    objects = []
    for coord in objects_coords:
        new_mesh = mesh.copy()
        if rotate:
            new_mesh = new_mesh.apply_transform(trimesh.transformations.random_rotation_matrix())

        new_mesh.apply_scale(mesh_scale).apply_translation(coord)

        new_mesh.visual.vertex_colors = trimesh.visual.random_color()
        objects.append(new_mesh)

    container.visual.vertex_colors = [250, 255, 255, 100]

    objects.append(container)
    scene = trimesh.Scene(objects)

    return scene
