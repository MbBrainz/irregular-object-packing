import numpy as np
import pyvista as pv
from sklearn.cluster import KMeans


def resample_pyvista_mesh_kmeans(mesh: pv.PolyData, target_vertices: int):
    # Convert PyVista mesh to NumPy points|
    points = mesh.points

    # Cluster points using KMeans to get the target number of vertices
    kmeans = KMeans(n_clusters=target_vertices)
    kmeans.fit(points)
    new_points = kmeans.cluster_centers_

    # Create a new mesh from the reduced points
    cloud = pv.PolyData(new_points)

    # Regenerate the surface mesh using Delaunay triangulation
    new_mesh = cloud.reconstruct_surface()

    # Extract the surface of the 3D triangulation
    new_mesh = new_mesh.extract_surface()

    # Smooth the mesh
    new_mesh = new_mesh.smooth(n_iter=10)

    return new_mesh


def resample_pyvista_mesh(mesh: pv.PolyData, target_faces):
    """Resample a PyVista mesh to a target number of faces.

    If the number of faces
    of the mesh is less than the target number of faces, we use subdivision,
    If the number of faces of the mesh is greater than the target number of faces,
    decimation is used."""
    # Compute the decimation factor based on the target number of faces
    if mesh.n_faces > target_faces:
        new_mesh = downsample_pv_mesh(mesh, target_faces)
    elif mesh.n_faces < target_faces:
        new_mesh = upsample_pv_mesh(mesh, target_faces)

    # Smooth the mesh
    new_mesh = new_mesh.smooth(n_iter=10)
    return new_mesh


def upsample_pv_mesh(input_mesh: pv.PolyData, target_faces: int):
    """
    Upsample a mesh using Loop subdivision to reach the desired number of faces.

    :param input_mesh: A PyVista PolyData object representing the input mesh
    :param target_faces: The target number of faces in the upsampled mesh
    :return: A PyVista PolyData object representing the upsampled mesh
    """
    num_input_faces = input_mesh.n_faces
    if target_faces <= num_input_faces:
        raise ValueError(
            "Target number of faces should be greater than the input mesh's \
            number of faces.")

    # Calculate the number of required subdivision iterations
    num_subdivisions = int(np.ceil(np.log2(target_faces / num_input_faces) / np.log2(4)))

    # Perform Loop subdivision
    upsampled_mesh = input_mesh.subdivide(num_subdivisions, subfilter='loop')

    return upsampled_mesh


def downsample_pv_mesh(mesh: pv.PolyData, target_faces: int):
    num_faces = mesh.n_faces
    if num_faces < target_faces:
        raise ValueError("Target number of faces must be less than the number of faces \
            in the mesh.")
    decimation_factor = 1 - target_faces / num_faces
    # Decimate the mesh using the decimation factor
    new_mesh = mesh.decimate(decimation_factor, inplace=False)
    return new_mesh


def mesh_simplification_condition(scale_factor: float, alpha: float = 0.05, beta: float = 0.1) -> float:
    """Compute a mesh simplification condition based on the scale factor and the
    parameters alpha and beta.

    :param scale_factor: The scale factor of the object
    :param alpha: The alpha parameter of the mesh simplification condition
    :param beta: The beta parameter of the mesh simplification condition
    :return: The mesh simplification condition
    """
    return alpha * (1 + alpha**(1 / beta) - scale_factor) ** (-beta)


def resample_mesh_by_triangle_area(example_mesh: pv.PolyData, target_mesh: pv.PolyData):
    """Resample a target mesh to match the average triangle area of the example mesh.
    function assumes that both meshes are triangulated surface meshes
    """
    # Compute average triangle area for both meshes
    example_avg_area = compute_average_triangle_area(example_mesh)
    target_avg_area = compute_average_triangle_area(target_mesh)

    # Calculate the desired number of triangles in the target mesh
    target_num_triangles = int(target_mesh.n_faces * (target_avg_area / example_avg_area))

    # Use the decimation algorithm to reduce the number of triangles in the target mesh
    resampled_mesh = resample_pyvista_mesh(target_mesh, target_faces=target_num_triangles)

    return resampled_mesh


def compute_average_triangle_area(mesh: pv.PolyData):
    """Compute the average triangle area of a mesh."""
    return mesh.area / mesh.n_faces
