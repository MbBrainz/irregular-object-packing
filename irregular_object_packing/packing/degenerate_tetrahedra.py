import warnings
import re
import numpy as np
import trimesh
import pyvista as pv
import io
import contextlib
import re
import vtk

import numpy as np
import irregular_object_packing.packing.nlc_optimisation as nlc
from irregular_object_packing.packing.growth_based_optimisation import Optimizer
import pyvista as pv


def count_degenerate_tetrahedra(tetmesh, threshold=1e-5):
    """
    Count degenerate tetrahedra in the given tetrahedral mesh by checking the volume of each tetrahedron.

    Parameters:
    -----------
    tetmesh : pyvista.UnstructuredGrid
        The tetrahedral mesh.
    threshold : float
        The volume threshold to consider a tetrahedron degenerate.

    Returns:
    --------
    int
        The number of degenerate tetrahedra in the mesh.
    """
    num_degenerate = 0
    for i in range(tetmesh.n_cells):
        tetra = tetmesh.extract_cells([i])
        volume = tetra.volume

        if abs(volume) < threshold:
            num_degenerate += 1

    return num_degenerate


def get_degenerate_triangles(optimizer, container_sample_rate, mesh_sample_rate):
    container_points = trimesh.sample.sample_surface_even(optimizer.container, container_sample_rate)[0]
    sample_points = trimesh.sample.sample_surface_even(optimizer.shape, mesh_sample_rate)[0]

    obj_points = [
        trimesh.transform_points(sample_points.copy(), nlc.construct_transform_matrix(transform_data))
        for transform_data in optimizer.tf_arrs
    ]

    pc = pv.PolyData(np.concatenate((obj_points + [container_points])))

    tetmesh = pc.delaunay_3d(tol=1e-6)

    return count_degenerate_tetrahedra(tetmesh)


# %%
def main():
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    print("WARNING: This may take around 10 minutes to run. Adjust the sample rates to reduce the runtime.")

    optimizer = Optimizer.default_setup()
    # List of sample rates to test
    mesh_sample_rates = np.linspace(100, 1000, 10, dtype=int)  # Adjust these values as needed
    container_sample_rates = np.linspace(100, 1000, 10, dtype=int)  # Adjust these values as needed

    # Initialize an empty array to store the number of degenerate triangles
    degenerate_triangle_counts = np.zeros((len(mesh_sample_rates), len(container_sample_rates)))

    # Iterate over all combinations of mesh_sample_rate and container_sample_rate
    for i, mesh_sample_rate in enumerate(mesh_sample_rates):
        for j, container_sample_rate in enumerate(container_sample_rates):
            degenerate_triangles = get_degenerate_triangles(optimizer, container_sample_rate, mesh_sample_rate)
            degenerate_triangle_counts[i, j] = degenerate_triangles

    # Find the indices of the minimum value in the degenerate_triangle_counts array
    min_indices = np.unravel_index(
        np.argmin(degenerate_triangle_counts, axis=None), degenerate_triangle_counts.shape
    )
    min_mesh_sample_rate = mesh_sample_rates[min_indices[0]]
    min_container_sample_rate = container_sample_rates[min_indices[1]]
    print(
        f"Minimum degenerate triangles found for mesh_sample_rate: {min_mesh_sample_rate}, container_sample_rate: {min_container_sample_rate}"
    )

    # Create a heatmap of the number of degenerate triangles
    sns.set()
    ax = sns.heatmap(
        degenerate_triangle_counts,
        annot=True,
        fmt=".0f",
        xticklabels=container_sample_rates,
        yticklabels=mesh_sample_rates,
    )
    ax.set_xlabel("Container Sample Rate")
    ax.set_ylabel("Mesh Sample Rate")
    plt.show()

if __name__ == "__main__":
    main()