from itertools import combinations

import numpy as np
from pyvista import PolyData


def compute_collision(mesh: PolyData, with_mesh: PolyData, set_contacts) -> int:
    contact_mesh, n_contacts = mesh.collision(with_mesh, 0, cell_tolerance=1e-6)
    if n_contacts > 0:

        if set_contacts:
            mask = np.zeros(mesh.n_cells, dtype=bool)
            mask[contact_mesh["ContactCells"]] = True
            try :
                mesh["collisions"] = np.logical_or(mesh["collisions"], mask)
            except KeyError:
                mesh["collisions"] = mask
        return n_contacts
    return None


def compute_object_collisions(p_meshes: list[PolyData], set_contacts=False):
    colls = []

    for (_i1, mesh), (_i2, mesh_2) in combinations(enumerate(p_meshes), 2):
        n_contacts = compute_collision(mesh, mesh_2, set_contacts)
        if n_contacts is not None:
            colls.append([(_i1, _i2) , n_contacts])

    return colls


def compute_container_violations(p_meshes, container, set_contacts=False):
    violations = []

    for i, mesh in enumerate(p_meshes):
        n_contacts = compute_collision(mesh, container, set_contacts)
        if n_contacts is not None:
            violations.append([i, n_contacts])

    return violations


def compute_cat_violations(p_meshes, cat_meshes, set_contacts=False):
    violations = []
    for i, (mesh, cat_mesh) in enumerate(zip(p_meshes, cat_meshes, strict=True)):
        n_contacts = compute_collision(mesh, cat_mesh, set_contacts)
        if n_contacts is not None:
            violations.append([i, n_contacts])

    return violations


def compute_and_add_all_collisions(p_meshes, cat_meshes, container, set_contacts=False):
    cat_viols = compute_cat_violations(p_meshes, cat_meshes, set_contacts)
    con_viols = compute_container_violations(p_meshes, container, set_contacts)
    collisions = compute_object_collisions(p_meshes, set_contacts)
    return cat_viols, con_viols, collisions

def compute_outside_points(enclosing_mesh: PolyData, inside_mesh:PolyData) -> PolyData:
    enclosed = enclosing_mesh.select_enclosed_points(inside_mesh, tolerance=1e-8, inside_out=True)
    pts = inside_mesh.extract_points(enclosed["SelectedPoints"].view(bool), adjacent_cells=False)
    return pts
