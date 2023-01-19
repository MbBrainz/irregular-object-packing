
from taichi import MeshInstance as TaiMeshInstance
import gmsh
import pymesh



def taichi_to_pymesh(mesh: TaiMeshInstance):
    """
    Convert a taichi mesh to an open3d mesh
    """
    verts = mesh.get_position_as_numpy()
    
    mesh = pymesh.from_mesh(vertices=verts, faces=mesh.faces)


