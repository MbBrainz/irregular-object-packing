import pyvista as pv


def get_pv_container(container_name):
    """Returns the container from a string"""
    match container_name:
        case "cube":
            return pv.Cube()
        case "sphere":
            return pv.Sphere()
        case "cylinder":
            return pv.Cylinder().clean()
        case "tetrahedron":
            return pv.Tetrahedron()
        case "cone":
            return pv.Cone()
        case _:
            raise ValueError(f"Container {container_name} not found")


def get_pv_shape(shape_name):
    """Returns the shape from a string"""
    match shape_name:
        case "cube":
            return pv.Cube()
        case "sphere":
            return pv.Sphere()
        case "cylinder":
            return pv.Cylinder().clean()
        case "tetrahedron":
            return pv.Tetrahedron()
        case "cone":
            return pv.Cone()
        case "normal_red_blood_cell":
            return pv.read("../../data/mesh/RBC_normal.stl")
        case "sickle_red_blood_cell":
            return pv.read("../../data/mesh/sikleCell.stl")
        case _ :
            raise ValueError(f"Shape {shape_name} not found")
