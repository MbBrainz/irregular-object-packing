def sort_surfaces(point_surfaces):
    point_surfaces = [tuple(sorted(surface)) for surface in point_surfaces]
    return sorted(point_surfaces)


def sort_points_in_dict(dictionary):
    """
    Sorts the points in a dictionary of the form:
    {
        (x1, y1, z1): [
            [np.array([...]), np.array([...]), ...],
            ...
        ],
        (x2, y2, z2): [
            [np.array([...]), np.array([...]), ...],
            ...
        ],
        ...
    }
    """
    sorted_dict = {}
    for point, surfaces in dictionary.items():
        sorted_point = tuple(sorted(point))
        surfaces = [sort_surfaces(surface) for surface in surfaces]
        sorted_surfaces = sort_surfaces(surfaces)
        sorted_dict[sorted_point] = sorted_surfaces
    return dict(sorted(sorted_dict.items()))
