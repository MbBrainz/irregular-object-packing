#%%
import math
import numpy as np

def angle_between(point, line_start, line_end):
    """DEPRECATED"""
    line_vector = (line_end[0]-line_start[0], line_end[1]-line_start[1], line_end[2]-line_start[2])


    point_vector = (point[0]-line_start[0], point[1]-line_start[1], point[2]-line_start[2])
    dot_product = point_vector[0]*line_vector[0] + point_vector[1]*line_vector[1] + point_vector[2]*line_vector[2]
    point_magnitude = math.sqrt(point_vector[0]**2 + point_vector[1]**2 + point_vector[2]**2)
    line_magnitude = math.sqrt(line_vector[0]**2 + line_vector[1]**2 + line_vector[2]**2)
    return math.acos(dot_product / (point_magnitude * line_magnitude))

def sort_points_clockwise(points, start, end):
    # Create normal vector from line start and end points
    start = np.array(start)
    end = np.array(end)
    
    vector = np.array([end[0] - start[0], end[1] - start[1], end[2] - start[2]])
    norm = np.linalg.norm(vector)
    n = vector / norm
    
    print(points)
    
    p = points[0] - points[0].dot(n) * n # take the first point to compute the first orthogonal vector
    q = np.cross(n, p)
    
    angles = []
    for point in points:
        t = np.dot(n, np.cross((point - start), p))
        u = np.dot(n, np.cross((point - start), q))
        angles.append(math.atan2(u, t))
        
    sorted_points = [x for _, x in sorted(zip(angles, points), key=lambda pair: pair[0])]
    return sorted_points

def test_sort_points_clockwise():
    points = np.array([
        [-1,-1,-16], # 225º
        [-1, 2, 23], # 135º
        [ 1, 1, 1],# 45º 
        [ 1,-1, 2],# 315º
    ] )
    
    # represents the normal vector of the x-y plane
    start = [0, 0, 0]
    end = [0, 0, 1]
    
    expected_points= [
        ( 1, 1, 1), # 45º 
        (-1, 2, 23), # 135º
        (-1,-1,-16), # 225º
        ( 1,-1, 2), # 315º
    ]
    sorted_points = sort_points_clockwise(points, start, end)
    print(sorted_points)
    print('-------')
    print(expected_points)

# test_sort_points_clockwise()

def sort_face_points_by_length(expected_faces):
    sorted_faces = []
    for face in expected_faces:
        sorted_faces.append(
            sorted(face, key=lambda point: point[0]**2 + point[1]**2 + point[2]**2))
    sorted_faces_by_hash = {}
    return sorted_faces

def sort_faces_dict(faces):
    for k, v in faces.items():
        faces[k] = sort_face_points_by_length(faces[k])
        
    return faces
# %%

