import numpy as np


def split_4(p: np.ndarray):
    """Create the faces of a tetrahedron with 4 different objects."""
    assert np.shape(p) == (4, 3)
    p0123 = np.sum(p, axis=0) / 4
    p01 = (p[0] + p[1]) / 2
    p02 = (p[0] + p[2]) / 2
    p03 = (p[0] + p[3]) / 2
    p12 = (p[1] + p[2]) / 2
    p23 = (p[2] + p[3]) / 2
    p13 = (p[1] + p[3]) / 2

    p012 = (p[0] + p[1] + p[2]) / 3
    p013 = (p[0] + p[1] + p[3]) / 3
    p023 = (p[0] + p[2] + p[3]) / 3
    p123 = (p[1] + p[2] + p[3]) / 3

    faces_0 = np.array([
        [p0123, p01, p012],
        [p0123, p012, p02],
        [p0123, p02, p023],
        [p0123, p023, p03],
        [p0123, p03, p013],
        [p0123, p013, p01],
    ])

    faces_1 = np.array([
        [p0123, p01, p012],
        [p0123, p012, p12],
        [p0123, p12, p123],
        [p0123, p123, p13],
        [p0123, p13, p013],
        [p0123, p013, p01],
    ])

    faces_2 = np.array([
        [p0123, p02, p012],
        [p0123, p012, p12],
        [p0123, p12, p123],
        [p0123, p123, p23],
        [p0123, p23, p023],
        [p0123, p023, p02],
    ])

    faces_3 = np.array([
        [p0123, p03, p013],
        [p0123, p013, p13],
        [p0123, p13, p123],
        [p0123, p123, p23],
        [p0123, p23, p023],
        [p0123, p023, p03],
    ])

    return faces_0, faces_1, faces_2, faces_3

def split_3(p: np.ndarray):
    """Create the faces of a tetrahedron with 3 different objects."""
    assert np.shape(p) == (4, 3)

    p023 = (p[0] + p[2] + p[3]) / 3
    p123 = (p[1] + p[2] + p[3]) / 3

    p02 = (p[0] + p[2]) / 2
    p03 = (p[0] + p[3]) / 2
    p12 = (p[1] + p[2]) / 2
    p13 = (p[1] + p[3]) / 2
    p23 = (p[2] + p[3]) / 2

    face_01_2 = [p023, p123, p12, p02]
    face_01_3 = [p023, p123, p13, p03]
    face_23 = [p023, p123, p23]

    faces_01 = [face_01_2,face_01_3]

    faces_2 = [face_01_2, face_23]
    faces_3 = [face_01_3, face_23]

    return faces_01, faces_01, faces_2, faces_3


def split_2_3331(p: np.ndarray):
    """Create the faces of a tetrahedron with 3 points from object a and 1 from object b.

    Args:
        - p: the points in the tetrahedron
    """
    assert np.shape(p) == (4, 3), "p must be a 4x3 array"

    p03 = (p[0] + p[3]) / 2
    p13 = (p[1] + p[3]) / 2
    p23 = (p[2] + p[3]) / 2

    face = np.array([p03, p13, p23])
    return ([face,],) * 4


def split_2_2222(p: np.ndarray):
    """Create the faces of a tetrahedron with 2 points from object a and 2 from object b.

    Args:
        - p: the points in the tetrahedron
    """
    assert np.shape(p) == (4, 3), "p must be a 4x3 array"

    p02 = (p[0] + p[2]) / 2
    p03 = (p[0] + p[3]) / 2
    p12 = (p[1] + p[2]) / 2
    p13 = (p[1] + p[3]) / 2

    face = np.array([p02, p03, p13, p12])
    return ([face,],) * 4
