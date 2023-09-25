import numpy as np
from numba import njit, prange
from typing import Optional, Tuple



'''

Angle Functions

'''

def dihedrals(a: np.array, b: np.array, c: np.array, d: np.array) -> np.float64:
    """
    Compute the dihedral angle between four points (a, b, c, d) in space.
    
    Parameters:
        a (np.array): The coordinates of point a.
        b (np.array): The coordinates of point b.
        c (np.array): The coordinates of point c.
        d (np.array): The coordinates of point d.
    
    Returns:
        np.float64: The dihedral angle in radians.
    """
    # Calculate the vectors connecting the points
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c
    
    # Normalize b1
    b1 /= np.linalg.norm(b1)
    
    # Calculate the cross and dot products
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    # Calculate the dihedral angle
    angle = np.arctan2(y, x)
    
    return angle

def unit_vector(vector: np.array) -> np.array:
    """
    Compute the unit vector of a given vector.
    
    Parameters:
        vector (np.array): The input vector.
    
    Returns:
        np.array: The unit vector of the input vector.
    """
    # Compute the norm of the vector
    norm = np.linalg.norm(vector)
    
    # Compute the unit vector
    unit_vector = vector / norm
    
    return unit_vector

def vector_angle(v1: np.array, v2: np.array) -> np.float64:
    """
    Compute the angle in radians between two vectors.
    
    Parameters:
        v1 (np.array): The first vector.
        v2 (np.array): The second vector.
    
    Returns:
        np.float64: The angle in radians between the two vectors.
    """
    # Compute the unit vectors of v1 and v2
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    
    # Calculate the dot product of the unit vectors
    dot_product = np.dot(v1_u, v2_u)
    
    # Ensure the dot product is within the valid range
    if dot_product < -1:
        dot_product = -1
    if dot_product > 1:
        dot_product = 1
    
    # Calculate the angle using the arccosine function
    angle = np.arccos(dot_product)
    
    return angle

def pairwise_angle_matrix(pts: np.array, axis: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the pairwise angle matrix between points in the given array.
    
    Parameters:
        pts (np.array): The array of points.
        axis (Optional[np.ndarray]): The optional array specifying the axis of points.
    
    Returns:
        np.ndarray: The pairwise angle matrix.
    """
    # If axis is not provided, use all points as axis
    if axis is None:
        axis = np.arange(0, pts.shape[0])
    
    n_pts = pts.shape[0]
    n_axis = axis.shape[0]
    
    # Initialize the pairwise angle matrix
    a = np.zeros(shape=(n_pts, n_pts))
    
    # Compute the pairwise angles
    for idx_axis in prange(0, n_axis):
        i = axis[idx_axis]
        for j in np.arange(0, n_pts):
            # Compute the angle from i to j and from j to i
            angle_ij = vector_angle(pts[i], pts[j])
            
            # Assign the angles to the matrix
            a[i, j] = angle_ij
            a[j, i] = angle_ij
    
    return a

def point_array_angle(pt: np.array, arr: np.ndarray) -> np.array:
    """
    Compute the angles between a point and an array of vectors.
    
    Parameters:
        pt (np.array): The point.
        arr (np.ndarray): The array of vectors.
    
    Returns:
        np.array: The array of angles between the point and vectors.
    """
    # Ensure the dimensions of pt and arr are compatible
    assert pt.shape[0] == arr.shape[1]
    
    # Initialize the array of angles
    D = np.zeros(shape=arr.shape[0])
    
    # Compute the angles
    for i in np.arange(0, arr.shape[0]):
        D[i] = vector_angle(pt, arr[i, :])
    
    return D

def perpendicular_vector(vector: np.array) -> np.array:
    """
    Compute a vector perpendicular to the given vector.
    
    Parameters:
        vector (np.array): The input vector.
    
    Returns:
        np.array: A vector perpendicular to the input vector.
    """
    # Implementation logic here
    return

def angle_plane_vector(p1: np.array, p2: np.array, p3: np.array, v: np.array) -> np.float64:
    """
    Compute the angle between a plane and a vector.
    
    Parameters:
        p1 (np.array): The first point on the plane.
        p2 (np.array): The second point on the plane.
        p3 (np.array): The third point on the plane.
        v (np.array): The vector.
    
    Returns:
        np.float64: The angle between the plane and the vector.
    """
    # Implementation logic here
    return

def angle_plane_plane(p1: np.array, p2: np.array, p3: np.array, q1: np.array, q2: np.array, q3: np.array) -> np.float64:
    """
    Compute the angle between two planes.
    
    Parameters:
        p1 (np.array): The first point on the first plane.
        p2 (np.array): The second point on the first plane.
        p3 (np.array): The third point on the first plane.
        q1 (np.array): The first point on the second plane.
        q2 (np.array): The second point on the second plane.
        q3 (np.array): The third point on the second plane.
    
    Returns:
        np.float64: The angle between the two planes.
    """
    # Implementation logic here
    return

def project_vector_onto_plane(v: np.array, n: np.array) -> np.array:
    """
    Compute the projection of a vector onto a plane.
    
    Parameters:
        v (np.array): The vector to be projected.
        n (np.array): The normal vector of the plane.
    
    Returns:
        np.array: The projected vector onto the plane.
    """
    # Implementation logic here
    return


'''

Distance Functions

'''


def p_norm(p1: np.array, p2: np.array, p: Optional[float] = 2):
    """
    Compute the p-norm between two points.
    
    Parameters:
        p1 (np.array): The first point.
        p2 (np.array): The second point.
        p (Optional[float]): The norm parameter (default: 2).
    
    Returns:
        np.float64: The p-norm between the two points.
    """
    # Ensure that the shapes of the points are aligned
    assert p1.shape == p2.shape, 'requires aligned shape'
    
    # Compute the p-norm
    return np.power(np.sum(np.power(p1 - p2, p)), 1/p)

def rmsd(coords1: np.ndarray, coords2: np.ndarray):
    """
    Compute the Root-Mean-Square Deviation (RMSD) between two sets of coordinates.
    
    Parameters:
        coords1 (np.ndarray): The first set of coordinates.
        coords2 (np.ndarray): The second set of coordinates.
    
    Returns:
        np.float64: The RMSD between the two sets of coordinates.
    """
    # Ensure that the shapes of the coordinate arrays are aligned
    assert coords1.shape == coords2.shape, 'requires aligned shapes'
    
    # Compute the differences between corresponding coordinates and store them in an array
    diff = np.array([p_norm(coords1[idx], coords2[idx]) for idx in range(coords1.shape[0])])
    
    # Compute the squared RMSD by summing the squares of the differences and dividing by the number of coordinates
    squared_rmsd = np.sum(np.power(diff, 2)) / diff.shape[0]
    
    # Compute the RMSD by taking the square root of the squared RMSD
    return np.power(squared_rmsd, 1/2)

def pairwise_distance_matrix(pts: np.ndarray, axis: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the pairwise distance matrix between points in the given array.
    
    Parameters:
        pts (np.ndarray): The array of points.
        axis (Optional[np.ndarray]): The optional array specifying the axis of points.
    
    Returns:
        np.ndarray: The pairwise distance matrix.
    """
    # If the axis is not provided, use all points as the axis
    if axis is None:
        axis = np.arange(0, pts.shape[0])
    
    # Get the number of points and the number of axis points
    n_pts = pts.shape[0]
    n_axis = axis.shape[0]
    
    # Initialize the matrix to store the pairwise distances
    m = np.zeros(shape=(n_pts, n_pts))
    
    # Compute the pairwise distances between points
    for idx_axis in prange(0, n_axis):
        i = axis[idx_axis]
        for j in np.arange(0, n_pts):
            dist_ij = np.power(np.sum((pts[i] - pts[j])**2), 1/2)
            m[i, j] = dist_ij
            m[j, i] = dist_ij
    
    return m

def point_array_distance(pt: np.array, arr: np.ndarray) -> np.array:
    """
    Compute the distances between a point and an array of vectors.
    
    Parameters:
        pt (np.array): The point.
        arr (np.ndarray): The array of vectors.
    
    Returns:
        np.array: The array of distances between the point and vectors.
    """
    # Ensure that the dimensions of the point and array of vectors are aligned
    assert pt.shape[0] == arr.shape[1]
    
    # Initialize an array to store the distances
    D = np.zeros(shape=arr.shape[0])
    
    # Compute the distances between the point and each vector in the array
    for i in np.arange(0, arr.shape[0]):
        D[i] = p_norm(p1=pt, p2=arr[i,])
    
    return D

def calculate_centroid(points: np.ndarray) -> np.array:
    """
    Calculate the centroid of a set of points.
    
    Parameters:
        points (np.ndarray): The array of points.
    
    Returns:
        np.array: The centroid point.
    """
    centroid = np.sum(points, axis=0) / points.shape[0]
    return centroid

def find_furthest_points(points: np.ndarray) -> Tuple[int, int]:
    """
    Find the pair of points with the maximum distance between them.
    
    Parameters:
        points (np.ndarray): The array of points.
    
    Returns:
        Tuple[int, int]: Indices of the furthest points.
    """
    max_distance = 0
    max_indices = (0, 1)
    
    for i in range(points.shape[0] - 1):
        for j in range(i + 1, points.shape[0]):
            distance = p_norm(points[i], points[j])
            if distance > max_distance:
                max_distance = distance
                max_indices = (i, j)
    
    return max_indices

def find_closest_points(points: np.ndarray) -> Tuple[int, int]:
    """
    Find the pair of points with the minimum distance between them.
    
    Parameters:
        points (np.ndarray): The array of points.
    
    Returns:
        Tuple[int, int]: Indices of the closest points.
    """
    min_distance = np.inf
    min_indices = (0, 1)
    
    for i in range(points.shape[0] - 1):
        for j in range(i + 1, points.shape[0]):
            distance = p_norm(points[i], points[j])
            if distance < min_distance:
                min_distance = distance
                min_indices = (i, j)
    
    return min_indices

def centre_of_mass(p: np.ndarray, w: Optional[np.ndarray] = None) -> np.array:
    """
    Compute the center of mass of a set of points using a weighting scheme.
    
    Parameters:
        points (np.ndarray): The array of points.
        weights (Optional[np.ndarray]): The optional array of weights for each point.
    
    Returns:
        np.array: The center of mass point.
    """
    if w is None:
        return calculate_centroid(p)
    
    assert p.shape[0] == w.shape[0], 'Number of points and weights must match'
    total_weight = np.sum(w)
    centroid = np.sum(p * w[:, np.newaxis], axis=0) / total_weight
    return centroid



if __name__ == '__main__':
    a = np.random.normal(size = 3)
    b = np.random.normal(size = 3)
    c = np.random.normal(size = 3)
    d = np.random.normal(size = 3)
    A = np.array([a,b,c,d])
    print(pairwise_angle_matrix(A))
    print(pairwise_distance_matrix(A))
    

