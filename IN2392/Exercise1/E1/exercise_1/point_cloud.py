"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """

    # ###############
    # TODO: Implement
    # raise NotImplementedError
    # ###############

    areas = np.ones(faces.shape[0]) # Areas of the triangles.
    sampled_points = list()
    for index,vertex in enumerate(vertices):
        #Area by 3d coordinates is given by: 1/2 * |AB X AC| (Cross product of two vectors)
        areas[index] = 0.5 * np.linalg.norm(np.cross(vertex[1] - vertex[0], vertex[2]-vertex[0]))
    probabilities = areas / sum(areas)
    weighted_random_indices = np.random.choice(range(len(areas)), size = n_points, p=probabilities)

    for weighted_random_index in weighted_random_indices:
        vertex_indices = vertices[weighted_random_index]
        r_1 = np.random.rand()
        r_2 = np.random.rand()
        u = 1. - np.sqrt(r_1)
        v = np.sqrt(r_1) * (1. - r_2)
        w = np.sqrt(r_1) * r_2          # Could also be w = 1 - (u + v)
        # p = uA + vB + wC
        sampled_points.append(u*vertex_indices[0] + v*vertex_indices[1]+ w*vertex_indices[2])
    return np.array(sampled_points)
