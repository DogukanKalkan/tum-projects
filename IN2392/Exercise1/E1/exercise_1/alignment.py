""" Procrustes Aligment for point clouds """
import numpy as np
from pathlib import Path


def procrustes_align(pc_x, pc_y):
    """
    calculate the rigid transform to go from point cloud pc_x to point cloud pc_y, assuming points are corresponding
    :param pc_x: Nx3 input point cloud
    :param pc_y: Nx3 target point cloud, corresponding to pc_x locations
    :return: rotation (3, 3) and translation (3,) needed to go from pc_x to pc_y
    """
    R = np.zeros((3, 3), dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)

    # TODO: Your implementation starts here ###############
    # 1. get centered pc_x and centered pc_y
    # 2. create X and Y both of shape 3XN by reshaping centered pc_x, centered pc_y
    # 3. estimate rotation
    # 4. estimate translation
    # R and t should now contain the rotation (shape 3x3) and translation (shape 3,)

    mean_x = np.mean(pc_x, axis=0)
    mean_y = np.mean(pc_y, axis=0)
    #print(f"Mean of X: {mean_x}")
    #print(f"Mean of Y: {mean_y}")
    centered_x = pc_x - mean_x
    centered_y = pc_y - mean_y
    assert pc_x.shape == centered_x.shape
    assert pc_y.shape == centered_y.shape
    #print(f"New Mean of X: {np.mean(centered_x, axis=0)}")
    #print(f"New Mean of Y: {np.mean(centered_y,axis=0)}")

    X = centered_x.T # 3xN
    Y = centered_y.T # 3xN
    #print(f"Shape of Translation Vector: {t.shape}")
    xyt = np.matmul(X,Y.T) # 3xN * Nx3
    #print(f"Shape of Matrix Multiplication (To Feed Into SVD): {xyt.shape}")
    u, d, vt = np.linalg.svd(xyt)
    s = np.diag([1,1,np.linalg.det(u)*np.linalg.det(vt.T)])
    #print(f"Shapes of U, S, V^T: {u.shape}, {s.shape}, {vt.shape}")
    R = np.matmul(np.matmul(vt.T, s), u.T) # R = VSU^T
    t = mean_y - np.matmul(R, mean_x)

    # TODO: Your implementation ends here ###############

    t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, pc_x.shape[0]))
    print('Procrustes Aligment Loss: ', np.abs((np.matmul(R, pc_x.T) + t_broadcast) - pc_y.T).mean())

    return R, t


def load_correspondences():
    """
    loads correspondences between meshes from disk
    """

    load_obj_as_np = lambda path: np.array(list(map(lambda x: list(map(float, x.split(' ')[1:4])), path.read_text().splitlines())))
    path_x = (Path(__file__).parent / "resources" / "points_input.obj").absolute()
    path_y = (Path(__file__).parent / "resources" / "points_target.obj").absolute()
    return load_obj_as_np(path_x), load_obj_as_np(path_y)
