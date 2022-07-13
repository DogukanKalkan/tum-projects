"""SDF to Occupancy Grid"""
import numpy as np


def occupancy_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An occupancy grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with value 0 outside the shape and 1 inside.
    """

    # ###############
    # TODO: Implement
    # raise NotImplementedError
    # ###############
    x = np.linspace(-1/2, 1/2, resolution)
    y = np.linspace(-1 / 2, 1 / 2, resolution)
    z = np.linspace(-1 / 2, 1 / 2, resolution)
    gridx, gridy, gridz = np.meshgrid(x,y,z,sparse=False,indexing='ij')
    gridx = gridx.flatten()
    gridy = gridy.flatten()
    gridz = gridz.flatten()
    grid = sdf_function(gridx,gridy,gridz)
    grid = grid.reshape((resolution,resolution,resolution))
    grid = np.where(grid < 0, 1, 0)
    print(f"Shape of Occupancy Grid: {grid.shape}")
    assert grid.shape == (resolution,resolution,resolution)
    return grid
