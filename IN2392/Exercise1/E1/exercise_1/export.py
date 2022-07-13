"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    # TODO: Implement
    # raise NotImplementedError
    # ###############
    the_file = open(path, 'w')
    for vertices, edge in zip(vertices, faces):
        for vertex in vertices:
            the_file.write("v {0} {1} {2}\n".format(vertex[0], vertex[1], vertex[2]))
        the_file.write("f {0} {1} {2}\n".format(edge[0] + 1, edge[1] + 1, edge[2] + 1))
    the_file.close()


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    # raise NotImplementedError
    # ###############
    the_file = open(path, 'w')
    for point in pointcloud:
        the_file.write("v {0} {1} {2}\n".format(point[0], point[1], point[2]))
    the_file.close()
