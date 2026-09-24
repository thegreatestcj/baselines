#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import trimesh as tm
from errno import EEXIST
from os import makedirs, path
import matplotlib.pyplot as plt




def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)


def check_gs_model(model_path, saving_iterations, fix_pcd=False):
    file_name = "point_cloud_fix_pcd" if fix_pcd else "point_cloud"
    p = os.path.join(model_path, file_name)
    if not os.path.exists(model_path):
        return False
    if not os.path.exists(p):
        return False
    if len(os.listdir(p)) <= 0:
        return False
    if not searchForMaxIteration(p) == saving_iterations[-1]:
        return False
    else:
        max_iter = searchForMaxIteration(p)
        print(f'Find the model: {p}, iterations: {max_iter}')
        return True

def draw_curve(curve, path, name='loss'):
    x = range(1, len(curve) + 1)
    plt.figure()
    plt.plot(x, curve, label=name)

    plt.xlabel('Training Steps')
    plt.ylabel(name)
    plt.title('{} curve'.format(name))

    plt.legend()

    plt.savefig('{}/{}.png'.format(path, name))

def write_particles(particles, idx, path, name='', vertex_colors=None, object_labels=None):

    if type(particles) == np.ndarray:
        numpy_array = particles
    else:
        numpy_array = particles.cpu().detach().numpy()
    if not os.path.exists(os.path.join(path, 'mpm')):
        mkdir_p(os.path.join(path, 'mpm'))

    # Use object labels as colors if provided
    if object_labels is not None:
        if hasattr(object_labels, 'cpu'):
            object_labels = object_labels.cpu().numpy()
        # Store object_id in all RGB channels
        vertex_colors = np.zeros((len(object_labels), 3), dtype=np.uint8)
        vertex_colors[:, 0] = object_labels.astype(np.uint8)  # R = object_id
        vertex_colors[:, 1] = object_labels.astype(np.uint8)  # G = object_id
        vertex_colors[:, 2] = object_labels.astype(np.uint8)  # B = object_id

    # Create trimesh object with colors
    mesh = tm.Trimesh(numpy_array, vertex_colors=vertex_colors)

    # Export to PLY
    mesh.export(os.path.join(path, f'mpm/{name}_{idx}.ply'))


def read_particles_with_labels(ply_path):
    """
    Read particles from PLY file including object labels if present
    Returns: positions (numpy array), object_labels (numpy array or None)
    """
    import trimesh
    mesh = trimesh.load(ply_path)
    positions = np.array(mesh.vertices)

    # Check if object_id attribute exists
    object_labels = None
    if hasattr(mesh, 'vertex_attributes') and 'object_id' in mesh.vertex_attributes:
        object_labels = mesh.vertex_attributes['object_id']

    return positions, object_labels


def write_ply_with_labels(filepath, positions, colors=None, object_labels=None):
    """
    Write PLY using plyfile to properly save object_id property
    """
    from plyfile import PlyData, PlyElement

    if hasattr(positions, 'cpu'):
        positions = positions.cpu().numpy()

    num_verts = len(positions)

    # Build vertex data with object_id
    if object_labels is not None:
        if hasattr(object_labels, 'cpu'):
            object_labels = object_labels.cpu().numpy()

        if colors is not None:
            if hasattr(colors, 'cpu'):
                colors = colors.cpu().numpy()

            vertex_data = np.zeros(num_verts,
                                  dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                        ('object_id', 'i4')])
            vertex_data['x'] = positions[:, 0]
            vertex_data['y'] = positions[:, 1]
            vertex_data['z'] = positions[:, 2]
            vertex_data['red'] = colors[:, 0]
            vertex_data['green'] = colors[:, 1]
            vertex_data['blue'] = colors[:, 2]
            vertex_data['object_id'] = object_labels
        else:
            vertex_data = np.zeros(num_verts,
                                  dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                        ('object_id', 'i4')])
            vertex_data['x'] = positions[:, 0]
            vertex_data['y'] = positions[:, 1]
            vertex_data['z'] = positions[:, 2]
            vertex_data['object_id'] = object_labels
    else:
        vertex_data = np.zeros(num_verts,
                              dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_data['x'] = positions[:, 0]
        vertex_data['y'] = positions[:, 1]
        vertex_data['z'] = positions[:, 2]

    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(filepath)


def read_ply_with_labels(filepath):
    """
    Read PLY using plyfile to properly load object_id property
    """
    from plyfile import PlyData

    ply_data = PlyData.read(filepath)
    vertex = ply_data['vertex']

    # Get positions
    positions = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

    # Get object_id if exists
    object_labels = None
    # Check if object_id property exists in the vertex data
    if 'object_id' in vertex.data.dtype.names:
        object_labels = vertex.data['object_id'].astype(np.int32)

    return positions, object_labels