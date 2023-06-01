import os
import pickle
import numpy as np
import math
import torch
import open3d as o3d
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#from configs.config import get_config

#cfg = get_config()

def load_pickle(pickle_path):
    assert os.path.exists(pickle_path), 'Cannot access pickle file: {}'.format(pickle_path)
    print('Loading pickle file: {}...'.format(pickle_path))
    with open(pickle_path, 'rb') as handle:
        descriptors = pickle.load(handle)

    return descriptors

def load_pc(filename):
    # Load point cloud, does not apply any transform
    # Returns Nx3 matrix
    file_path = os.path.join(filename)
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1,4)[:,:3]
    # coords are within -1..1 range in each dimension
    # assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(file_path)
    l = 25
    ind = np.argwhere(pc[:, 0] <= l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 0] >= -l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 1] <= l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 1] >= -l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 2] <= l).reshape(-1)
    pc = pc[ind]
    ind = np.argwhere(pc[:, 2] >= -l).reshape(-1)
    pc = pc[ind]

    if cfg.remove_ground_plane:
        not_ground_mask = np.ones(len(pc), np.bool)
        raw_pc = make_open3d_point_cloud(pc[:,:3], color=None)
        _, inliers = raw_pc.segment_plane(0.2, 3, 250)
        not_ground_mask[inliers] = 0
        pc = pc[not_ground_mask]
    # sample to 4096
    # if pc.shape[0] >= 4096:
    #     ind = np.random.choice(pc.shape[0], 4096, replace=False)
    #     pc = pc[ind, :]
    # else:
    #     ind = np.random.choice(pc.shape[0], 4096, replace=True)
    #     pc = pc[ind, :]

    mean = np.mean(pc, axis=0)
    pc = pc - mean
    scale = np.max(abs(pc))
    pc = pc/scale
    # pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    pc = torch.tensor(pc, dtype=torch.float)
    return pc

def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

# Load poses
def transfrom_cam2velo(Tcam):
    R = np.array([ 7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    return Tcam @ cam2velo

def load_poses_from_txt(file_name):
    """
    Modified function from: https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    transforms = {}
    positions = []
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        transforms[frame_idx] = transfrom_cam2velo(P)
        positions.append([P[0, 3], P[2, 3], P[1, 3]])
    return transforms, np.asarray(positions)

# Load timestamps
def load_timestamps(file_name):
    # file_name = data_dir + '/times.txt'
    file1 = open(file_name, 'r+')
    stimes_list = file1.readlines()
    s_exp_list = np.asarray([float(t[-4:-1]) for t in stimes_list])
    times_list = np.asarray([float(t[:-2]) for t in stimes_list])
    times_listn = [times_list[t] * (10**(s_exp_list[t]))
                   for t in range(len(times_list))]
    file1.close()
    return times_listn