import argparse
import dgl
import json
import numpy as np
from numpy.linalg import norm
import os
import pickle
import torch
from tqdm import tqdm
from yacs.config import CfgNode as CN


class GraphMaker:
    def __init__(self, thresh=3.0) -> None:
        super().__init__()
        self.graph = dgl.DGLGraph()
        self.thresh = thresh

    def _check_neighbours(self, curr_pose, db_poses, thresh):
        is_neighbour = 0
        # for i in range(num_dbs):
        dist = norm(curr_pose - db_poses)
        if (dist < thresh):
            is_neighbour = 1
            # break
        return is_neighbour

    def _make_adjacency(self, query_id, target_id, poses):
        distance = norm(poses[query_id] - poses[target_id])
        if distance < self.thresh:
            self.adjacency_matrix[query_id][target_id] = 1
            self.adjacency_matrix[target_id][query_id] = 1

    def build_graph(self, features, poses):
        edge_list = []
        num_nodes = len(features)
        self.graph.add_nodes(num_nodes)
        self.adjacency_matrix = torch.zeros(num_nodes, num_nodes)
        for i in tqdm(range(num_nodes)):
            self.graph.nodes[[i]].data['feature'] = torch.FloatTensor(
                features[i]
            )
            self.graph.nodes[[i]].data['pose'] = torch.FloatTensor(
                np.array([poses[i]])
            )
            for j in range(i, num_nodes):
                if self._check_neighbours(poses[i], poses[j], 3.0):
                    edge = (i, j)
                    edge_list.append(edge)
                    self._make_adjacency(i, j, poses)

        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        self.graph.add_edges(src, dst)

def default_config():
    '''
    The one-stop reference point for all configurable options

    **overrides the options should use YAML configuration files**
    '''
    cfg = CN()

    cfg.DIR = CN()
    # feature data directory
    cfg.DIR.FEATURES = "../data/kitti/logg3d_descriptor.pickle"
    # pose data directory
    cfg.DIR.POSES = "/mnt/kitti/semantic_kitti/semantic_kitti/dataset/sequences/08/poses.txt"
    # data saving directory
    cfg.DIR.SAVING = "../data/model_train/"
    
    cfg.PARAM = CN()
    # 
    cfg.PARAM.NODES_NUM = 10
    cfg.PARAM.WINDOW_SHIFT_STEP = 1
    cfg.PARAM.THRESH = 3.0
    return cfg

def load_pickle(pickle_path):
    assert os.path.exists(
        pickle_path), 'Cannot access pickle file: {}'.format(pickle_path)
    print('Loading pickle file: {}...'.format(pickle_path))
    with open(pickle_path, 'rb') as handle:
        descriptors = pickle.load(handle)

    return descriptors


def transfrom_cam2velo(Tcam):
    R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
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


def segmentation(graph, number_nodes, window_shift_step):
    '''
    Segment the feature and poses sequence into subgraph

    Input:
        graph: graph
            A graph embeds the features and position information
            of the lidar signature
        number_nodes: int
            the number of nodes in the subgraph
        window_shift_step: int
            The number of steps to shift the window 
            to generate the next subgraph
    Output:
        segmented_data: dictionary
            'features': [number_subgraph, number_nodes, feature_dim]
            'poses': [number_subgraph, number_nodes, pose_dim]
            'subgraph_nodes': [number_subgraph, number_nodes]
    '''
    features = []
    poses = []
    subgraph_node = []
    print('Start segmenting...')
    assert graph.number_of_nodes() > number_nodes
    print('Nodes number of graph must be larger '
          'than that of subgraph. Nodes '
          'number in the total graph is %i and the nodes number set '
          'for subgraphs is %i' % (
              graph.number_of_nodes(), number_nodes))
    for node in tqdm(
        range(0, graph.number_of_nodes() - number_nodes, window_shift_step)
    ):
        nodes_batch = []
        features_batch = []
        poses_batch = []
        for batch in range(number_nodes):
            node_id = node + batch
            nodes_batch.append(node_id)
            features_batch.append(graph.nodes[[node_id]].data['feature'])
            poses_batch.append(graph.nodes[[node_id]].data['pose']
                               - graph.nodes[[node]].data['pose'])
        features.append(torch.cat(features_batch))
        poses.append(torch.cat(poses_batch))
        subgraph_node.append(torch.tensor(nodes_batch))
    features = torch.stack(features)
    poses = torch.stack(poses)
    subgraph_node = torch.stack(subgraph_node)
    return {'features': features, 'poses': poses, 'subgraph_nodes': subgraph_node}


def pairing(segmented_data, adjacency_matrix, number_nodes, thresh):
    '''
    Pair the segmented data

    Input: 
        segmented_data: dictionary
            'features': [number_subgraph, number_nodes, features_dim]
            'poses': [number_subgraph, number_nodes, poses_dim]
            'start_node': [number_subgraph]
        adjacency_matrix: 2d array, [number_subgraph, number_poses]
        number_nodes: int
            the number of nodes in the subgraph
        thresh: float
            The distance to determine whether 
            the two place are same
    Output:
        overlap_dictionary: dictionary
            the keys in the dictionary are the index of the segmented
            subgraphs, and the values are the lists of the index of the
            subgraphs which overlap with the subgraph in the key.
    '''
    num_pieces = len(segmented_data['features'])
    print('Start pairing...')
    overlap_dictionary = {}
    for piece_first in tqdm(range(num_pieces)):
        overlap_piece = []
        for piece_second in range(num_pieces):
            if piece_first == piece_second: continue
            node_first = segmented_data['subgraph_nodes'][piece_first][0]
            node_second = segmented_data['subgraph_nodes'][piece_second][0]
            sub_adjacency = adjacency_matrix[
                node_first : node_first + number_nodes,
                node_second : node_second + number_nodes
            ]
            matrix_sum = torch.sum(sub_adjacency)
            if matrix_sum > 0:
                overlap_piece.append(piece_second)
        overlap_dictionary[piece_first] = overlap_piece
    return overlap_dictionary
    
def generate_dataset(features, poses, number_nodes, window_shift_step, thresh):
    '''
    Generate dataset

    Input:
        features: [number_poses, 1, features_dim]
        poses: [number_poses, poses_dim]
        number_nodes: int
            the number of nodes in the subgraphs
        window_shift_step: int
            The number of steps to shift the window 
            to generate the next subgraph
        thresh: float
            The distance to determine whether 
            the two place are same
    Output:
        segmented_data: dictionary
            'features': [number_subgraph, number_nodes, feature_dim]
            'poses': [number_subgraph, number_nodes, pose_dim]
            'start_node': [number_subgraph]
        paired_data: dictionary
            the keys in the dictionary are the index of the segmented
            subgraphs, and the values are the lists of the index of the
            subgraphs which overlap with the subgraph in the key.
        adjacency_matrix: [num_nodes, num_nodes]
            The adjacency matrix represent the distance between two nodes
    '''
    rep_graph = GraphMaker(thresh=thresh)
    rep_graph.build_graph(features, poses)

    # Segment the long trail sequence into sub-graph
    segmented_data = segmentation(
        rep_graph.graph,
        number_nodes=number_nodes,
        window_shift_step=window_shift_step,
    )
    paired_data = pairing(
        segmented_data,
        rep_graph.adjacency_matrix,
        number_nodes,
        thresh,
    )

    return segmented_data, paired_data, rep_graph.adjacency_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate dataset in graph structure'
    )
    parser.add_argument(
        "--config_file", 
        default="configs/data_generator_config.yml", 
        metavar="FILE", 
        help="path to config file", 
        type=str
    )
    parser.add_argument(
        '--opts',
        default=None,
        help='Modify config options using the command-line',
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg = default_config()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print("------------------------------ Arguments -----------------------------")
    print(cfg.dump())
    print("----------------------------------------------------------------------")

    # Read the feature and position information for each scan
    features = load_pickle(cfg.DIR.FEATURES)
    _, poses = load_poses_from_txt(cfg.DIR.POSES)

    segmented_data, pairded_data, adjacency_matrix = generate_dataset(
        features, 
        poses, 
        cfg.PARAM.NODES_NUM,
        cfg.PARAM.WINDOW_SHIFT_STEP,
        cfg.PARAM.THRESH,
    )


    torch.save(segmented_data['features'], 
        cfg.DIR.SAVING + "features.pt")
    torch.save(segmented_data['poses'], 
        cfg.DIR.SAVING + "poses.pt")
    torch.save(segmented_data['subgraph_nodes'], 
        cfg.DIR.SAVING + "subgraph_nodes.pt")
    torch.save(adjacency_matrix,
        cfg.DIR.SAVING + "adjacency_matrix.pt")
    json_file = open(cfg.DIR.SAVING + 'paired.json', 'w')
    json.dump(pairded_data, json_file)
    json_file.close()

    print("----------------------------Data Generation---------------------------")
    print("Features, nodes and adjacency matrix have successfully been generated.")
    print("----------------------------------------------------------------------")
