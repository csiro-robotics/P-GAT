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

import torch.nn.functional as F

class GraphMaker:
    def __init__(self, total_num_nodes) -> None:
        super().__init__()
        self.graph = dgl.DGLGraph()
        self.adjacency_matrix = torch.zeros(total_num_nodes, total_num_nodes)
        self.weight_matrix = torch.zeros(total_num_nodes, total_num_nodes)
        self.map = {}

    def _check_neighbours(self, curr_pose, db_poses, threshold):
        is_neighbour = 0
        # for i in range(num_dbs):
        dist = norm(curr_pose - db_poses)
        if (dist < threshold):
            is_neighbour = 1
            # break
        return is_neighbour

    def _update_matrix(self, updating_matrix, query_id, target_id):
        updating_matrix[query_id][target_id] = 1
        updating_matrix[target_id][query_id] = 1

    def build_graph(
        self, 
        features, 
        poses, 
        threshold, 
        min_interval, 
        neg_threshold=None
    ):
        edge_list = []
        num_nodes = len(features)
        if num_nodes in self.map:
            node_idx = self.map[num_nodes] + 1
            visited_num = num_nodes
        else:
            node_idx = 0
            visited_num = 0
        cum_distance = 0
        for idx_current in tqdm(range(num_nodes)):
            if idx_current != 0:
                distance = norm(poses[idx_current] - poses[idx_current - 1])
                cum_distance += distance
                if cum_distance < min_interval: continue
            cum_distance = 0
            self.graph.add_nodes(1)
            self.graph.nodes[[node_idx]].data['feature'] = torch.FloatTensor(
                features[idx_current]
            )
            self.graph.nodes[[node_idx]].data['pose'] = torch.FloatTensor(
                np.array([poses[idx_current]])
            )
            for idx_previous in range(node_idx + 1):
                if self._check_neighbours(
                    self.graph.nodes[[node_idx]].data['pose'], 
                    self.graph.nodes[[idx_previous]].data['pose'], 
                    threshold):
                    edge = (node_idx, idx_previous)
                    edge_list.append(edge)
                    self._update_matrix(self.adjacency_matrix, node_idx, idx_previous)
                    self._update_matrix(self.weight_matrix, node_idx, idx_previous)
                elif neg_threshold is not None and not self._check_neighbours(
                    self.graph.nodes[[node_idx]].data['pose'], 
                    self.graph.nodes[[idx_previous]].data['pose'], 
                    neg_threshold):
                    self._update_matrix(self.weight_matrix, node_idx, idx_previous)
            self.map[idx_current + visited_num] = node_idx
            node_idx += 1

        self.map[idx_current + visited_num + 1] = node_idx - 1

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
    cfg.DIR.FEATURES_DATABASE = "../data/oxford/dataset_embeddings.pickle"
    # pose data directory
    cfg.DIR.FEATURES_QUERY= "../data/oxford/query_embeddings.pickle"
    # data saving directory
    cfg.DIR.SAVING = "../data/model_train/oxford/"
    
    cfg.PARAM = CN()
    # 
    cfg.PARAM.MIN_INTERVAL = 5.0
    cfg.PARAM.DISTANCE_THRESHOLD = 50.0
    cfg.PARAM.THRESH_DATABASE = 10.0
    cfg.PARAM.NEG_THRESH_DATABASE = 50.0
    cfg.PARAM.THRESH_QUERY = 25.0
    cfg.PARAM.POSE_NORMAL = True
    cfg.PARAM.DESCRIPTOR_NORMAL = True
    return cfg

def load_pickle(pickle_path):
    assert os.path.exists(
        pickle_path), 'Cannot access pickle file: {}'.format(pickle_path)
    print('Loading pickle file: {}...'.format(pickle_path))
    with open(pickle_path, 'rb') as handle:
        descriptors = pickle.load(handle)

    return descriptors

def merge_runs(data, start_idx=0):
    '''
    Merge the datasets from multiple runs

    Input:
        data: list
    Output:
        merged_data: dictionary
            'features': [total_number_poses, 1, features_dim]
            'poses': [total_number_poses, poses_dim]
        idx_per_runs: dictionary
            key: ids of the runs
            values: the start and end nodes' unique ids in each run
    '''
    features = []
    poses = []
    idx_per_runs = {}
    for run_idx in range(len(data)):
        features.append(data[run_idx][0])
        poses.append(data[run_idx][1])
        number_nodes = data[run_idx][0].shape[0]
        idx_per_runs[run_idx] = [start_idx, start_idx + number_nodes - 1]
        start_idx = start_idx + number_nodes
    merged_data = {'features':np.concatenate(features),
        'poses': np.concatenate(poses)}
    return merged_data, idx_per_runs

def segmentation(
        graph, 
        idx_per_runs, 
        distance_threshold, 
        pose_normalize, 
        descriptor_normalize,
        map
    ):
    '''
    Segment the feature and poses sequence into subgraph

    Input:
        graph: graph
            A graph embeds the features and position information
            of the lidar signature
        idx_per_runs: dictionary
            key: ids of the runs
            values: the start and end nodes' unique ids in each run
        distance_threshold: float
            The travel distance threshold for each sub-graph
        pose_normalize:
            If true, the pose will be normalized within sub-graph
        descriptor_normalize:
            If true, the descriptor will be normalized to a unit vector
        map: hash table
            link the original and new id of nodes
    Output:
        segmented_data: dictionary
            'features': float
                [number_subgraph, number_nodes, feature_dim]
            'poses': float
                [number_subgraph, number_nodes, pose_dim]
            'masks': boolean 
                [number_subgraph, number_nodes]
            'subgraph_nodes': 
                list of node id of each sub_graph
            'subgraph_run_id': int
                [number_subgraph]
    '''
    features = []
    poses = []
    masks = []
    subgraph_node = []
    subgraph_run_id = []
    max_length = 0
    
    print('Start segmenting...')
    # create list of node list of subgraph
    for run_idx in idx_per_runs:
        node_range = idx_per_runs[run_idx]
        node_range[0] = map[node_range[0]]
        node_range[1] = map[node_range[1]+1]
        sub_graph = [node_range[0]]
        distance_interval = []
        total_distance = 0
        for node in range(node_range[0] + 1, node_range[1] + 1):
            pre_node_pos = graph.nodes[[node - 1]].data['pose']
            cur_node_pos = graph.nodes[[node]].data['pose']
            distance = torch.sqrt(torch.sum((pre_node_pos - cur_node_pos)**2))
            sub_graph.append(node)
            total_distance += distance
            distance_interval.append(distance)
            if total_distance >= distance_threshold:
                subgraph_node.append([sub_graph[0], len(sub_graph) - 1])
                subgraph_run_id.append(run_idx)
                max_length = max(max_length, len(sub_graph) - 1)
                while total_distance > distance_threshold:
                    sub_graph.pop(0)
                    total_distance -= distance_interval[0]
                    distance_interval.pop(0)

    features_dim = graph.nodes[[0]].data['feature'].shape[-1]
    poses_dim = graph.nodes[[0]].data['pose'].shape[-1]
    for sub_graph in subgraph_node:
        sub_features = torch.zeros(max_length, features_dim)
        sub_poses = torch.zeros(max_length, poses_dim)
        sub_masks = torch.zeros(max_length)
        node_iter = range(sub_graph[0], sub_graph[0] + sub_graph[1])
        for relative_idx, node_idx in enumerate(node_iter):
            # features
            sub_features[relative_idx, :] = graph.nodes[[node_idx]].data['feature']
            # poses
            sub_poses[relative_idx, :] = graph.nodes[[node_idx]].data['pose'] - graph.nodes[[sub_graph[0]]].data['pose']
            # masks
            sub_masks[relative_idx] = 1
        if pose_normalize:
            pose_mean = torch.mean(sub_poses[:sub_graph[1], :])
            pose_std = torch.std(sub_poses[:sub_graph[1], :])
            sub_poses[:sub_graph[1], :] = (sub_poses[:sub_graph[1], :] - pose_mean) / pose_std
        features.append(sub_features)
        poses.append(sub_poses)

        sub_masks = sub_masks == 0
        masks.append(sub_masks)

    features = torch.stack(features)
    if descriptor_normalize:
        features = F.normalize(features, dim=2)
    poses = torch.stack(poses)
    masks = torch.stack(masks)
    subgraph_node = torch.tensor(np.stack(subgraph_node))
    subgraph_run_id = torch.tensor(subgraph_run_id)
    return {'features': features, 
        'poses': poses, 
        'masks': masks,
        'subgraph_nodes': subgraph_node, 
        'subgraph_run_id': subgraph_run_id}


def pairing(segmented_data, adjacency_matrix):
    '''
    Pair the segmented data

    Input: 
        segmented_data: dictionary
            'features': [number_subgraph, number_nodes, feature_dim]
            'poses': [number_subgraph, number_nodes, pose_dim]
            'subgraph_nodes': [number_subgraph, number_nodes]
            'subgraph_run_id': [number_subgraph]
        adjacency_matrix: 2d array, [number_subgraph, number_poses]
    Output:
        overlap_dictionary: dictionary
            the keys in the dictionary are the index of the segmented
            subgraphs, and the values are the lists of the index of the
            subgraphs which overlap with the subgraph in the key.
    '''
    num_subgraph = len(segmented_data['features'])
    num_runs = torch.max(segmented_data['subgraph_run_id'])
    print('Start pairing...')
    overlap_dictionary = {}
    for first_subgraph_id in tqdm(range(num_subgraph)):
        overlap_piece = [[] for _ in range(num_runs + 1)]
        for second_subgraph_id in range(num_subgraph):
            if first_subgraph_id == second_subgraph_id: continue
            nodes_first = segmented_data['subgraph_nodes'][first_subgraph_id]
            nodes_second = segmented_data['subgraph_nodes'][second_subgraph_id]
            sub_adjacency = adjacency_matrix[
                nodes_first[0] : nodes_first[0] + nodes_first[1],
                nodes_second[0] : nodes_second[0] + nodes_second[1]
            ]
            matrix_sum = torch.sum(sub_adjacency)
            if matrix_sum > 0:
                run_id = segmented_data['subgraph_run_id'][second_subgraph_id]
                overlap_piece[run_id].append(second_subgraph_id)
        overlap_dictionary[first_subgraph_id] = overlap_piece
    return overlap_dictionary
    
def generate_dataset(
    rep_graph, 
    idx_per_runs, 
    is_database,
    pose_normalize,
    descriptor_normalize,
):
    '''
    Generate dataset

    Input:
        rep_graph: graph
            The graph includes the feature and pose information of each node
        idx_per_runs: dictionary
            key: ids of the runs
            values: the start and end nodes' unique ids in each run
        number_nodes: int
            the number of nodes in the subgraphs
        is_database: boolean
            define whether the dataset is database
        pose_normalize: boolean
            if True, the pose will be normalized within subgraph
        descriptor_normalize:
            If true, the descriptor will be normalized to a unit vector
    Output:
        segmented_data: dictionary
            'features': [number_subgraph, number_nodes, feature_dim]
            'poses': [number_subgraph, number_nodes, pose_dim]
            'start_node': [number_subgraph]
        paired_data: dictionary
            the keys in the dictionary are the index of the segmented
            subgraphs, and the values are the lists of the index of the
            subgraphs which overlap with the subgraph in the key.
    '''
    # Segment the long trail sequence into sub-graph
    segmented_data = segmentation(
        rep_graph.graph,
        idx_per_runs,
        distance_threshold=cfg.PARAM.DISTANCE_THRESHOLD,
        pose_normalize=pose_normalize,
        descriptor_normalize=descriptor_normalize,
        map=rep_graph.map,
    )
    if is_database:
        paired_data = pairing(
            segmented_data,
            rep_graph.adjacency_matrix,
        )
        return segmented_data, paired_data
    else:
        return segmented_data


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
        'opts',
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
    data_database = load_pickle(cfg.DIR.FEATURES_DATABASE)
    data_query = load_pickle(cfg.DIR.FEATURES_QUERY)
    total_runs = max(len(data_database), len(data_query))

    # Merge the dataset in different runs into a numpy array
    merged_database, idx_per_runs_database = merge_runs(data_database)
    merged_query, idx_per_runs_query = merge_runs(
        data_query, 
        start_idx=merged_database['features'].shape[0], 
    )
    total_num_nodes = merged_database['features'].shape[0] + merged_query['features'].shape[0]
    rep_graph = GraphMaker(total_num_nodes=total_num_nodes)
    print('Creating a graph with database dataset')
    rep_graph.build_graph(
        np.expand_dims(merged_database['features'], 1), 
        merged_database['poses'],
        threshold=cfg.PARAM.THRESH_DATABASE,
        min_interval=cfg.PARAM.MIN_INTERVAL,
        neg_threshold=cfg.PARAM.NEG_THRESH_DATABASE)
    print('Adding nodes into the graph with query dataset')
    rep_graph.build_graph(
        np.expand_dims(merged_query['features'], 1), 
        merged_query['poses'],
        threshold=cfg.PARAM.THRESH_QUERY,
        min_interval=cfg.PARAM.MIN_INTERVAL,
    )
    print('Graph has %d nodes.' % rep_graph.graph.number_of_nodes())
    print('Graph has %d edges.' % rep_graph.graph.number_of_edges())

    segmented_data_database, paired_data_database = generate_dataset(
        rep_graph,
        idx_per_runs_database,
        is_database=True,
        pose_normalize=cfg.PARAM.POSE_NORMAL,
        descriptor_normalize=cfg.PARAM.DESCRIPTOR_NORMAL,
    )
    segmented_data_query = generate_dataset(
        rep_graph,
        idx_per_runs_query,
        is_database=False,
        pose_normalize=cfg.PARAM.POSE_NORMAL,
        descriptor_normalize=cfg.PARAM.DESCRIPTOR_NORMAL,
    )

    # Save representation graph
    dgl.save_graphs(cfg.DIR.SAVING + "rep_graph.bin",
        rep_graph.graph)
    # Save adjacency matrix
    torch.save(rep_graph.adjacency_matrix,
        cfg.DIR.SAVING + "adjacency_matrix.pt")
    # Save switches matrix
    torch.save(rep_graph.weight_matrix,
        cfg.DIR.SAVING + "switches_matrix.pt")
    # Save database material 
    torch.save(segmented_data_database['features'], 
        cfg.DIR.SAVING + "features_database.pt")
    torch.save(segmented_data_database['poses'], 
        cfg.DIR.SAVING + "poses_database.pt")
    torch.save(segmented_data_database['masks'], 
        cfg.DIR.SAVING + "masks_database.pt")
    torch.save(segmented_data_database['subgraph_run_id'], 
        cfg.DIR.SAVING + "subgraph_run_id_database.pt")
    torch.save(segmented_data_database['subgraph_nodes'], 
        cfg.DIR.SAVING + "subgraph_nodes_database.pt")
    with open(cfg.DIR.SAVING + 'paired_database.json', 'w') as paired_file:
        json.dump(paired_data_database, paired_file)
    # Save query material  
    torch.save(segmented_data_query['features'], 
        cfg.DIR.SAVING + "features_query.pt")
    torch.save(segmented_data_query['poses'], 
        cfg.DIR.SAVING + "poses_query.pt")
    torch.save(segmented_data_query['masks'], 
        cfg.DIR.SAVING + "masks_query.pt")
    torch.save(segmented_data_query['subgraph_run_id'], 
        cfg.DIR.SAVING + "subgraph_run_id_query.pt")
    torch.save(segmented_data_query['subgraph_nodes'], 
        cfg.DIR.SAVING + "subgraph_nodes_query.pt")

    print("----------------------------Data Generation---------------------------")
    print("Features, nodes and adjacency matrix have successfully been generated.")
    print("----------------------------------------------------------------------")
