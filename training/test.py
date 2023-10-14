import argparse
import json
import os
from re import sub
from unittest import result
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from attentional_graph.config_init import cfg
from attentional_graph.engine.inference import do_inference
from attentional_graph.modeling.pose_gat import PoseGAT
from datasets import build_dataset, data_normalize

def calculate_recall_per_run(
    sub_similarity_matrix,
    sub_ground_truth,
    num_candidates,
):
    recall = np.zeros(num_candidates)
    # remove the query without matching nodes in ground truth
    mask = (np.sum(sub_ground_truth, axis=1) > 0)
    sub_ground_truth = sub_ground_truth[mask, :]
    sub_similarity_matrix = sub_similarity_matrix[mask, :]
    sorted_index = np.argsort(-1 * sub_similarity_matrix)
    one_percent_threshold = max(int(round(sorted_index.shape[1] / 100)), 1)
    for row_idx in range(sorted_index.shape[0]):
        sorted_ground_truth = sub_ground_truth[row_idx][sorted_index[row_idx]]
        candidates_ground_truth = sorted_ground_truth[:num_candidates]
        candidates_ground_truth = np.cumsum(candidates_ground_truth)
        candidates_ground_truth = np.minimum(
            candidates_ground_truth,
            np.ones_like(candidates_ground_truth)
        )
        recall += candidates_ground_truth
    return recall / sorted_index.shape[0], recall[one_percent_threshold] / sorted_index.shape[0]
    
def concat_database_query(matrix, query_range, database_range=None, query_range=None):
    if database_range is not None and len(query_range) != 0:
        query_matrix = matrix[query_range[0]: query_range[1], query_range[0]:query_range[1]]
        database_matrix = matrix[query_range[0]: query_range[1], database_range[0]:database_range[1]]
        return np.concatenate((database_matrix, query_matrix), axis=1)
    elif len(query_range) != 0:
        query_matrix = matrix[query_range[0]: query_range[1], query_range[0]:query_range[1]]
        return query_matrix
    elif database_range is not None:
        database_matrix = matrix[query_range[0]: query_range[1], database_range[0]:database_range[1]]
        return database_matrix
    else:
        raise RuntimeError('The database_range and query_range should not be empty at same time.')


def evaluate(
    similarity_matrix, 
    ground_truth,
    query_run,
    query_node,
    database_per_run,
    has_database,
):
    num_candidates = 25
    recall = np.zeros(num_candidates)
    recall_one_percent = 0
    count = 0
    for database_run_id in database_per_run:
        for query_run_id in database_per_run:
            if database_run_id == query_run_id: continue
            print('query run id: %i, database run id: %i'%(query_run_id, database_run_id))
            query_subgraph_idx = (query_run == query_run_id).nonzero(as_tuple=True)[0]
            if query_subgraph_idx.shape[0] == 0: continue
            query_node_idx = query_node[query_subgraph_idx]
            query_range = [
                query_node_idx[0, 0], 
                query_node_idx[-1, 0] + query_node_idx[-1, 1]
            ]
            if has_database:
                database_range = database_per_run[database_run_id]['database_nodes_range']
                query_range = database_per_run[database_run_id]['query_nodes_range']
                recall_per_run, recall_one_percent_per_run = calculate_recall_per_run(
                    concat_database_query(
                        similarity_matrix, 
                        query_range, 
                        database_range=database_range, 
                        query_range=query_range
                    ),
                    concat_database_query(
                        ground_truth,
                        query_range, 
                        database_range=database_range, 
                        query_range=query_range
                    ),
                    num_candidates,
                )
                recall += recall_per_run
                recall_one_percent += recall_one_percent_per_run
            else:
                query_range = database_per_run[database_run_id]['query_nodes_range']
                recall_per_run, recall_one_percent_per_run = calculate_recall_per_run(
                    concat_database_query(
                        similarity_matrix, 
                        query_range, 
                        query_range=query_range
                    ),
                    concat_database_query(
                        ground_truth,
                        query_range, 
                        query_range=query_range
                    ),
                    num_candidates,
                )
                recall += recall_per_run
                recall_one_percent += recall_one_percent_per_run
            count += 1
    
    return {
        'recall@k': (recall / count).tolist(), 
        'recall@1%': recall_one_percent / count
    }

def retrieve_info_per_run(dataset, run_idx, data_selector):
    subgraph_idx = (dataset['run_id'][data_selector] == run_idx).nonzero(as_tuple=True)[0]
    features = dataset['features'][data_selector][subgraph_idx]
    poses = dataset['poses'][data_selector][subgraph_idx]
    masks = dataset['masks'][data_selector][subgraph_idx]
    nodes_info = dataset['nodes_info'][data_selector][subgraph_idx]
    if subgraph_idx.shape[0] != 0:
        nodes_range = [
            nodes_info[0, 0], 
            nodes_info[-1, 0] + nodes_info[-1, 1]
        ]
    else:
        nodes_range = []
    return {
        'features': features,
        'poses': poses,
        'masks': masks,
        'nodes_info': nodes_info,
        'nodes_range': nodes_range,
    }

def database_splitting(dataset, has_database):
    '''
    Split the dataset by the run_id

    Input:
        dataset: dictionary
            The dataset produced by build.py
        has_database: boolen
            indicate if the dataset has database set for the evaluation
    Output:
        database_per_run: dictionary
            keys: run_id
            value: dictionary dataset for each run
    '''
    database_per_run = {}
    runs = torch.unique(torch.cat(dataset['run_id']))
    for run_idx in runs:
        run_idx = run_idx.item()
        # select all the sub-graphs with the current run_idx from database dataset
        if has_database:
            database_per_run = retrieve_info_per_run(
                dataset, 
                run_idx, 
                data_selector=0,
            )
            query_per_run = retrieve_info_per_run(
                dataset, 
                run_idx, 
                data_selector=1,
            )
            features = [database_per_run['features'], query_per_run['features']]
            poses = [database_per_run['poses'], query_per_run['poses']]
            masks = [database_per_run['masks'], query_per_run['masks']]
            nodes_info = [database_per_run['nodes_info'], query_per_run['nodes_info']]
            sub_dataset = {
                'feature': features,
                'pose': poses,
                'mask': masks,
                'node_ids': nodes_info,
                'database_nodes_range': database_per_run['nodes_range'],
                'query_nodes_range': query_per_run['nodes_range'],
            }
        else:
            query_per_run = retrieve_info_per_run(
                dataset, 
                run_idx, 
                data_selector=0,
            )
            features = [query_per_run['features']]
            poses = [query_per_run['poses']]
            masks = [query_per_run['masks']]
            nodes_info = [query_per_run['nodes_info']]
            sub_dataset = {
                'feature': features,
                'pose': poses,
                'mask': masks,
                'node_ids': nodes_info,
                'query_nodes_range': query_per_run['nodes_range'],
            }
        database_per_run[run_idx] = sub_dataset
    return database_per_run


def main():
    # ---------------------------------------------------------------------------- #
    # configuration
    # ---------------------------------------------------------------------------- #
    parser = argparse.ArgumentParser(
        description='Evaluate the trained p-gat model'
    )
    parser.add_argument(
        "--config_file", 
        default="attentional_graph/config/param.yml", 
        metavar="FILE", 
        help="path to config file", 
        type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # Check if gpu acceleration is available
    print("is gpu available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    print("------------------------- Test configuration -------------------------")
    print(cfg.dump())
    print("----------------------------------------------------------------------")

    # ---------------------------------------------------------------------------- #
    # Data loading
    # ---------------------------------------------------------------------------- #
    normalized_test_data = build_dataset(
        cfg,
        0, # dataset id
        is_train=False,
    )
    has_database_dataset = len(normalized_test_data['features']) == 2
    test_iteration = DataLoader(TensorDataset(
        normalized_test_data['features'][-1],
        normalized_test_data['poses'][-1],
        normalized_test_data['masks'][-1],
        normalized_test_data['nodes_info'][-1],
        normalized_test_data['run_id'][-1],
    ), batch_size=1, shuffle=False)
    query_node = normalized_test_data['nodes_info'][-1]
    query_run = normalized_test_data['run_id'][-1]
    database_per_run = database_splitting(normalized_test_data, has_database_dataset)

    # ---------------------------------------------------------------------------- #
    # Model definition
    # ---------------------------------------------------------------------------- #
    super_glue_model = PoseGAT(
        pose_dim=cfg.MODEL.ATTENTION_GRAPH.POSE_DIM, 
        feature_dim=cfg.MODEL.ATTENTION_GRAPH.FEATURE_DIM, 
        keypoint_enc_hidden_dim=cfg.MODEL.ATTENTION_GRAPH.KEYPOINT_HIDDEN_DIM,
        include_pose=cfg.MODEL.ATTENTION_GRAPH.INCLUDE_POSE,
        num_heads=cfg.MODEL.ATTENTION_GRAPH.NUM_HEADS,
        num_layers=cfg.MODEL.ATTENTION_GRAPH.NUM_LAYERS,
        dropout=cfg.MODEL.ATTENTION_GRAPH.DROPOUT,
    ).to(device)
    print('Model parameters number:')
    print(sum(p.numel() for p in super_glue_model.parameters() if p.requires_grad))
    model_param = cfg.OUTPUT_DIR + cfg.TEST.MODEL_PARAM
    super_glue_model.load_state_dict(torch.load(model_param))

    # ---------------------------------------------------------------------------- #
    # Model Inferencing
    # ---------------------------------------------------------------------------- #  
    print('Inferencing ...')
    ground_truth_matrix = normalized_test_data['scores'].numpy()
    similarity_scores_matrix = np.stack(
        [np.zeros_like(ground_truth_matrix), np.zeros_like(ground_truth_matrix)]
    )
    for anchor_graph in tqdm.tqdm(test_iteration):
        run_id = anchor_graph[4].item()
        for database_run_idx in database_per_run:
            if database_run_idx == run_id: continue
            for database_query_indicator in range(2):
                if not has_database_dataset and database_query_indicator == 1:
                    # if the dataset only has query set, the maximum 
                    # database_query_indicator is 0
                    continue
                query = {
                    'feature': anchor_graph[0], 
                    'pose': anchor_graph[1], 
                    'mask': anchor_graph[2],
                    'node_ids': anchor_graph[3][0] # the loaded node_ids shape is [1, num_nodes]
                }
                similarity_scores_matrix = do_inference(
                    super_glue_model, 
                    query, 
                    database_per_run[database_run_idx],
                    database_query_indicator,
                    similarity_scores_matrix,
                    device,
                )
    # calculate the average similarity scores
    similarity_scores_matrix[1] -= 1e-6
    similarity_scores_matrix[0] /= similarity_scores_matrix[1]
    stat = evaluate(
        similarity_scores_matrix[0],
        ground_truth_matrix,
        query_node=query_node,
        query_run=query_run,
        database_per_run=database_per_run,
        has_database=has_database_dataset,
    )
    print(stat)
    with open(cfg.OUTPUT_DIR + 'evaluation_result.json', 'w') as result_file:
        json.dump(stat, result_file)

if __name__ == "__main__":
    main()