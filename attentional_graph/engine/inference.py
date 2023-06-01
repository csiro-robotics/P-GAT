from platform import node
import numpy as np
import torch
import time

def compute_on_dataset(model, sub_graph, query, train_test_indicator, device):
    model.eval()
    features = torch.stack(
        (query['feature'], sub_graph['feature'][train_test_indicator]), 
        dim=1,
    )
    poses = torch.stack(
        (query['pose'], sub_graph['pose'][train_test_indicator]), 
        dim=1,
    )
    masks = torch.stack(
        (query['mask'], sub_graph['mask'][train_test_indicator]),
        dim=1,
    )
    with torch.no_grad():
        features = features.to(device)
        poses = poses.to(device)
        masks = masks.to(device)
        output = model(features, poses, masks)
    return output

def repeat_query(query, number):
    for key in query:
        if key == 'node_ids' or key == 'mask':
            query[key] = query[key].repeat(number, 1)
        else:
            query[key] = query[key].repeat(number, 1, 1)
    return query

def do_inference(
    model,
    query,
    data_base,
    train_test_indicator,
    similarity_scores_matrix,
    device,
):
    '''
    Recognize the visited place with trained model

    input:
        model: trained model
        query: the query sub-graph
            [features, poses, node_ids]
        data_base:
            the sequence of visited sub-graphs, each sub-graph has same
            information as query
        add_query_to_database: boolean
            if it is True, the query will be added into database
    output:
        same_places: list of tuples of the pairs for same places
    '''
    num_subgraphs = data_base['feature'][train_test_indicator].shape[0]
    if num_subgraphs == 0:
        return similarity_scores_matrix
    query = repeat_query(query, num_subgraphs)
    query_id_start = query['node_ids'][0][0]
    query_num_node = query['node_ids'][0][-1]
    query_id_end = query_id_start + query_num_node
    similarity_score = compute_on_dataset(
        model, 
        data_base, query, 
        train_test_indicator, 
        device,
    )
    similarity_score = similarity_score.to(torch.device('cpu')).detach().numpy()
    data_base_node_ids = data_base['node_ids'][train_test_indicator]
    for data_base_idx in range(similarity_score.shape[0]):
        data_base_id_start = data_base_node_ids[data_base_idx][0]
        data_base_num_node = data_base_node_ids[data_base_idx][-1]
        data_base_id_end = data_base_id_start + data_base_num_node
        similarity_scores_matrix[
            0,
            query_id_start: query_id_end, 
            data_base_id_start: data_base_id_end
        ] += similarity_score[
            data_base_idx,
            :query_num_node,
            :data_base_num_node
        ]
        similarity_scores_matrix[
            1,
            query_id_start: query_id_end,
            data_base_id_start: data_base_id_end
        ] += np.ones_like(
            similarity_score[
                data_base_idx,
                :query_num_node,
                :data_base_num_node
            ]
        )

    return similarity_scores_matrix