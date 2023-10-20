import logging
import torch
import random

def random_pairing(index_batch, pair_info, pos_rate):
    target_index = []
    target_dataset_id = []
    for i in range(index_batch[0].shape[0]):
        query_index = index_batch[0][i].item()
        dataset_id = index_batch[1][i].item()
        if random.uniform(0,1) < pos_rate and len(pair_info[dataset_id][str(query_index)]) > 0:
            target_index.append(
                random.choice(pair_info[dataset_id][str(query_index)])
            )
            target_dataset_id.append(dataset_id)
        else:
            select_index = query_index
            while select_index == query_index:
                select_index = int(random.choice(list(pair_info[dataset_id])))
            target_index.append(select_index)
            target_dataset_id.append(dataset_id)
    return [torch.tensor(target_index), torch.tensor(target_dataset_id)]

def get_subset_for_pairs(num_nodes, index_batch, paired_index, node_info, scores):
    ground_truth_scores = []
    for i in range(index_batch[0].shape[0]):
        dataset_id = index_batch[1][i].item()
        query_idx = index_batch[0][i].item()
        target_idx = paired_index[0][i].item()
        query = node_info[dataset_id][query_idx][0]
        target = node_info[dataset_id][target_idx][0]
        ground_truth_scores.append(
            scores[dataset_id][query: query + num_nodes,
            target: target + num_nodes]
        )
    return torch.stack(ground_truth_scores)

def get_batched_data(data, key, index_batch, num_nodes):
    batched_data = []
    for i in range(index_batch[0].shape[0]):
        subgraph_id = index_batch[0][i].item()
        dataset_id = index_batch[1][i].item()
        cur_length = data[key][dataset_id][subgraph_id].shape[0]
        if key == 'masks':
            batched_data.append(torch.ones(num_nodes) == 1)
            batched_data[-1][:cur_length] = data[key][dataset_id][subgraph_id]
        else:
            batched_data.append(
                torch.zeros(num_nodes, data[key][dataset_id][subgraph_id].shape[-1])
            )
            batched_data[-1][:cur_length, :] = data[key][dataset_id][subgraph_id]
    return torch.stack(batched_data)

def create_pairs(num_nodes, normalized_data, index_batch, pos_rate):
    scores = normalized_data['scores']
    switches = normalized_data['switches']
    pair_info = normalized_data['paired_info']
    node_info = normalized_data['nodes_info']
    query_features = get_batched_data(normalized_data, 'features', index_batch, num_nodes)
    query_poses = get_batched_data(normalized_data, 'poses', index_batch, num_nodes)
    query_masks = get_batched_data(normalized_data, 'masks', index_batch, num_nodes)
    paired_index = random_pairing(index_batch, pair_info, pos_rate)
    target_features = get_batched_data(normalized_data, 'features', paired_index, num_nodes)
    target_poses = get_batched_data(normalized_data, 'poses', paired_index, num_nodes)
    target_masks = get_batched_data(normalized_data, 'masks', paired_index, num_nodes)
    subset_score = get_subset_for_pairs(
        num_nodes,
        index_batch, 
        paired_index, 
        node_info,
        scores,
    )
    subset_switches = get_subset_for_pairs(
        num_nodes,
        index_batch, 
        paired_index, 
        node_info,
        switches,
    )
    features = torch.stack((query_features, target_features), dim=1)
    poses = torch.stack((query_poses, target_poses), dim=1)
    masks = torch.stack((query_masks, target_masks), dim=1)
    return features, poses, subset_score, masks, subset_switches

def weight_loss(loss, scores, weights, masks, switches, device):
    '''
    Provide loss from positive pair a higher weight, and
    loss from negative pair a lower weight. 
    Ignore the pair with distance between 10 - 50 m

    Input:
        loss: [batchsize, num_nodes**2]
            The cross entropy loss for each pair
        weight: list
            The weight for the list, the weight[0] is the positive weight,
            and the weight[1] is the negative weight
        masks: boolean 
            [batchsize, 2, num_nodes]
            The mask to indicate the padding
        switch: [batchsize, num_nodes**2]
            The contribution to the loss for each pair
    output:
        total_loss: float
        The reduction (average) of the loss array
    '''
    scores = scores.flatten(start_dim=1)
    switches = switches.flatten(start_dim=1)
    masks = (~masks).int()
    scores_masks = torch.einsum(
        'bn, bm -> bnm', 
        masks[:, 0, :],
        masks[:, 1, :],
    ).flatten(start_dim=1)
    #weight_ = torch.tensor(weights)[scores.data.view(-1).long()].view_as(scores)
    #weight_ = weight_.to(device)
    #loss = loss * weight_ * switches
    loss = loss * switches
    loss = loss * scores_masks
    return torch.mean(loss)

def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats

 

def do_train(
    cfg,
    model,
    normalized_data,
    index_loader,
    num_nodes,
    optimizer,
    criterion,
    device,
):
    logger = logging.getLogger("attentional_graph.trainer")
    logger.info("Start training")
    model.train()
    batch_stats = {}
    running_stats =[]

    for iteration, index_batch in enumerate(index_loader):
        features, poses, scores, masks, switches = create_pairs(
            num_nodes,
            normalized_data, 
            index_batch, 
            cfg.TRAIN.POS_RATE,
        )
        iteration = iteration + 1
        
        features = features.to(device)
        poses = poses.to(device)
        if torch.isnan(poses).max():
            continue
        masks = masks.to(device)
        scores = scores.to(device)
        switches = switches.to(device)

        optimizer.zero_grad()
        output = model(features, poses, masks)
        loss = criterion(output.flatten(start_dim=1), scores.flatten(start_dim=1))
        loss = weight_loss(
            loss, 
            scores, 
            cfg.TRAIN.WEIGHT, 
            masks,
            switches,
            device,
        )
        loss.backward()
        optimizer.step()
        
        temp_stats = {'loss':loss.item()}
        temp_stats = tensors_to_numbers(temp_stats)
        running_stats.append(temp_stats)   
    return running_stats      