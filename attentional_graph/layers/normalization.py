import torch

def layer_norm(batch_features, batch_masks):
    '''
    Layer normalization with masks
    Input:
        batch_features: [batch_size, feature_dim, max_length]
        batch_masks: [batch_size, max_length]
    output:
        norm_features: [batch_size, feature_dim, max_length]
    '''
    batch_size = batch_features.shape[0]
    feature_dim = batch_features.shape[1]
    max_length = batch_features.shape[2]

    batch_masks = (~batch_masks).int()
    batch_sum = torch.sum(batch_features, [1, 2])
    batch_count = torch.sum(batch_masks, 1) * feature_dim
    batch_mean = batch_sum / batch_count
    batch_mean = batch_mean.view(batch_size, 1, 1).repeat(
        1, feature_dim, max_length
    )
    norm_features = batch_features - batch_mean
    norm_features = norm_features * batch_masks.view(
        batch_size, 1, max_length
        ).repeat(1, feature_dim, 1)

    batch_square_sum = torch.sum(norm_features**2, [1, 2])
    batch_std = torch.sqrt(batch_square_sum / batch_count)
    batch_std = batch_std.view(batch_size, 1, 1).repeat(
        1, feature_dim, max_length
    )
    norm_features = norm_features / (batch_std + 1e-5)
    return norm_features