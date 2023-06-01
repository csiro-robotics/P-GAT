import torch
import torch.nn as nn
import torch.nn.functional as F

from attentional_graph.layers import AttentionalGraph

class PoseGAT(nn.Module):
    '''
    PoseGAT feature matching
    Given two sets of keynodes and locations, we determine the
    correspondences

    Args:
        pose_dim: Pose dimension e.g. 3 for x, y, and z
        feature_dim: Feature dimension of each pose
        keypoint_enc_hidden_dim: list
            hidden layer dimension for position encoder
            default: [32, 64, 128]
        include_pose: boolean
        num_heads: int
        num_layers: int
        dropout: float
    '''

    def __init__(self, 
        pose_dim, 
        feature_dim, 
        keypoint_enc_hidden_dim,
        include_pose,
        num_heads,
        num_layers,
        dropout,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.attentional_graph = AttentionalGraph(
            pose_dim=pose_dim, 
            feature_dim=feature_dim,
            keypoint_enc_hidden_dim=keypoint_enc_hidden_dim,
            include_pose=include_pose,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.output_fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, features, poses, masks): 
        '''
        Args:
            features: [batch_size, 2, node_num, feature_dim]
                feature representation of each pose
            poses: [batch_size, 2, node_num, pose_dim]
                poses information
            masks: [batch_size, 2, node_num]
                number of nodes masks as same as key_padding_mask
                option in torch.nn.MultiHeadAttention
        Returns:
            scores: [batch_size, node_num, node_num]
            matching descriptor scores
        '''
        features = self.attentional_graph(features, poses, masks)
        # equation (6), linear projection
        features = self.output_fc(features)
        # convert each descripter vector in features to unit vector
        features = F.normalize(features, dim=3)
        # calculate the cosine similarity
        scores = torch.einsum(
            'bnd, bmd -> bnm', 
            features[:, 0, :, :], 
            features[:, 1, :, :]
        )
        # shift the range of scores from [-1, 1] to (0, 1)
        # values of scores should not be 0 or 1
        scores = scores / 2.0001 + 0.5
        return scores