import torch
import torch.nn as nn
import torch.nn.functional as F

from .multilayer_perceptron import MultiLayerPerceptron

class AttentionAggregation(nn.Module):
    '''
    Aggregate an attention within or cross graph

    Args:
        embed_dim: total dimension of the model.
        num_heads: Number of parallel attention heads. Note that embed_dim will
                 be split across num_heads 
                 (i.e. each head will have dimension embed_dim // num_heads).

    Shape:
        - Input: [batch_size, node_num, feature_dim]
        - Output: [batch_size, node_num, feature_dim]

    Atrributes:

    Examples:

    '''

    def __init__(self, embed_dim, num_heads, dropout) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)

    def forward(
        self, 
        layer_representation, 
        cross_representation=None,
        key_padding_mask=None,    
    ):
        '''
        Args:
            layer_representation: [batch_size, 2, node_num, feature_dim]
                intermediate representation
                this varaible will be used as query, key, and value 
                    in self attention
                this varaible will be used as query in cross attention
            cross_representation: [batch_size, node_num, feature_dim]
                intermediate representation from external graph
                if it is not None, this function is cross attention and
                this variable is used as key and value.
            key_padding_mask: [batch_size, node_num]
                same definition as the key_padding_mask option in
                torch.nn.MultiHeadAttention

        Returns:
            ave_message: [batch_size, node_num, feature_dim]
                weighted average of the value in each nodes
        '''
        if cross_representation is not None:
            query = layer_representation
            key = value = cross_representation
        else:
            query = key = value = layer_representation
        ave_message, _ = self.attention(
            query, 
            key, 
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return ave_message


class AttentionalGraph(nn.Module):
    '''
    Apply attention aggregation to graph
    The formulation is inspired by SuperGlue (Sarlin et al. 2020)

    Args:
        pose_dim: Pose dimension e.g. 3 for x, y, and z
        feature_dim: Feature dimension for each pose
        num_heads: Number of parallel attention heads. 
        num_layers: Number of Attention Aggregation blocks.
    '''
    def __init__(
        self, 
        pose_dim, 
        feature_dim, 
        keypoint_enc_hidden_dim,
        include_pose, 
        num_heads=1, 
        num_layers=1, 
        normalization='batch', 
        dropout=0.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.include_pose = include_pose

        self.poses_mlp = MultiLayerPerceptron(
            [pose_dim] + keypoint_enc_hidden_dim + [feature_dim],
            norm=normalization,
        )
        self.attention_aggregation = nn.ModuleList([])
        self.aggregating_mlp = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.self_attention = AttentionAggregation(
                feature_dim, 
                num_heads,
                dropout=dropout)
            self.cross_attention = AttentionAggregation(
                feature_dim, 
                num_heads,
                dropout=dropout)
            self.self_mlp_layer = MultiLayerPerceptron(
                [2 * feature_dim, 2 * feature_dim, feature_dim],
                norm=normalization,
            )
            self.cross_mlp_layer = MultiLayerPerceptron(
                [2 * feature_dim, 2 * feature_dim, feature_dim],
                norm=normalization,
            )
            self.attention_aggregation.append(self.self_attention)
            self.attention_aggregation.append(self.cross_attention)
            self.aggregating_mlp.append(self.self_mlp_layer)
            self.aggregating_mlp.append(self.cross_mlp_layer)

    def embeding(self, features, poses, masks):
        '''
        Embed the position information into a high dimensional feature
        with a Multilayer Perceptron
        Refer to equation (2) in Sarlin et al. 2020

        Args:
            features: [batch_size, node_num, feature_dim]
                The descriptors for each pose
            poses: [batch_size, node_num, poses_dim]
                position coordinates for each pose
            masks: [batch_size, node_num]
                indication of zero paddings
        Returns:
            encoded_feature: [batch_size, node_num, feature_dim]
                joint feature representing appearance and position
        '''
        embeded_poses = self.poses_mlp(poses, masks)
        encoded_feature = features + embeded_poses
        return encoded_feature

    def aggregating(self, encoded_target, masks, message, layer):
        '''
        The residual message passing update
        Refer to equation (3) in Sarlin et al. 2020

        Args:
            encoded_target: [batch_size, node_num, feature_dim]
                joint feature for target graph
            masks: [batch_size, node_num]
                indication of zero paddings
            message: [batch_size, node_num, feature_dim]
                aggregation message from all keypoints in 
                corresponding graph
            layer: int
                the index of the layer
        Returns:
            encoded_target: [batch_size, node_num, feature_dim]
        '''
        concat_feature = torch.cat((encoded_target, message), dim=2)
        inter_feature = self.aggregating_mlp[layer](concat_feature, masks)
        encoded_target = inter_feature + encoded_target
        return encoded_target

    def forward(self, features, poses, masks):
        '''
        Args:
            features: [batch_size, 2, node_num, feature_dim]
                The descriptors for each pose in target graph
            poses: [batch_size, 2, node_num, poses_dim]
                position coordinates for each pose in target graph
            masks: [batch_size, 2, node_num]
                number of nodes masks as same as key_padding_mask
                option in torch.nn.MultiHeadAttention
        Returns:
            descriptor: [batch_size, 2, node_num, feature_dim]
                The final descriptors for the downstream tasks
        '''
        if self.include_pose:
            for iteration in range(2):
                features[:, iteration, :, :] = self.embeding(
                    features[:, iteration, :, :],
                    poses[:, iteration, :, :],
                    masks[:, iteration, :],
                )
        for layer in range(self.num_layers):
            # self attention
            for iteration in range(2):
                self_message = self.attention_aggregation[2 * layer](
                    features[:, iteration, :, :],
                    key_padding_mask=masks[:, iteration, :],
                )
                features[:, iteration, :, :] = self.aggregating(
                    features[:, iteration, :, :], 
                    masks[:, iteration, :],
                    self_message, 
                    2 * layer,
                )
            # cross attention
            for iteration in range(2):
                cross_message = self.attention_aggregation[2 * layer + 1](
                    features[:, iteration, :, :], 
                    features[:, 1 - iteration, :, :],
                    key_padding_mask=masks[:, 1 - iteration, :],
                )
                features[:, iteration, :, :] = self.aggregating(
                    features[:, iteration, :, :], 
                    masks[:, iteration, :],
                    cross_message, 
                    2 * layer + 1,
                )
        return features