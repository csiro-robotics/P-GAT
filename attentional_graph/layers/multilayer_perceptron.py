import torch.nn as nn

from .normalization import layer_norm

class Perceptron(nn.Module):
    '''
    Args:
        input_dim:
        output_dim:
    Returns:
        single layer perceptron output
    '''

    def __init__(
        self, 
        input_dim, 
        output_dim, 
        output_layer,
    ) -> None:
        super().__init__()
        self.perceptron = nn.ModuleList([])

        self.fc_layer = nn.Conv1d(
            input_dim, 
            output_dim, 
            kernel_size=1, 
            bias=True
        )
        self.perceptron.append(self.fc_layer)
        if not output_layer:
            self.perceptron.append(nn.ReLU())

    def forward(self, features, masks):
        # Fully connected layer
        features = self.perceptron[0](features)

        if len(self.perceptron) > 1:
            # normalization
            features = layer_norm(features, masks)
            # activate function
            features = self.perceptron[1](features)
        return features

class MultiLayerPerceptron(nn.Module):
    '''
    Args:
        layer_dim: list
           The output feature dimension of each layer
    Returns:
        Output
    '''

    def __init__(self, layer_dim, norm) -> None:
        super().__init__()
        layer_num = len(layer_dim)

        self.mlp = nn.ModuleList([])
        for layer_id in range(1, layer_num-1):
            self.perceptron = Perceptron(
                layer_dim[layer_id - 1],
                layer_dim[layer_id],
                output_layer=False,
            )
            self.mlp.append(self.perceptron)
        # Final layer of the multilayer perceptron
        # no need normalization and activation function
        self.perceptron = Perceptron(
            layer_dim[-2],
            layer_dim[-1],
            output_layer=True,
        )
        self.mlp.append(self.perceptron)

    def forward(self, features, masks):
        # swap the last two axes: 
        # from [batch_size, length_sequence, feature_dim] 
        # to [batch_size, feature_dim, length_sequence]
        features = features.transpose(1,2).contiguous()

        for perceptron in self.mlp:
            features = perceptron(features, masks)

        # swap the axes back
        # from [batch_size, feature_dim, length_sequence]
        # to [batch_size, length_sequence, feature_dim] 
        features = features.transpose(1,2).contiguous()
        
        return features
            