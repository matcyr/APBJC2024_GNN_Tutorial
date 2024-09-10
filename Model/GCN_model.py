import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.layers import GCNLayer, global_mean_pooling, global_max_pooling

class GCN(nn.Module):
    def __init__(self, in_feat, hid_feat, num_class, layers=2, pooling="mean"):
        """
        Initializes a multi-layer GCN model with global pooling.
        Parameters:
        - in_feat: Input feature dimension.
        - hid_feat: Hidden layer dimension.
        - num_class: number of classes.
        - layers: Number of hidden layers.
        - pooling: Pooling method ('mean', 'max', or 'sum')
        """
        super(GCN, self).__init__()

        # Input layer (first GCN layer)
        self.init_layer = GCNLayer(in_feat, hid_feat)
        
        # Hidden layers (additional GCN layers)
        self.hidden_layers = nn.ModuleList([GCNLayer(hid_feat, hid_feat) for _ in range(layers)])
        
        # Output layer (final GCN layer)
        self.out_layer = GCNLayer(hid_feat, num_class)
        
        # Pooling method
        self.pooling = pooling

    def forward(self, X: torch.Tensor, edge_list: list) -> torch.Tensor:
        """
        Forward pass of the GCN model.
        Parameters:
        - X: Input feature matrix (nodes x input_features)
        - edge_list: List of edges in the graph (undirected).
        
        Returns:
        - Softmax probabilities for graph-level classification
        """
        # Pass through the initial GCN layer
        H = F.relu(self.init_layer(X, edge_list))
        
        # Pass through the hidden GCN layers
        for layer in self.hidden_layers:
            H = F.relu(layer(H, edge_list))
        
        # Output layer
        H_out = self.out_layer(H, edge_list)
        
        # Apply the specified pooling method to get graph-level features
        if self.pooling == "mean":
            H_graph = global_mean_pooling(H_out)
        elif self.pooling == "max":
            H_graph = global_max_pooling(H_out)
        else:
            H_graph = torch.sum(H_out, dim=0, keepdim=True)  # Sum pooling
        return H_graph