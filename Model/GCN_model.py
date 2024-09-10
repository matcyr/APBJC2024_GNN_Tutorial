import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from Model.layers import GCNLayer
from torch_geometric.nn import GCNConv
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.pool import global_mean_pool as gap
from torch_geometric.nn.pool import global_max_pool as gmp
## GCN_toy is built for one graph, not for batched data

def toy_global_mean_pooling(H: torch.Tensor) -> torch.Tensor:
    """
    Global mean pooling: Aggregates node features by averaging them.
    Parameters:
    - H: Node feature matrix (nodes x features)
    
    Returns:
    - Aggregated graph-level feature vector (1 x features)
    """
    return torch.mean(H, dim=0, keepdim=True)


def toy_global_max_pooling(H: torch.Tensor) -> torch.Tensor:
    """
    Global max pooling: Aggregates node features by taking the maximum.
    Parameters:
    - H: Node feature matrix (nodes x features)
    
    Returns:
    - Aggregated graph-level feature vector (1 x features)
    """
    return torch.max(H, dim=0, keepdim=True).values
class GCN_toy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class, layers=2, pooling="mean"):
        """
        Initializes a multi-layer GCN model with global pooling.
        Parameters:
        - input_dim: Input feature dimension.
        - hidden_dim: Hidden layer dimension.
        - num_class: number of classes.
        - layers: Number of hidden layers.
        - pooling: Pooling method ('mean', 'max', or 'sum')
        """
        super(GCN, self).__init__()

        # Input layer (first GCN layer)
        self.init_layer = GCNLayer(input_dim, hidden_dim)
        
        # Hidden layers (additional GCN layers)
        self.hidden_layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(layers)])
        
        # Output layer (final GCN layer)
        self.out_layer = GCNLayer(hidden_dim, num_class)
        
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
            H_graph = toy_global_mean_pooling(H_out)
        elif self.pooling == "max":
            H_graph = toy_global_max_pooling(H_out)
        else:
            H_graph = torch.sum(H_out, dim=0, keepdim=True)  # Sum pooling
        return H_graph
    


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class, layers=2, pooling="mean"):
        """
        Initializes a multi-layer GCN model with global pooling.
        Parameters:
        - input_dim: Input feature dimension.
        - hidden_dim: Hidden layer dimension.
        - num_class: number of classes.
        - layers: Number of hidden layers.
        - pooling: Pooling method ('mean', 'max', or 'sum')
        """
        super(GCN, self).__init__()

        # Input layer (first GCN layer)
        self.init_layer = GCNLayer(input_dim, hidden_dim)
        
        # Hidden layers (additional GCN layers)
        self.hidden_layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(layers)])
        
        # Output layer (final GCN layer)
        self.out_layer = GCNLayer(hidden_dim, num_class)
        
        # Pooling method
        self.pooling = pooling

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the GCN model.
        Parameters:
        - data: Batched Data object from torch_geometric, contains node features, edge_index, and batch.
        
        Returns:
        - Graph-level output for classification.
        """
        # Extract node features (X), edge list (edge_index), and batch assignment
        X = data.x  # Node feature matrix (num_nodes x input_dim)
        edge_list = data.edge_index  # Edge list (2 x num_edges)
        batch = data.batch  # Batch vector that maps nodes to graphs
        
        # Pass through the initial GCN layer
        H = F.relu(self.init_layer(X, edge_list))
        
        # Pass through the hidden GCN layers
        for layer in self.hidden_layers:
            H = F.relu(layer(H, edge_list))
        
        # Output layer
        H_out = self.out_layer(H, edge_list)
        
        # Apply the specified pooling method to aggregate node features into graph-level features
        if self.pooling == "mean":
            H_graph = gap(H_out, batch)  # Mean pooling
        elif self.pooling == "max":
            H_graph = gmp(H_out, batch)  # Max pooling
        
        return H_graph




class GCN_pyg(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class, layers=2, pooling="mean"):
        """
        Initializes a multi-layer GCN model with global pooling.
        Parameters:
        - input_dim: Input feature dimension.
        - hidden_dim: Hidden layer dimension.
        - num_class: number of classes.
        - layers: Number of hidden layers.
        - pooling: Pooling method ('mean', 'max', or 'sum')
        """
        super(GCN_pyg, self).__init__()

        # First GCN Layer
        self.init_conv = GCNConv(input_dim, hidden_dim)

        # Additional GCN Layers
        self.hidden_convs = torch.nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(layers)]
        )

        # Output Layer (Final GCN Layer)
        self.out_conv = GCNConv(hidden_dim, num_class)

        # Pooling method
        self.pooling = pooling

    def forward(self, data):
        """
        Forward pass of the GCN model.
        Parameters:
        - data: Batched Data object from torch_geometric, containing node features, edge_index, and batch.
        
        Returns:
        - Graph-level output for classification.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Initial GCN Layer with ReLU activation
        x = F.relu(self.init_conv(x, edge_index))

        # Hidden GCN Layers with ReLU activation
        for conv in self.hidden_convs:
            x = F.relu(conv(x, edge_index))

        # Final GCN Layer (no activation before pooling)
        x = self.out_conv(x, edge_index)

        # Apply the specified global pooling method
        if self.pooling == "mean":
            x = gap(x, batch)  # Mean pooling for graph-level representation
        elif self.pooling == "max":
            x = gmp(x, batch)  # Max pooling for graph-level representation
        else:
            x = torch.sum(x, dim=0, keepdim=True)  # Sum pooling (optional)

        return x