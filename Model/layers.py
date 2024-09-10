import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Model.model_utils import GraphProcessor


def global_mean_pooling(H: torch.Tensor) -> torch.Tensor:
    """
    Global mean pooling: Aggregates node features by averaging them.
    Parameters:
    - H: Node feature matrix (nodes x features)
    
    Returns:
    - Aggregated graph-level feature vector (1 x features)
    """
    return torch.mean(H, dim=0, keepdim=True)


def global_max_pooling(H: torch.Tensor) -> torch.Tensor:
    """
    Global max pooling: Aggregates node features by taking the maximum.
    Parameters:
    - H: Node feature matrix (nodes x features)
    
    Returns:
    - Aggregated graph-level feature vector (1 x features)
    """
    return torch.max(H, dim=0, keepdim=True).values

class GCNLayer(nn.Module):
    def __init__(self, in_feat_dim, out_feat_dim, seed=42):
        """
        Initialize the GCN layer with random weights and zero biases.
        Parameters:
        - in_feat_dim: Number of input features (dimensionality of input node features).
        - out_feat_dim: Number of output features (dimensionality of output node features).
        """
        super(GCNLayer, self).__init__()  # Initialize nn.Module
        torch.manual_seed(seed)  # Set seed for reproducibility
        
        # Define the weight as a learnable parameter
        self.weight = nn.Parameter(torch.randn(in_feat_dim, out_feat_dim))
        
        # Define the bias as a learnable parameter
        self.bias = nn.Parameter(torch.zeros(out_feat_dim))

        self.A_normalized = None  # Normalized adjacency matrix will be calculated

    def forward(self, X: torch.Tensor, edge_list: list) -> torch.Tensor:
        """
        Forward pass of the GCN layer.
        Parameters:
        - X: Input feature matrix (nodes x input_features)
        - edge_list: List of edges in the graph (undirected).
        
        Returns:
        - Output feature matrix after applying GCN layer.
        """
        # Add self-loops and normalize the adjacency matrix
        graph_processor = GraphProcessor(edge_list)
        graph_processor.add_self_loops()
        self.A_normalized = graph_processor.normalize_adj_matrix()

        # Message passing (A_normalized * X)
        message_passing = torch.matmul(self.A_normalized, X)

        # Linear transformation (message_passing * weight + bias)
        output = torch.matmul(message_passing, self.weight) + self.bias
        
        # Apply ReLU activation function using PyTorch
        return F.relu(output)