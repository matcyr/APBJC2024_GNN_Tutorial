import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Model.model_utils import GraphProcessor



class GCNLayer(nn.Module):
    def __init__(self, input_dim_dim, out_feat_dim, seed=42):
        """
        Initialize the GCN layer with random weights and zero biases.
        Parameters:
        - input_dim_dim: Number of input features (dimensionality of input node features).
        - out_feat_dim: Number of output features (dimensionality of output node features).
        """
        super(GCNLayer, self).__init__()  # Initialize nn.Module
        torch.manual_seed(seed)  # Set seed for reproducibility
        
        # Define the weight as a learnable parameter
        self.weight = nn.Parameter(torch.randn(input_dim_dim, out_feat_dim))
        
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