
import torch

class GraphProcessor:
    def __init__(self, edge_list):
        """
        Initialize the GraphProcessor with an edge list and number of nodes.
        Parameters:
        - edge_list: List of edges in the graph (undirected).
        """
        self.edge_list = edge_list
        self.num_nodes = max(torch.tensor(edge_list).flatten()) + 1
        self.adj_matrix = self.build_adj_matrix()
        
    def build_adj_matrix(self):
        """
        Converts the edge list to an adjacency matrix.
        Returns:
        - Adjacency matrix (num_nodes x num_nodes).
        """
        adj_matrix = torch.zeros((self.num_nodes, self.num_nodes))
        for edge in self.edge_list:
            adj_matrix[edge[0], edge[1]] = 1
        return adj_matrix

    def add_self_loops(self):
        """
        Adds self-loops to the adjacency matrix.
        """
        self.adj_matrix += torch.eye(self.num_nodes)

    def normalize_adj_matrix(self):
        """
        Normalizes the adjacency matrix using D^(-1/2) * A * D^(-1/2).
        Returns:
        - Normalized adjacency matrix.
        """
        # Compute degree matrix (sum of connections for each node)
        degree_matrix = torch.diag(torch.sum(self.adj_matrix, dim=1))
        
        # Compute D^(-1/2) (invert the square root of degree matrix)
        degree_matrix_inv_sqrt = torch.pow(degree_matrix, -0.5)
        degree_matrix_inv_sqrt[torch.isinf(degree_matrix_inv_sqrt)] = 0  # Set inf to 0 (for isolated nodes)
        
        # Return normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)
        adj_matrix_normalized = degree_matrix_inv_sqrt @ self.adj_matrix @ degree_matrix_inv_sqrt
        
        return adj_matrix_normalized
    
