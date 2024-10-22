import torch
from torch_geometric.nn import GINConv, GCNConv
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
import torch.nn as nn

class DrpModel(torch.nn.Module):
    def __init__(self, max_atom_num, num_genes, n_output=1,embed_dim=128, embed_output_dim=128, graph_conv = 'GIN'):
        """
        Initializes the DrpModel class.
        Description:
        This model is designed for drug response prediction using Graph Neural Networks (GNNs). 
        It supports both Graph Isomorphism Networks (GIN) and Graph Convolutional Networks (GCN) 
        for graph convolution operations. The model consists of multiple graph convolution layers 
        followed by batch normalization and fully connected layers to combine graph and cell line 
        features.
        Args:
            max_atom_num (int): The maximum number of unique atoms in the dataset.
            num_genes (int): The number of genes in the cell line data.
            n_output (int, optional): The number of output neurons. Default is 1.
            embed_dim (int, optional): The embedding dimension for atom features. Default is 128.
            embed_output_dim (int, optional): The output dimension for the embedding layer. Default is 128.
            graph_conv (str, optional): The type of graph convolution to use ('GIN' or 'GCN'). Default is 'GIN'.
        Attributes:
            relu (nn.ReLU): ReLU activation function.
            n_output (int): The number of output neurons.
            atom_int_embed_nn (torch.nn.Embedding): Embedding layer for atom features.
            conv1, conv2, conv3, conv4, conv5: Graph convolution layers.
            bn1, bn2, bn3, bn4, bn5: Batch normalization layers.
            fc1_xd (Linear): Fully connected layer for graph features.
            cell_line_branch (nn.Sequential): Sequential model for cell line features.
            fc1, fc2, fc3 (nn.Linear): Fully connected layers for combined features.
        """
    
    
        super(DrpModel, self).__init__()
        dim = embed_dim
        self.relu = nn.ReLU()
        self.n_output = n_output
        self.atom_int_embed_nn = torch.nn.Embedding(max_atom_num, dim)
        # convolution layers
        nn1 = Sequential(Linear(dim,dim), ReLU(), Linear(dim, dim))
        if graph_conv == 'GIN':
            self.conv1 = GINConv(nn1)
        elif graph_conv == 'GCN':
            self.conv1 = GCNConv(dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        if graph_conv == 'GIN':
            self.conv2 = GINConv(nn2)
        elif graph_conv == 'GCN':
            self.conv2 = GCNConv(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        if graph_conv == 'GIN':
            self.conv3 = GINConv(nn3)
        elif graph_conv == 'GCN':
            self.conv3 = GCNConv(dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, embed_output_dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.fc1_xd = Linear(dim, embed_output_dim)

        self.cell_line_branch = nn.Sequential(Linear(num_genes, dim), nn.GELU(), Linear(dim, dim))

        # combined layers
        self.fc1 = nn.Linear(2*embed_output_dim, 1024)  # Adjusted input size based on CNN output
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

        # activation and regularization
        self.relu = nn.ReLU()

    def forward(self, exp, drug):
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x = self.atom_int_embed_nn(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))

        # RNA-seq input feed-forward:

        # 1d conv layers
        cell_line_features = self.cell_line_branch(exp)

        # flatten
        xt = cell_line_features.view(cell_line_features.size(0), -1)

        # concat
        combined_features = torch.cat((x, xt), 1)
        # add some dense layers
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x