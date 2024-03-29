import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN1(torch.nn.Module):
    ''' GCN '''
    def __init__(self, num_node_features, num_classes, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        Z = x
        out   = F.log_softmax(x, dim=1)
        return Z


class GCN_var_2layer(torch.nn.Module):
    ''' GCN '''
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv_1 = GCNConv(1433, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, 7)

    def forward(self, data):
        assert data.x.shape[1] == 1433, 'this model only works for data with 1433 node features'
        assert data.y.unique().shape[0] == 7, 'this model only works for data with 7 classes'

        x, edge_index = data.x, data.edge_index
        z = self.conv_1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_2(z, edge_index)
        
        return z
