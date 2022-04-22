import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN0(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64, cached=True)
        self.conv2 = GCNConv(64, num_classes, cached=True)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(F.relu(self.conv1(x, edge_index)), training=self.training)
        x = self.conv2(x, edge_index)

        return x

class GCN1(torch.nn.Module):
    ''' GCN '''
    def __init__(self, num_node_features, num_classes, hidden_channels=16, seed = 123456):
        super().__init__()
        torch.manual_seed(seed)
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