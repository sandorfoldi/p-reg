# torch
import torch
import torch.nn.functional as F

# pyg
from torch_geometric.nn import GATConv

# other
import warnings
warnings.filterwarnings("ignore")


class GAT(torch.nn.Module):
    ''' GAT '''
    def __init__(self, dataset, hidden_channels=16):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * hidden_channels, dataset.num_classes,  heads=1, dropout=0.6)  

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        Z = x
        out   = F.log_softmax(x, dim=1)
        return Z