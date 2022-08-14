# ========================================== 
# Dependencies
# ========================================== 
# torch
import torch
from torch.nn import Linear
import torch.nn.functional as F

# pyg
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

# other
import warnings
warnings.filterwarnings("ignore")






# ========================================== 
# Models
# ========================================== 
# ----------------- 
# MLP
# -----------------
class MLP(torch.nn.Module):
    ''' MLP '''
    def __init__(self, dataset, hidden_channels=16, seed = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = Linear(dataset.num_features, hidden_channels)
        self.fc2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, data):
        x = data.x
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        Z = x
        out   = F.log_softmax(x, dim=1)
        return out, Z


# ----------------- 
# GCN
# -----------------
class GCN(torch.nn.Module):
    ''' GCN '''
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
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


# ----------------- 
# GAT
# -----------------
class GAT(torch.nn.Module):
    ''' GAT '''
    def __init__(self, dataset, hidden_channels=16, seed = 0):
        super().__init__()
        torch.manual_seed(seed)
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
        return out, Z