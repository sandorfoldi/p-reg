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


class GCN_fix_2layer(torch.nn.Module):
    ''' GCN '''
    def __init__(self):
        super().__init__()
        self.conv_1 = GCNConv(1433, 16)
        self.conv_2 = GCNConv(16, 7)

    def forward(self, data):
        assert data.x.shape[1] == 1433, 'this model only works for data with 1433 node features'
        assert data.y.unique().shape[0] == 7, 'this model only works for data with 7 classes'

        x, edge_index = data.x, data.edge_index
        z = self.conv_1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_2(z, edge_index)
        
        return z


class GCN_fix_2layer_64hidden(torch.nn.Module):
    ''' GCN '''
    def __init__(self):
        super().__init__()
        self.conv_1 = GCNConv(1433, 64)
        self.conv_2 = GCNConv(64, 7)

    def forward(self, data):
        assert data.x.shape[1] == 1433, 'this model only works for data with 1433 node features'
        assert data.y.unique().shape[0] == 7, 'this model only works for data with 7 classes'

        x, edge_index = data.x, data.edge_index
        z = self.conv_1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_2(z, edge_index)
        
        return z

class GCN_fix_3layer(torch.nn.Module):
    ''' GCN '''
    def __init__(self):
        super().__init__()
        self.conv_1 = GCNConv(1433, 256)
        self.conv_2 = GCNConv(256, 32)
        self.conv_3 = GCNConv(32, 7)

    def forward(self, data):
        assert data.x.shape[1] == 1433, 'this model only works for data with 1433 node features'
        assert data.y.unique().shape[0] == 7, 'this model only works for data with 7 classes'

        x, edge_index = data.x, data.edge_index
        z = self.conv_1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_2(z, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_3(z, edge_index)
        
        return z


class GCN_fix_4layer(torch.nn.Module):
    ''' GCN '''
    def __init__(self):
        super().__init__()
        self.conv_1 = GCNConv(1433, 512)
        self.conv_2 = GCNConv(512, 128)
        self.conv_3 = GCNConv(128, 16)
        self.conv_4 = GCNConv(16, 7)

    def forward(self, data):
        assert data.x.shape[1] == 1433, 'this model only works for data with 1433 node features'
        assert data.y.unique().shape[0] == 7, 'this model only works for data with 7 classes'

        x, edge_index = data.x, data.edge_index
        z = self.conv_1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_2(z, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_3(z, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_4(z, edge_index)
        
        return z


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

class GCN_64_node_2layer(torch.nn.Module):
    ''' GCN '''
    def __init__(self):
        super().__init__()
        self.conv_1 = GCNConv(1433, 128)
        self.conv_2 = GCNConv(128, 7)

    def forward(self, data):
        assert data.x.shape[1] == 1433, 'this model only works for data with 1433 node features'
        assert data.y.unique().shape[0] == 7, 'this model only works for data with 7 classes'

        x, edge_index = data.x, data.edge_index
        z = self.conv_1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)


        z = self.conv_2(x, edge_index)
        return z

class GCN_64_node_3layer(torch.nn.Module):
    ''' GCN '''
    def __init__(self):
        super().__init__()
        self.conv_1 = GCNConv(1433, 48)
        self.conv_2 = GCNConv(48, 30)
        self.conv_3 = GCNConv(30, 7)
        
    def forward(self, data):
        assert data.x.shape[1] == 1433, 'this model only works for data with 1433 node features'
        assert data.y.unique().shape[0] == 7, 'this model only works for data with 7 classes'

        x, edge_index = data.x, data.edge_index
        z = self.conv_1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_2(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_3(x, edge_index)

        return z

class GCN_64_node_4layer(torch.nn.Module):
    ''' GCN '''
    def __init__(self):
        super().__init__()
        self.conv_1 = GCNConv(1433, 42)
        self.conv_2 = GCNConv(42, 24)
        self.conv_3 = GCNConv(24, 12)
        self.conv_4 = GCNConv(12, 7)
        


    def forward(self, data):
        assert data.x.shape[1] == 1433, 'this model only works for data with 1433 node features'
        assert data.y.unique().shape[0] == 7, 'this model only works for data with 7 classes'

        x, edge_index = data.x, data.edge_index
        z = self.conv_1(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_2(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_3(x, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=0.5, training=self.training)

        z = self.conv_4(x, edge_index)

        return z
