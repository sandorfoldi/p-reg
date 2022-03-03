import torch
import torch.nn.functional as F

from torch.nn import Linear

class NN0(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.dense1 = Linear(num_node_features, 64)
        self.dense2 = Linear(64, 64)
        self.dense3 = Linear(64, 32)
        self.dense4 = Linear(32, num_classes)

    def forward(self, data):
        x = data.x

        x = self.dense1(x)
        x = F.relu(x)
        
        x = self.dense2(x)
        x = F.relu(x)

        x = self.dense2(x)
        x = F.relu(x)

        x = self.dense2(x)
        x = F.relu(x)

        x = self.dense3(x)

        return F.log_softmax(x, dim=1)


class NN1(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.dense1 = Linear(num_node_features, 16)
        self.dense2 = Linear(16, num_classes)

    def forward(self, data):
        x = data.x

        x = self.dense1(x)
        x = F.relu(x)
        
        x = self.dense2(x)

        return F.log_softmax(x, dim=1)