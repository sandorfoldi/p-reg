from evaluate_model import icd_apolline_1
from evaluate_model import icd_apolline_2
from evaluate_model import icd_apolline_3
from evaluate_model import icd_apolline_4
from evaluate_model import icd_apolline_5
from evaluate_model import icd_saf_0
from evaluate_model import icd_saf_1

from gcn import GCN1

import torch

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

data.reg_mask = torch.ones_like(data.train_mask, dtype=torch.bool)


model = GCN1(
    num_node_features=dataset.num_node_features,
    num_classes=dataset.num_classes).to(device)

icd_apo_1_ = icd_apolline_1(model, data)
icd_apo_2_ = icd_apolline_2(model, data)
icd_apo_3_ = icd_apolline_3(model, data)
# icd_apo_4_ = icd_apolline_4(model, data)
# icd_apo_5_ = icd_apolline_5(model, data)

icd_saf_0_ = icd_saf_0(model, data)
icd_saf_1_ = icd_saf_1(model, data)

print(icd_apo_1_)
print(icd_apo_2_)
print(icd_apo_3_)
# print(icd_apo_4_)
# print(icd_apo_5_)
print(icd_saf_0_)
print(icd_saf_1_)