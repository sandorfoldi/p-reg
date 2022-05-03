# ========================================== 
# Dependencies
# ========================================== 
# torch
import torch

# pyg
import torch_geometric.utils as utils

# other
import warnings
warnings.filterwarnings("ignore")






# ========================================== 
# Label Smoothing 
# ========================================== 
# ----------------- 
# Define functions 
# -----------------
def label_smoothing(y_true, num_classes, mu):
    """
    source: https://github.com/pytorch/pytorch/issues/7455
    if mu == 0, it's one-hot method
    if 0 < mu < 1, it's smooth method
    """
    assert 0 <= mu < 1
    confidence = 1.0 - mu
    label_shape = torch.Size((y_true.size(0), num_classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=y_true.device)
        true_dist.fill_(mu / (num_classes - 1))
        true_dist.scatter_(1, y_true.data.unsqueeze(1), confidence)
    return true_dist

# To Do!