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
# Laplacian Regularization Loss Function 
# ========================================== 
# ----------------- 
# Define functions 
# -----------------
def lap_loss(Z, data):
    """
    See section 4.1.3 from 
    Rethinking Graph Regularization for Graph Neural Networks
    """
    Z_sources = Z[data.edge_index[0],:]
    Z_targets = Z[data.edge_index[1],:]
    norm_l2 = (torch.norm(Z_sources - Z_targets, p=2, dim=1)**2).sum()
    return 1/Z_sources.shape[0] * norm_l2
