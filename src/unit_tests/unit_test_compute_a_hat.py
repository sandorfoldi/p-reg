import torch
from torch_geometric.data import Data
from src.models.train_model import compute_a_hat


x = torch.tensor([
    [1,2,3],
    [0,0,0],
    [2,3,2],
    [4,3,2]],
    dtype=torch.float)

edge_index = torch.tensor([
    [0,1],
    [0,2],
    [0,3],
    [2,3],])

data = Data(x=x, edge_index=edge_index.T)

A_hat = compute_a_hat(data)

A_hat_gt = torch.tensor([
    [0, 1/3, 1/3, 1/3],
    [1, 0, 0, 0],
    [0.5, 0, 0, 0.5],
    [0.5, 0, 0.5, 0]])

epsilon = torch.ones_like(A_hat)*1e-12

eq = torch.any(torch.less(torch.square(A_hat - A_hat_gt), epsilon)).item()

assert eq, 'ERROR in compute_a_hat()'
