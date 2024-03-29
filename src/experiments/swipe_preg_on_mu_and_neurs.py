import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.models.gcn import GCN_var_2layer

from src.models.train_model import train_with_loss
from src.models.train_model import random_splits

from src.models.reg import compute_a_hat
from src.models.reg import make_preg_ce_ce

from src.models.evaluate_model import acc
from src.models.evaluate_model import icd_saf_0
from src.models.evaluate_model import icd_saf_1
from src.models.evaluate_model import icd_saf_2
from src.models.evaluate_model import icd_saf_3

import torch

import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

A_hat = compute_a_hat(data)

data.reg_mask = torch.ones_like(data.train_mask, dtype=torch.bool)


print('-------------------------------------------------------------')
print(f'train size: {data.train_mask.sum()}')
print(f'val size: {data.val_mask.sum()}')
print(f'test size: {data.test_mask.sum()}')
print('-------------------------------------------------------------')

metrics = []
for seed in range(4):
    for hidden_channels in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for mu in range(0,21, 2):
            mu /= 10
            torch.manual_seed(seed)

            loss_fn = make_preg_ce_ce(mu, A_hat)
            
            model = GCN_var_2layer(hidden_channels)

            model = train_with_loss(model, data, loss_fn)

            train_acc, val_acc, test_acc = acc(model, data)

            icd0 = icd_saf_0(model, data)[2]
            icd1 = icd_saf_1(model, data)[2]
            icd2 = icd_saf_2(model, data)[2]
            icd3 = icd_saf_3(model, data)[2]
            
            metrics.append({
                'seed': seed, 
                'hidden_channels': hidden_channels, 
                'mu': mu, 
                'train_acc': np.round(train_acc,4), 
                'val_acc': np.round(val_acc,4), 
                'test_acc': np.round(test_acc,4),
                'icd0': np.round(icd0, 4),
                'icd1': np.round(icd1, 4),
                'icd2': np.round(icd2, 4),
                'icd3': np.round(icd3, 4),
                })
            
            print(metrics[-1])



df = pd.DataFrame(metrics)
df.to_csv('reports/swipe_preg_on_mu_and_neurs.csv')
