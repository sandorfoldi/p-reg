import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.models.gcn import GCN1
from src.models.gat import GAT

from src.models.train_model import random_splits
from src.models.train_model import train_with_loss

from src.models.reg import compute_a_hat
from src.models.reg import set_reg_mask
from src.models.reg import make_preg_ce_ce

from src.models.evaluate_model import acc
from src.models.evaluate_model import icd_apolline_1

import torch

import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

a_hat = compute_a_hat(data)
mu = 0.7

metrics = []
for seed in range(4):
    for model_name in ('gcn', 'gat'):
        for alpha in  [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
            data = set_reg_mask(data, int(alpha*data.x.shape[0]))
            loss_fn = make_preg_ce_ce(mu, a_hat)
            torch.manual_seed(1)

            if model_name == 'gcn':
                model = GCN1(
                    num_node_features=dataset.num_node_features,
                    num_classes=dataset.num_classes,).to(device)

            elif model_name == 'gat':
                model = GAT(
                    dataset=dataset, 
                    hidden_channels=64).to(device)

            model = train_with_loss(model, data, loss_fn)

            train_acc, val_acc, test_acc = acc(model, data)
            icd = icd_apolline_1(model, data)

            metrics.append({
                'seed': seed,
                'model': model_name, 
                'unmask preg ratio': alpha, 
                'train_acc': np.round(train_acc, 4), 
                'val_acc': np.round(val_acc, 4),
                'test_acc': np.round(test_acc, 4),
                'icd': np.round(icd, 4),
                })

            print(metrics[-1])


df = pd.DataFrame(metrics)
df.to_csv('reports/swipe_preg_on_unmask_preg_ratio.csv')

fig, ax = plt.subplots()

filt = (df['model'] == 'gcn')
ax.plot(df[filt]['unmask preg ratio'], df[filt]['test_acc'], '-*b', label='GCN')

filt = (df['model'] == 'gat')
ax.plot(df[filt]['unmask preg ratio'], df[filt]['test_acc'], '-*r', label='GAT')

ax.set(ylim=(.77, .86), xlabel='Unmask p-reg ratio', ylabel='Test accuracy')
ax.legend()

plt.savefig('reports/swipe_preg_on_unmask_preg_ratio.png', dpi=300)
