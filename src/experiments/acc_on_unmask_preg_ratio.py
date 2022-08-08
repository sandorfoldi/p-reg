import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.models.gcn import GCN1
from src.models.gat import GAT

from src.models.train_model import train_with_loss
from src.models.train_model import random_splits

from src.models.reg import make_preg_ce_ce
from src.models.reg import compute_a_hat
from src.models.reg import set_reg_mask

from src.models.evaluate_model import evaluate0
from src.models.evaluate_model import evaluate1

import torch

import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

L_cls = lambda x, y: F.cross_entropy(x, y, reduction='sum')
L_preg = lambda x, y: - (x * torch.log(y)).sum()

A_hat = compute_a_hat(data)
mu = 0.6

metrics = []
for model_name in ('gcn', 'gat'):
      for alpha in  [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
        data = set_reg_mask(data, int(alpha*data.x.shape[0]))
        loss_fn = make_preg_ce_ce(mu, A_hat)
        torch.manual_seed(1)
        random.seed(1)

        if model_name == 'gcn':
            model = GCN1(num_node_features=dataset.num_node_features,
                    num_classes=dataset.num_classes,
                    hidden_channels=64) \
                    .to(device)
        elif model_name == 'gat':
            model = GAT(dataset=dataset, hidden_channels=64).to(device)

        model = train_with_loss(model, data, loss_fn, num_epochs=200)

        acc = evaluate0(model, data)

        train_acc, val_acc, test_acc = evaluate1(model, data)
        metrics.append({'model': model_name, 'unmask preg ratio': alpha, 'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc})
        print(metrics[-1])


fig, ax = plt.subplots()

filt = (df['model'] == 'gcn')
ax.plot(df[filt]['unmask preg ratio'], df[filt]['test_acc'], '-*b', label='GCN')

filt = (df['model'] == 'gat')
ax.plot(df[filt]['unmask preg ratio'], df[filt]['test_acc'], '-*r', label='GAT')

ax.set(ylim=(.77, .86), xlabel='Unmask p-reg ratio', ylabel='Test accuracy')
ax.legend()

df.to_csv('reports/figures/acc_on_reg_factor.csv')
plt.savefig('reports/figures/acc_on_reg_factor.png', dpi=300)
