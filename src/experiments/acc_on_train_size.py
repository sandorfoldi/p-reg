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

from src.models.evaluate_model import evaluate1

import torch

import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

A_hat = compute_a_hat(data)
mu = 0.6

loss_fn = make_preg_ce_ce(mu, A_hat)
metrics = []
for model_name in ('gcn', 'gat'):
    for num_training_nodes in  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        data = random_splits(data, num_training_nodes, 20)
        data.reg_mask = torch.ones_like(data.train_mask, dtype=torch.bool)
        splits = data.train_mask, data.val_mask, data.test_mask

        torch.manual_seed(1)
        random.seed(1)

        if model_name == 'gcn':
            model = GCN1(num_node_features=dataset.num_node_features,
                    num_classes=dataset.num_classes,
                    hidden_channels=64) \
                    .to(device)
        elif model_name == 'gat':
            model = GAT(dataset=dataset).to(device)

        model = train_with_loss(model, data, loss_fn, num_epochs=200)

        train_acc, val_acc, test_acc = evaluate1(model, data)
        metrics.append({'model': model_name, 'num_training_nodes': num_training_nodes, 'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc})
        print(metrics[-1])

df = pd.DataFrame(metrics)
fig, ax = plt.subplots(figsize=(6.4,4.8))

filt = (df['model'] == 'gcn')
ax.plot(df[filt]['num_training_nodes'], df[filt]['test_acc'], '-*b', label='GCN')

filt = (df['model'] == 'gat')
ax.plot(df[filt]['num_training_nodes'], df[filt]['test_acc'], '-*r', label='GAT')

ax.set(ylim=(.74, .9), xlabel='No. training nodes', ylabel='Test accuracy')
ax.legend()

df.to_csv('reports/figures/acc_on_train_size.csv')
plt.savefig('reports/figures/acc_on_train_size.png', dpi=300)
