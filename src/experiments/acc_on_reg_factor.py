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
from src.models.reg import make_lap_loss_ce
from src.models.reg import compute_a_hat

from src.models.evaluate_model import acc

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

from sklearn.manifold import TSNE


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

for loss_fn_name in ['preg_loss', 'lap_loss',]:
    for model_name in ['gcn', 'gat']:
        for mu in range(9):
            torch.manual_seed(1)
            random.seed(1)

            mu = mu / 10

            if loss_fn_name == 'lap_loss':
                loss_fn = make_lap_loss_ce(mu)
            elif loss_fn_name == 'preg_loss':
                loss_fn = make_preg_ce_ce(mu, A_hat)
            else:
                raise(Exception('Not Implemented'))
            
            if model_name == 'gcn':
                model = GCN1(num_node_features=dataset.num_node_features,
                    num_classes=dataset.num_classes,
                    hidden_channels=16) \
                    .to(device)
            elif model_name == 'gat':
                model = GAT(dataset=dataset).to(device)
            else:
                raise(Exception('Not Implemented'))

            model = train_with_loss(model, data, loss_fn, num_epochs=200)


            train_acc, val_acc, test_acc = acc(model, data)
            metrics.append({'model': model_name, 'reg': loss_fn_name, 'mu': mu, 'train_acc': np.round(train_acc,4), 'val_acc': np.round(val_acc,4), 'test_acc': np.round(test_acc,4)})
            print(metrics[-1])


df = pd.DataFrame(metrics)
fig, ax = plt.subplots()

filt = (df['model'] == 'gcn') & (df['reg'] == 'preg_loss')
ax.plot(df[filt]['mu'], 9*[df[filt]['test_acc'][df[filt]['mu']==0]], '--b')

filt = (df['model'] == 'gat') & (df['reg'] == 'preg_loss')
ax.plot(df[filt]['mu'], 9*[df[filt]['test_acc'][df[filt]['mu']==0]], '--r')

filt = (df['model'] == 'gcn') & (df['reg'] == 'preg_loss')
ax.plot(df[filt]['mu'], df[filt]['test_acc'], '-*b')

filt = (df['model'] == 'gat') & (df['reg'] == 'preg_loss')
ax.plot(df[filt]['mu'], df[filt]['test_acc'], '-*r')

filt = (df['model'] == 'gcn') & (df['reg'] == 'lap_loss')
ax.plot(df[filt]['mu'], df[filt]['test_acc'], '-ob', label='GCN')

filt = (df['model'] == 'gat') & (df['reg'] == 'lap_loss')
ax.plot(df[filt]['mu'], df[filt]['test_acc'], '-or', label='GAT')

ax.set(ylim=(.5, .9), xlabel='Regularization factor $\mu$', ylabel='Test accuracy')
ax.legend()


df.to_csv('reports/figures/acc_on_reg_factor.csv')
plt.savefig('reports/figures/acc_on_reg_factor.png', dpi=300)
