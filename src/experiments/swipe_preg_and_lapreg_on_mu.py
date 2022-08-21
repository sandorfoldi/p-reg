import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.models.gcn import GCN1
from src.models.gat import GAT
from src.models.dense import NN1

from src.models.train_model import random_splits
from src.models.train_model import train_with_loss

from src.models.reg import compute_a_hat
from src.models.reg import make_preg_ce_ce
from src.models.reg import make_lap_loss_ce

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
    for loss_fn_name in ['preg_loss', 'lap_loss',]:
        for model_name in ['gcn', 'mlp']:
            for mu in range(9):
                torch.manual_seed(1)

                mu = mu / 10

                if loss_fn_name == 'lap_loss':
                    loss_fn = make_lap_loss_ce(mu)
                elif loss_fn_name == 'preg_loss':
                    loss_fn = make_preg_ce_ce(mu, A_hat)
                else:
                    raise(Exception('Not Implemented'))
                
                if model_name == 'gcn':
                    model = GCN1(
                        num_node_features=dataset.num_node_features,
                        num_classes=dataset.num_classes).to(device)

                elif model_name == 'gat':
                    model = GAT(dataset=dataset).to(device)
                
                elif model_name == 'mlp':
                    model = NN1(
                        num_node_features=dataset.num_node_features,
                        num_classes=dataset.num_classes).to(device)

                else:
                    raise(Exception('Not Implemented'))

                model = train_with_loss(model, data, loss_fn)


                train_acc, val_acc, test_acc = acc(model, data)
                icd0 = icd_saf_0(model, data)[2]
                icd1 = icd_saf_1(model, data)[2]
                icd2 = icd_saf_2(model, data)[2]
                icd3 = icd_saf_3(model, data)[2]

                metrics.append({
                    'seed': seed,
                    'model': model_name, 
                    'reg': loss_fn_name, 
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
df.to_csv('reports/swipe_preg_and_lapreg_on_mu.csv')


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

plt.savefig('reports/swipe_preg_and_lapreg_on_mu.png', dpi=300)
