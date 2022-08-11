import random

import numpy as np
import pandas as pd

from src.models.dense import NN0
from src.models.dense import NN1
from src.models.gcn import GCN0
from src.models.gcn import GCN1
from src.models.gcn import GCN_var_2layer

from src.models.gat import GAT

from src.models.train_model import train_with_loss
from src.models.train_model import random_splits

from src.models.reg import make_preg_ce_ce
from src.models.reg import make_preg_ce_ce_alt
from src.models.reg import make_lap_loss_ce

from src.models.reg import compute_a_hat

from src.models.evaluate_model import evaluate0
from src.models.evaluate_model import evaluate1

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.manifold import TSNE

from tqdm import tqdm

from mpl_toolkits.axes_grid1 import make_axes_locatable


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
for seed in [0,1,2,3,4]:
    for hidden_channels in [1, 2, 4, 8, 16, 32, 64, 128]:
        for mu in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
            torch.manual_seed(seed)
            random.seed(seed)

            loss_fn = make_preg_ce_ce_alt(mu, A_hat)
            
            model = GCN_var_2layer(hidden_channels)

            model = train_with_loss(model, data, loss_fn, num_epochs=200)

            acc = evaluate0(model, data)

            train_acc, val_acc, test_acc = evaluate1(model, data)
            metrics.append({'seed': seed, 'hidden_channels': hidden_channels, 'mu': mu, 'train_acc': np.round(train_acc,4), 'val_acc': np.round(val_acc,4), 'test_acc': np.round(test_acc,4)})
            print(metrics[-1])



df = pd.DataFrame(metrics)
# ridiculously slow implementation, but I don't want to figure this out now
arr = np.zeros((df['hidden_channels'].unique().shape[0], df['mu'].unique().shape[0]))
for ind_i, i in enumerate(df['hidden_channels'].unique()):
    for ind_j, j in enumerate(df['mu'].unique()):
        # print((df['hidden_channels'] == i) & (df['mu'] == j))
        # print(df[(df['hidden_channels'] == i) & (df['mu'] == j)]['test_acc'])
        arr[ind_i,ind_j] = df[(df['hidden_channels'] == i) & (df['mu'] == j)]['test_acc'].mean()

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

fig, ax = plt.subplots()
im = ax.pcolormesh([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.], [1, 2, 4, 8, 16, 32, 64, 128], arr, )
ax.set_yscale('log')
ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
ax.set_yticklabels([1, 2, 4, 8, 16, 32, 64, 128])
add_colorbar(im, fig, ax)
ax.set(xlabel='Regularization factor', ylabel='No. neurons in hidden layer')
ax.legend()

df.to_csv('reports/figures/acc_on_mu_and_neurons.csv')
plt.savefig('reports/figures/acc_on_mu_and_neurons.png', dpi=300)
