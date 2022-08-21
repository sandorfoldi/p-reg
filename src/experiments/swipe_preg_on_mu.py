import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

# src
from src.models.train_model import random_splits
from src.models.train_model import train_with_loss

from src.models.gcn import GCN1

from src.models.reg import compute_a_hat
from src.models.reg import make_l_abdul
from src.models.reg import make_preg_ce_ce

from src.models.evaluate_model import acc

from src.models.evaluate_model import icd_saf_0
from src.models.evaluate_model import icd_saf_1
from src.models.evaluate_model import icd_saf_2
from src.models.evaluate_model import icd_saf_3

from src.visualization.visualize import gen_fig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

data.reg_mask = torch.ones_like(data.train_mask, dtype=torch.bool)
a_hat = compute_a_hat(data)

metrics = []
for seed in range(1):
    for mu in range(0, 21, 2):
        if mu == 0 and seed == 0:    
            print('-------------------------------------------------------------')
            print(f'train size: {data.train_mask.sum()}')
            print(f'val size: {data.val_mask.sum()}')
            print(f'test size: {data.test_mask.sum()}')
            print('-------------------------------------------------------------')
        
        mu = mu / 10.
        
        torch.manual_seed(seed)

        loss_fn = make_preg_ce_ce(mu, a_hat)
        
        model = GCN1(
            num_node_features=dataset.num_node_features,
            num_classes=dataset.num_classes,).to(device)

        model = train_with_loss(model, data, loss_fn)

        train_acc, val_acc, test_acc = acc(model, data)
        # icd = icd_apolline_1(model, data)
        icd0 = icd_saf_0(model, data)[2]
        icd1 = icd_saf_1(model, data)[2]
        icd2 = icd_saf_2(model, data)[2]
        icd3 = icd_saf_3(model, data)[2]
        

        metrics.append({
            'mu': mu, 
            'seed': seed,
            'train_acc': np.round(train_acc, 4), 
            'val_acc': np.round(val_acc, 4), 
            'test_acc': np.round(test_acc, 4),
            'icd0': np.round(icd0, 4),
            'icd1': np.round(icd1, 4),
            'icd2': np.round(icd2, 4),
            'icd3': np.round(icd3, 4),
            })

        print(metrics[-1])


df = pd.DataFrame(metrics)
df.to_csv('reports/swipe_preg_on_mu.csv')


fig, ax = plt.subplots()
ax.plot(df['mu'], df['icd1'], '-r')
ax.set(xlabel='Regularization factor $\mu$', ylabel='Intra class distnce')
plt.savefig('reports/swipe_preg_on_mu_icd.png', dpi=300)

fig, ax = plt.subplots()
ax.plot(df['mu'], df['train_acc'], '-r', label='train')
ax.plot(df['mu'], df['val_acc'], '-g', label='valid')
ax.plot(df['mu'], df['test_acc'], '-b', label='test')
ax.set(xlabel='Regularization factor $\mu$', ylabel='Accuracy')
plt.savefig('reports/swipe_preg_on_mu_acc.png', dpi=300)
