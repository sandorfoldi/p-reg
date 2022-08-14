import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from src.models.gcn import GCN1
from src.abdul.models import GCN as GCN_ABDUL
from src.models.gat import GAT

from src.models.train_model import train_with_loss
from src.abdul.train_model import train as train_abdul
from src.models.train_model import random_splits

from src.models.reg import make_preg_ce_ce
from src.models.reg import make_lap_loss_ce
from src.models.reg import compute_a_hat
from src.models.reg import make_l_abdul

from src.models.evaluate_model import acc
from src.models.evaluate_model import icd0
from src.models.evaluate_model import icd1
from src.models.evaluate_model import icd2
from src.models.evaluate_model import icd3
from src.models.evaluate_model import icd4

from src.abdul.p_reg_loss import p_reg_loss
from src.abdul.p_reg_loss import A_hat_computations


import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)    

metrics = []
for seed in range(4):
    for mu in range(0, 21, 2):
        torch.manual_seed(seed)
        random.seed(seed)
        # data = random_splits(data, 80, 100)

        data.reg_mask = torch.ones_like(data.train_mask, dtype=torch.bool)
        # A_hat = compute_a_hat(data)
        A_hat, A_hat_mask, N = A_hat_computations(data)

        if mu == 0 and seed == 0:    
            print('-------------------------------------------------------------')
            print(f'train size: {data.train_mask.sum()}')
            print(f'val size: {data.val_mask.sum()}')
            print(f'test size: {data.test_mask.sum()}')
            print('-------------------------------------------------------------')

        mu = mu / 10

        # loss_fn = make_preg_ce_ce(mu, A_hat)
        criterion = torch.nn.CrossEntropyLoss()
        p_reg_dict = {
                'A_hat': A_hat, 
                'A_hat_mask': A_hat_mask, 
                'N': N, 
                'phi': 'cross_entropy'}  

        model = GCN_ABDUL(dataset,
            hidden_channels=64, 
            seed = 0 if seed is None else seed).to(device)

        l_abdul = make_l_abdul(criterion, p_reg_loss, mu, p_reg_dict)
        
        # train
        train_abdul(l_abdul, model, optimizer, criterion, data, mu, p_reg_dict, num_epochs=epochs)    

        train_acc, val_acc, test_acc = acc(model, data)

        d0 = icd0(model, data)
        d1 = icd1(model, data)
        d2 = icd2(model, data)
        d3 = icd3(model, data)
        d4 = icd4(model, data)

        metrics.append({
            'mu': mu, 
            'seed': seed,
            'train_acc': np.round(train_acc,4), 
            'val_acc': np.round(val_acc,4), 
            'test_acc': np.round(test_acc,4),
            'icd0': d0,
            'icd1': d1,
            'icd2': d2,
            'icd3': d3,
            'icd4_train': d4[0],
            'icd4_val': d4[1],
            'icd4_test': d4[2],
            })

        print(metrics[-1])


df = pd.DataFrame(metrics)
df.to_csv('reports/figures/icd_on_mu_abdul.csv')

fig, ax = plt.subplots()
ax.plot(df['mu'], df['icd0'], '-r')
ax.plot(df['mu'], df['icd1'], '-g')
ax.plot(df['mu'], df['icd2'], '-b')
ax.plot(df['mu'], df['icd3'], '-m')
ax.set(xlabel='Regularization factor $\mu$', ylabel='Intra class distnce')
plt.savefig('reports/figures/icd1-3_on_mu_abdul.png', dpi=300)


fig, ax = plt.subplots()
ax.plot(df['mu'], df['icd4_train'], '-r')
ax.plot(df['mu'], df['icd4_val'], '-g')
ax.plot(df['mu'], df['icd4_test'], '-b')
ax.set(xlabel='Regularization factor $\mu$', ylabel='Intra class distnce')
plt.savefig('reports/figures/icd4_on_mu_abdul.png', dpi=300)


fig, ax = plt.subplots()
ax.plot(df['mu'], df['train_acc'], '-r')
ax.plot(df['mu'], df['val_acc'], '-g')
ax.plot(df['mu'], df['test_acc'], '-b')
plt.savefig('reports/figures/acc_on_mu.png', dpi=300)

