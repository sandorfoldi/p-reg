import pandas as pd
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

# src 
from src.models.gcn import GCN1
from src.models.reg import make_l_abdul
from src.models.train_model import train_with_loss

from evaluate_model import acc
from evaluate_model import icd0


from src.abdul.random_split import random_split
from src.abdul.p_reg_loss import A_hat_computations




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root=f'/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)    

# Calculate A_hat as it's training invariant 
A_hat, A_hat_mask, N = A_hat_computations(data)
data.reg_mask = A_hat_mask


metrics = []
for seed in [0]:
    for mu in range(0, 41, 4):
        if mu == 0 and seed == 0:    
            print('-------------------------------------------------------------')
            print(f'train size: {data.train_mask.sum()}')
            print(f'val size: {data.val_mask.sum()}')
            print(f'test size: {data.test_mask.sum()}')
            print('-------------------------------------------------------------')
        
        mu = mu / 10.
        
        torch.manual_seed(seed)
        random.seed(seed)

        l_abdul = make_l_abdul(mu, A_hat)

        model = GCN1(
            num_node_features=dataset.num_features,
            num_classes=dataset.num_classes,
            hidden_channels=64, ).to(device)
        
        # train
        # model = train(l_abdul, model, data, mu, p_reg_dict, num_epochs=epochs)    
        model = train_with_loss(model, data, l_abdul, num_epochs=200)    

        train_acc, val_acc, test_acc = acc(model, data)
        
        d0 = icd0(model, data)
  
        
        metrics.append({
        'mu': mu, 
        'seed': seed,
        'train_acc': np.round(train_acc,4), 
        'val_acc': np.round(val_acc,4), 
        'test_acc': np.round(test_acc,4),
        'icd0': d0.item(),
        })

        print(metrics[-1])
df = pd.DataFrame(metrics)
df.to_csv('reports/figures/experiment_icd.csv')

df = pd.DataFrame(metrics)
df.to_csv('reports/figures/experiment_icd.csv')


