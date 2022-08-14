import pandas as pd
import random
# src 
from models import MLP, GCN, GAT
from src.models.gcn import GCN1
from random_split import random_split
from p_reg_loss import A_hat_computations, p_reg_loss
from lap_loss import lap_loss
from cp_loss import cp_loss
from helper import visualize_Atlas, visualize_TSNE, visualize_CM, print_dataset, print_data
from utils import report_vis
from utils import report_stats
import torch 

from evaluate_model import acc, icd0, icd1, icd2, icd3, icd4

import torch_geometric.transforms as T

# packages
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import time

from src.models.reg import make_l_abdul
from src.abdul.train_model import train


reg_loss='p_reg'
p_reg_phi = 'cross_entropy'
epochs = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root=f'/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)    

# Calculate A_hat as it's training invariant 
A_hat, A_hat_mask, N = A_hat_computations(data)
data.reg_mask = A_hat_mask

p_reg_dict = {
    'A_hat': A_hat, 
    'A_hat_mask': A_hat_mask, 
    'N': N, 
    'phi': p_reg_phi}  

metrics = []
# for seed in range(4):
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
        model = train(l_abdul, model, data, mu, p_reg_dict, num_epochs=epochs)    

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

