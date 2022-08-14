import random
# src 
from models import MLP, GCN, GAT
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

p_reg_dict = {
    'A_hat': A_hat, 
    'A_hat_mask': A_hat_mask, 
    'N': N, 
    'phi': p_reg_phi}  

metrics = []
# for seed in range(4):
for seed in [11, 12]:
    for mu in range(0, 21, 2):
        if mu == 0 and seed == 0:    
            print('-------------------------------------------------------------')
            print(f'train size: {data.train_mask.sum()}')
            print(f'val size: {data.val_mask.sum()}')
            print(f'test size: {data.test_mask.sum()}')
            print('-------------------------------------------------------------')
        
        mu = mu / 10.
        
        torch.manual_seed(seed)
        random.seed(seed)

        criterion = torch.nn.CrossEntropyLoss()

        l_abdul = make_l_abdul(criterion, p_reg_loss, mu, p_reg_dict)

        model = GCN(
            dataset,
            hidden_channels=64, 
            seed = 0).to(device)
        

        
        # train
        train(l_abdul, model, data, mu, p_reg_dict, num_epochs=epochs)    

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


