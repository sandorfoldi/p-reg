## IMPORTS ##

import torch
import torch_geometric
import torch_sparse
import torch_scatter

import numpy as np
import random

from src.models.gcn import GCN0

from src.models.train_model import train
from src.models.train_model import random_splits

#from src.models.evaluate_model import evaluate0
#from src.models.evaluate_model import evaluate1

#import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


## Function to compute the intra-class distance ##
softmax = torch.nn.Softmax(dim=0)

def intra_class_distance_alt(Z, data, mask = None):
    with torch.no_grad():
        icds = [] # intra class distances
        for c in data.y.unique().numpy():
            s_k = Z[data.y == c]
            icds.append(s_k.std()**2)
        
        return np.array(icds).mean()


def intra_class_distance_alt_1(Z, data, mask = None):
    with torch.no_grad():
        icds = [] # intra class distances
        for c in data.y.unique().numpy():
            s_k = softmax(Z[data.y == c])
            icds.append(s_k.std()**2)
        
        return np.array(icds).mean()


'''
def intra_class_distance_alt(Z, data, mask = None):
    Z = Z.detach().numpy()
    Y = data.y.detach()
    classes = data.y.detach().unique().numpy()
    icds = [] # intra class distances
    for c in classes:
        s_k = Z[Y == c]
        icds.append(s_k.std()**2)
    
    return np.array(icds).mean()
'''

def main():
        
    ## Dataset + Training ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
    # no normalization, using n != 0 and m!= 0 instead
    # doesnt work, just use normalization instead
    # dataset = Planetoid(root='data/Planetoid', name='Cora')

    data = dataset[0].to(device)
    data = random_splits(data, 50, 20)
    splits = data.train_mask, data.val_mask, data.test_mask

    print('-------------------------------------------------------------')
    print(f'train size: {data.train_mask.sum()}')
    print(f'valid size: {data.valid_mask.sum()}')
    print(f'test size: {data.test_mask.sum()}')
    print('-------------------------------------------------------------')

    ## Loop over values of mu ##

    mus = []; dist_mus = []
    for mu in range(0,21,2):
        mu = mu / 10
        mus.append(mu)
        torch.manual_seed(1)
        random.seed(1)

        gcn_model = GCN0(num_node_features=dataset.num_node_features,
                        num_classes=dataset.num_classes) \
                        .to(device)

        gcn_model = train(gcn_model, data, mu=mu, num_epochs=1000)
        out = gcn_model(data)
        pred = gcn_model(data).argmax(dim=1)

        dist_mu = intra_class_distance(out, data)
        dist_mus.append(dist_mu)
        print("mu = {}, avg intra class dist = {}".format(mu, dist_mu))


    ## Plot ##
    fig, ax = plt.subplots()
    ax.plot(mus, dist_mus, '-b*')
    ax.set(ylim=(0, 1))
    plt.show()


if __name__ == '__main__':
    main()