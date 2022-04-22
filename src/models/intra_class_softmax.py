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

from src.models.evaluate_model import test

## Function to compute the intra-class distance ##
m = torch.nn.Softmax(dim=0)

def intra_class_distance(out, data):
    
    out = out[data.train_mask] # try other masks or none
    pred = out.argmax(dim=1)
    N = len(pred)
    #print(N)
    w=0
    classes = np.unique(pred)
    #print("classes:",classes)

    for k in range(len(classes)):
        clk = classes[k] # class n°k
        # Sk = nodes in class k
        Sk = (pred == clk).nonzero(as_tuple=True)[0]

        # Compute ck
        ck = torch.Tensor(np.zeros(len(classes)))
        for i in Sk:
            #zi = torch.max(out[i]).item()
            zi = out[i].clone().detach()
            zi = m(zi)
            ck += zi
        ck = ck/len(Sk)

        for i in Sk:
            #zi = torch.max(out[i]).item()
            zi = out[i].clone().detach()
            zi = m(zi)
            w += torch.linalg.norm(zi-ck)
        
        #print(clk, ck, w)

    return w/N


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