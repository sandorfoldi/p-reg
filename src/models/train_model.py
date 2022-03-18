import torch
import torch.nn.functional as F

import random


def train0(model, data, lr=0.01, weight_decay=5e-4, num_epochs=200):
    """ Train model with no preg, nll loss"""
    # training the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # print(loss)
        loss.backward()
        optimizer.step()
    
    return model # not sure if inplace would work


def train1(model, data, preg_mask=None, lr=0.01, weight_decay=5e-4, num_epochs=100, mu=0.01):
    """ Train model with preg, ce loss"""

    # If no preg mask is given, use the entire dataset
    if preg_mask is None:
        preg_mask=torch.ones_like(data.train_mask, dtype=torch.bool)

    # A_hat is the normalized adjacency matrix, needed for preg
    A_hat = compute_a_hat(data)

    # training the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = reg_loss(
            p=out, # we make predictions on the entire dataset
            t=data.y[data.train_mask], # we assume to have ground-truth labels on only a subset of the dataset
            train_mask=data.train_mask, # hence, we also provide the train_mask, to know what nodes have labels
            preg_mask=torch.ones_like(data.train_mask, dtype=torch.bool),
            A_hat=A_hat, 
            mu=mu)
        loss.backward()
        optimizer.step()
    
    return model # not sure if inplace would work


def compute_a_hat(data):
    """ Compute the A_hat, the normalized adjacency matrix, as in the paper"""
    a = torch.zeros(data.x.shape[0], data.x.shape[0])

    for i in data.edge_index.T:
        a[i[0], i[1]] = 1
        a[i[1], i[0]] = 1
    
    d = torch.diag(a.sum(dim=0))
    a_hat = d.inverse()@a
    return a_hat


def reg_loss(p, t, train_mask, preg_mask, A_hat, mu=0.2, phi='ce'):
    """
    Regularization loss
    float tensor[N,C] p: predictions on the entire dataset
    int tensor[M] t: list of ground-truth class indeces for the training nodes
    bool tensor[N] train_mask: bool map for selecting the training nodes from the entire dataset
    bool tensor A_hat[N, N]: adjacency matrix for the entire dataset
    float mu: regularization factor
    """

    L_cls = F.cross_entropy(p[train_mask], t)

    if phi == 'ce':
        L_preg = F.cross_entropy(p[preg_mask], (A_hat@p)[preg_mask])
    elif phi == 'l2':
        L_preg = F.mse_loss(p[preg_mask], (A_hat@p)[preg_mask])
    elif phi == 'kld':
        L_preg = F.kl_div(p[preg_mask], (A_hat@p)[preg_mask])
    else:
        raise ValueError('phi must be one of ce (cross_entropy), l2 (squared error) or kld (kullback-leibler divergence)')

    
    M = train_mask.sum()
    N = train_mask.shape[0]

    return 1 / M * L_cls + mu / N * L_preg


def random_splits(data, A, B):
    """
    Modify data.train_mask, data.valid_mask and data.test_mask
    So that there will be exactly A number of samples of each class in the train_mask
    Exactly B number of samples of each class in the valid_mask
    And the rest in the test mask
    """
    class_names = torch.unique(data.y)
    class_masks = [(data.y == classname).nonzero(as_tuple=False).numpy().reshape(-1).tolist() for classname in class_names]

    train_indeces = []
    valid_indeces = []
    test_indeces = []
    #print(class_masks[0].shape)

    ind = 0
    
    for class_mask in class_masks:
        class_mask = set(class_mask)
        add_to_train = set(random.sample(class_mask, k=A))
        class_mask -= add_to_train
        add_to_valid = set(random.sample(class_mask, k=B))
        class_mask -= add_to_valid
        add_to_test = class_mask
        train_indeces += add_to_train
        valid_indeces += add_to_valid
        test_indeces += add_to_test
        ind += 1

    train_mask = torch.zeros(len(data.y), dtype=torch.bool)
    for i in train_indeces:
        train_mask[i] = True

    valid_mask = torch.zeros(len(data.y), dtype=torch.bool)
    for i in valid_indeces:
        valid_mask[i] = True

    test_mask = torch.zeros(len(data.y), dtype=torch.bool)
    for i in test_indeces:
        test_mask[i] = True
    
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask
    
    return data
