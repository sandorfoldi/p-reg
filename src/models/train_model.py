import torch
import torch.nn.functional as F

import random


def train(model, data, preg_mask=None, lr=0.01, weight_decay=5e-4, num_epochs=100, mu=0.01):
    """ Train model"""

    # If no preg mask is given, use the entire dataset
    if preg_mask is None:
        preg_mask=torch.ones_like(data.train_mask, dtype=torch.bool)

    # A_hat is the normalized adjacency matrix, needed for preg
    A_hat = compute_a_hat(data)

    # training the model
    # initializing the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # setting model to train mode
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        pred = model(data)
        prop = A_hat@pred
        
        loss = reg_loss(
            pred=F.softmax(pred, dim=1), # we make predictions on the entire dataset
            prop=F.softmax(prop, dim=1), # 
            target=data.y[data.train_mask], # we assume to have ground-truth labels on only a subset of the dataset
            train_mask=data.train_mask, # hence, we also provide the train_mask, to know what nodes have labels
            preg_mask=torch.ones_like(data.train_mask, dtype=torch.bool),
            L_cls=lambda x, y: F.nll_loss(torch.log(x), y),
            L_preg=lambda x, y: F.cross_entropy(x, y),
            mu=mu,
            )

        loss.backward()
        optimizer.step()
    
    return model # not sure if inplace would work


def compute_a_hat(data):
    """ Compute the A_hat, the normalized adjacency matrix, as in the paper"""
    a = torch.zeros(data.x.shape[0], data.x.shape[0])

    for i in data.edge_index.T:
        if i[0] != i[1]:
            a[i[0], i[1]] = 1
            a[i[1], i[0]] = 1
    
    d = torch.diag(a.sum(dim=0))
    a_hat = d.inverse()@a
    return a_hat


def reg_loss(pred, prop, target, train_mask, preg_mask, L_cls, L_preg, mu=0.2, phi='ce'):
    """
    Regularization loss
    Arguments:
    float tensor[N,C] pred: predictions on the entire dataset
    float tensor[N,C] prop: propagated node values on the entire dataset
    int tensor[M] target: list of ground-truth class indeces for the training nodes
    bool tensor[N] train_mask: bool map for selecting the training nodes from the entire dataset
    bool tensor[N] preg_mask: book map for selecting the nodes to compute regularization loss on
    float mu: regularization factor
    func L_cls(pred[train_mask], t): classification loss function
    func L_preg(pred[preg_mask], prop[preg_mask]): preg loss function
    """

    L_cls = L_cls(pred[train_mask], target)
    L_preg = L_preg(pred[preg_mask], prop[preg_mask])

    # Setting M!=0 and N!=0 messes up training entirely
    # Instead, we normalize the dataset, not totally sure if this makes sense though
    #     
    # M = train_mask.sum()
    # N = train_mask.shape[0]

    M = 1
    N = 1

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
