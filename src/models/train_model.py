import torch
import torch.nn.functional as F
import torch_geometric.utils as utils

from src.models.reg import compute_a_hat
from src.models.reg import reg_loss
from src.models.reg import cp_reg



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


def train_conf_pen(model, data, conf_pen_mask=None, lr=0.01, weight_decay=5e-4, num_epochs=100, beta=0.01):
    """ Train model"""

    # training the model
    # initializing the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # setting model to train mode
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        pred = model(data)
        
        loss = cp_reg(
            pred=F.softmax(pred, dim=1), # we make predictions on the entire dataset
            target=data.y[data.train_mask], # we assume to have ground-truth labels on only a subset of the dataset
            train_mask=data.train_mask, # hence, we also provide the train_mask, to know what nodes have labels
            conf_pen_mask=torch.ones_like(data.train_mask, dtype=torch.bool),
            L_cls=lambda x, y: F.nll_loss(torch.log(x), y),
            beta=beta,
            )

        loss.backward()
        optimizer.step()
    
    return model # not sure if inplace would work


def train_with_loss(model, data, loss_fn, lr=0.01, weight_decay=5e-4, num_epochs=100, beta=0.01):
    """ 
    Train model using given loss function
    Loss function must have the following structure:
    def loss(data: from torch_geometric.data import Data, pred: torch.tensor([num_nodes, num_classes])) -> scalar value

    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        pred = model(data)[data.train_mask]

        loss = loss_fn(data, pred)

        loss.backward()
        optimizer.step()
    
    return model # not sure if inplace would work

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


























