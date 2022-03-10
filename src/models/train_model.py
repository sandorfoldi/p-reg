import torch
import torch.nn.functional as F


def train0(model, data, lr=0.01, weight_decay=5e-4, num_epochs=200):
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


def train1(model, data, lr=0.01, weight_decay=5e-4, num_epochs=100, mu=0.01):
    # training the model
    A = compute_a(data)
    D = torch.diag(A.sum(dim=0))
    A_hat = D.inverse()@A

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = reg_loss(
            p=out, # we make predictions on the entire dataset
            t=data.y[data.train_mask], # we assume to have ground-truth labels on only a subset of the dataset
            train_mask=data.train_mask, # hence, we also provide the train_mask, to know what nodes have labels
            A_hat=A_hat, 
            mu=mu)
        loss.backward()
        optimizer.step()
    
    return model # not sure if inplace would work


def compute_a(data):
    a = torch.zeros(data.x.shape[0], data.x.shape[0])

    for i in data.edge_index.T:
        a[i[0], i[1]] = 1

    return a


def reg_loss(p, t, train_mask, A_hat, mu=0.2, phi='ce'):
    """
    Regularization loss
    float tensor[N,C] p: predictions on the entire dataset
    int tensor[M] t: list of ground-truth class indeces for the training nodes
    bool tensor[N] train_mask: bool map for selecting the training nodes from the entire dataset
    bool tensor A_hat[N, N]: adjacency matrix for the entire dataset
    float mu: regularization factor
    """

    L_1 = F.cross_entropy(p[train_mask], t)

    if phi == 'ce':
        L_2 = F.cross_entropy(p, A_hat@p)
    elif phi == 'l2':
        L_2 = F.mse_loss(p, A_hat@p)
    elif phi == 'kld':
        L_2 = F.kl_div(p, A_hat@p)
    else:
        raise ValueError('phi must be one of ce (cross_entropy), l2 (squared error) or kld (kullback-leibler divergence)')

    
    M = train_mask.sum()
    N = train_mask.shape[0]

    return 1 / M * L_1 + mu / N * L_2


def set_masks(data, split):
    train_size = split[0]
    valid_size = split[1]
    test_size = split[2]

    assert sum([train_size, valid_size, test_size]) == 2708

    data.train_mask = torch.concat([
        torch.ones(train_size, dtype=torch.bool),
        torch.zeros(valid_size + test_size, dtype=torch.bool),])

    data.valid_mask = torch.concat([
        torch.zeros(train_size, dtype=torch.bool),
        torch.ones(valid_size, dtype=torch.bool),
        torch.zeros(test_size, dtype=torch.bool),])

    data.test_mask = torch.concat([
        torch.zeros(train_size + valid_size, dtype=torch.bool),
        torch.ones(test_size, dtype=torch.bool),])
    
    return data