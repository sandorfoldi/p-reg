import torch
import torch.nn.functional as F


def train0(model, data, lr=0.01, weight_decay=5e-4):
    # training the model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    
    return model # not sure if inplace would work


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