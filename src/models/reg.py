import torch
import torch.nn.functional as F

import random


def compute_a_hat(data):
    """ Compute A_hat, the normalized adjacency matrix, as in the paper"""
    a = torch.zeros(data.x.shape[0], data.x.shape[0])

    for i in data.edge_index.T:
        if i[0] != i[1]:
            a[i[0], i[1]] = 1
            a[i[1], i[0]] = 1
    
    d = torch.diag(a.sum(dim=0))
    a_hat = d.inverse()@a
    return a_hat


def A_hat_computations(data):
    """
    Abdul's implementation
    Compute A_hat, the normalized adjacency matrix, as in the paper
    """
    edge_index = data.edge_index
    edge_index, _ =  utils.remove_self_loops(edge_index)
    edge_index, _, A_hat_mask = utils.remove_isolated_nodes(edge_index)

    A = utils.to_dense_adj(edge_index).squeeze()
    D = torch.diag(utils.degree(edge_index[0]))

    A_hat = torch.linalg.solve(D, A)
    A_hat.requires_grad = False
    N = A_hat.shape[0]

    return A_hat, A_hat_mask, N


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


def cp_reg(pred,
           target,
           train_mask,
           L_cls=lambda x, y: F.nll_loss(torch.log(x), y),
           conf_pen_mask = None,
           beta=0.1):
    """
    Confidence penalty inhanced loss function.
    """
    # classification loss
    L_cls = L_cls(pred[train_mask], target)

    # confidence penalty loss
    if conf_pen_mask is None:
        conf_pen_mask = torch.ones_like(preds, dtype=torch.bool),

    # H stands for entropy, it is usually computed using the 2 logarithm
    H = -torch.sum(pred[conf_pen_mask] * torch.log2(pred[conf_pen_mask])) / pred.shape[0]
    return L_cls - beta * H