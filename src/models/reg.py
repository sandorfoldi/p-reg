import torch
import torch.nn.functional as F
import torch_geometric.utils as utils
import random


def make_preg_loss(L_cls, L_preg, mu, A_hat):
    """
    Returns a preg_loss function with the given parameters.
    """
    def l(data, Z):
        P = F.softmax(Z, dim=1)
        Q = F.softmax((A_hat@Z), dim=1)
        Y = data.y
        M = data.train_mask.sum()
        N = data.train_mask.shape[0]

        l_cls = L_cls(P[data.train_mask], Y[data.train_mask])
        l_preg = L_preg(P[data.reg_mask], Q[data.reg_mask])

        return 1 / M * l_cls + mu / N * l_preg
    return l


def make_confidence_penalty_loss(L_cls, beta):
    """
    Returns a confidence penalty loss function with the given parameters.
    """
    def l(data, pred):
        L_cls = L_cls(pred[data.train_mask], data.y[data.train_mask])
        H = -torch.sum(pred[data.reg_mask] * torch.log2(pred[data.reg_mask])) / pred.shape[0]

        return L_cls - beta * H
    return l


def compute_a_hat(data):
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

    # return A_hat, A_hat_mask, N
    return A_hat


def set_reg_mask(data, num_reg_nodes):
    """
    Sets data.reg_mask to contain num_reg_nodes number of nodes.
    """
    reg_indeces = random.sample(range(data.x.shape[0]), k=num_reg_nodes)
    reg_mask = torch.zeros(len(data.y), dtype=torch.bool)
    for i in reg_indeces:
        reg_mask[i] = True
    data.reg_mask = reg_mask
    return data



def reg_loss(pred, prop, target, train_mask, preg_mask, L_cls, L_preg, mu=0.2, phi='ce'):
    """
    Deprecated!
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
    Deprecated!
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


