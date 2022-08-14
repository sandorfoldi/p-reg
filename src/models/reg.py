import torch
import torch.nn.functional as F
import torch_geometric.utils as utils
import random


def make_preg_ce_ce(mu, A_hat):
    """
    Returns a preg_loss function with 
    L_cls = crossentropy
    L_preg = crossentropy
    and the given parameters.
    """


    def l(data, Z):
        P = F.softmax(Z, dim=1)
        Q = F.softmax(torch.matmul(A_hat, Z), dim=1)
        Y = data.y
        M = data.train_mask.sum()
        N = data.train_mask.shape[0]
        
        L_cls = lambda x, y: F.cross_entropy(x, y, reduction='sum')
        L_preg = lambda x, y: - (x * torch.log(y)).sum()

        l_cls = L_cls(Z[data.train_mask], Y[data.train_mask])
        l_preg = L_preg(P[data.reg_mask], Q[data.reg_mask])

        return 1 / M * l_cls + mu / N * l_preg
    return l


def make_l_abdul(mu, A_hat):

    def l(data, Z):
        loss_1 = torch.nn.CrossEntropyLoss()(Z[data.train_mask], data.y[data.train_mask])
        
        Z_prime = torch.matmul(A_hat, Z)
        
        P = torch.softmax(Z[data.reg_mask], dim=1)
        Q = torch.softmax(Z_prime[data.reg_mask], dim=1)
        
        phi = - (P * torch.log(Q)).sum()
        loss_2 = (1/A_hat.shape[0]) * phi
        loss = loss_1 + mu * loss_2 
        return loss
    return l


def make_lap_loss_ce(mu):
    """
    See section 4.1.3 from 
    Rethinking Graph Regularization for Graph Neural Networks
    """

    def l(data, Z):
            P = F.softmax(Z, dim=1)
            Y = data.y

            L_cls = lambda x, y: F.cross_entropy(x, y, reduction='sum')
            l_cls = L_cls(Z[data.train_mask], Y[data.train_mask])

            Z_sources = Z[data.edge_index[0],:]
            Z_targets = Z[data.edge_index[1],:]
            l_lap = (torch.norm(Z_sources - Z_targets, p=2, dim=1)**2).sum()

            M = data.train_mask.sum()
            N = data.train_mask.shape[0]
            
            return 1 / M * l_cls + mu / N * l_lap
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

