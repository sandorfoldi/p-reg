# ========================================== 
# Dependencies
# ========================================== 
# torch
import torch

# pyg
import torch_geometric.utils as utils

# other
_SEED_ = 123456
import warnings
warnings.filterwarnings("ignore")






# ========================================== 
# Regularized loss function 
# ========================================== 
# ----------------- 
# Define functions
# -----------------
def A_hat_computations(data):
    """ 
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


def P_reg(Z, A_hat,  A_hat_mask, N, phi = 'squared_error'):
    """
    """
    Z = Z[ A_hat_mask, :]
    Z_prime = torch.matmul(A_hat, Z)
    
    if phi == 'squared_error': 
        # have a look at the table before eq (2) and appendix A
        phi = (1/2) * (torch.norm(Z_prime - Z, p=2, dim=1)**2).sum()

    elif phi == 'cross_entropy':
        # have a look at the table before eq (2) and appendix A
        P = torch.softmax(Z, dim=1)
        Q = torch.softmax(Z_prime, dim=1)
        phi = - (P * torch.log(Q)).sum()

    elif phi == 'kl_divergence':            
        # have a look at the table before eq (2) and appendix A
        P = torch.softmax(Z, dim=1)                
        Q = torch.softmax(Z_prime, dim=1)
        logP = torch.log(P)
        logQ = torch.log(Q)
        phi = (P * (logP - logQ)).sum()

    else: 
        raise NotImplementedError()

    return (1/N) * phi


def regularized_loss(Z, mu, A_hat, A_hat_mask, data, N, criterion, phi = 'squared_error'):
    """
    """
    loss = criterion(Z[data.train_mask], data.y[data.train_mask]) + mu * P_reg(Z, A_hat,  A_hat_mask, N, phi = phi)
    return loss
