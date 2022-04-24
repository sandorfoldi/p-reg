# ========================================== 
# Dependencies
# ========================================== 
# torch
import torch

# other
import warnings
warnings.filterwarnings("ignore")




# ========================================== 
# Confidence Penalty Loss Function 
# ========================================== 
# ----------------- 
# Define functions 
# -----------------
def cp_loss(Z, beta=-1):
    """
    Confidence Penalty
    See section 5.1:    
    Label smoothing Confidence penalty adds the
    negative entropy of the network outputs to the classification
    loss as a regularizer
    """
    assert beta < 0 , "beta should be negative"
    P = torch.softmax(Z, dim=1)
    neg_entropy  = beta * (P * torch.log(P)).sum(dim=1).mean()  
    return neg_entropy