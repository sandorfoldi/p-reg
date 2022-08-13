# ----------------- 
# Load 
# ----------------- 
# torch
import torch

# other
import numpy as np
import warnings
warnings.filterwarnings("ignore")



# ========================================== 
# Random Split 
# ========================================== 
def random_split(dataset, A, B, seed = 0):
    """ random split function """
    # Shuffle y
    torch.manual_seed(seed)
    y = dataset[0].y
    y = y[torch.randperm(len(y))]

    # Assert inputs
    counts = np.unique(y.numpy(), return_counts=True)[1]
    assert not np.any(counts < (A+B)) , 'Check inputs: A & B'

    # Init masks
    N = dataset[0].x.shape[0]
    train_mask = torch.zeros(N).bool()
    val_mask   = torch.zeros(N).bool()
    test_mask  = torch.zeros(N).bool()

    # Fill masks
    if B==0:
      for class_ in range(dataset.num_classes):
          class_indxs = np.argwhere((y==class_).numpy()).flatten()        
          train_mask[class_indxs[0:A]] = True        
          test_mask[class_indxs[A+1:len(class_indxs)]] = True
    else:
      for class_ in range(dataset.num_classes):
          class_indxs = np.argwhere((y==class_).numpy()).flatten()        
          train_mask[class_indxs[0:A]] = True        
          val_mask[class_indxs[A:A+B]] = True
          test_mask[class_indxs[A+B:len(class_indxs)]] = True

    return train_mask, val_mask, test_mask