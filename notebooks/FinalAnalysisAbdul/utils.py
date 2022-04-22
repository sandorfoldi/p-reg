# ========================================== 
# Load Dependencies
# ========================================== 
# src 
from models import MLP, GCN, GAT
from random_split import random_split
from p_reg_loss import A_hat_computations, p_reg_loss
from lap_loss import lap_loss
from helper import visualize_Atlas, visualize_TSNE, visualize_CM, print_dataset, print_data

# packages
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import time


# ==========================================
# Train and Evaluation functions 
# ==========================================
# Train function
def train(model, optimizer, criterion, data, train_mask, mu, reg_loss, p_reg_dict=None):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    _, Z = model(data)     # Perform a single forward pass.    
    loss_1 = criterion(Z[train_mask], data.y[train_mask])  # Compute the loss solely based on the training nodes.
    if reg_loss == 'p_reg':
      loss_2 = p_reg_loss(Z, 
                          p_reg_dict['A_hat'], 
                          p_reg_dict['A_hat_mask'], 
                          p_reg_dict['N'], 
                          phi = p_reg_dict['phi'])
    elif reg_loss == 'lap_reg': 
      loss_2 = lap_loss(Z, data) 
    elif reg_loss == 'no_reg':
      loss_2 = 0
      assert mu == 0
    loss = loss_1 + mu * loss_2      
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, Z

# Evaluation
def test(model, data, mask):
    model.eval()
    _, Z = model(data)
    y_pred = Z.argmax(dim=1)  # Use the class with highest probability.
    score = torch.softmax(Z, dim=1)
    acc = accuracy_score    (y_true = data.y[mask].cpu().detach().numpy(), y_pred  = y_pred[mask].cpu().detach().numpy())
    rms = mean_squared_error(y_true = data.y[mask].cpu().detach().numpy(), y_pred  = y_pred[mask].cpu().detach().numpy(), squared=False)
    roc = roc_auc_score     (y_true = data.y[mask].cpu().detach().numpy(), y_score = score [mask, :].cpu().detach().numpy(), multi_class='ovr')
    return acc, rms, roc, y_pred

# ==========================================
# Main: Train Evaluation Loop function
# ==========================================
def train_evaluation(A = None,
                     As = None, 
                     B = None,
                     Bs = None,   # 180 is the max for sum A and B 
                     seed = 123456,
                     seeds = None,
                     mu = 0,
                     mus = None,
                     phi = 'cross_entropy',
                     epochs = 201,
                     datasets = ['Cora', 'CiteSeer', 'PubMed'],
                     show = True,
                     reg_loss = None):
    
    # check regulization loss
    if (mu != 0) or (mus is not None):
        assert reg_loss in ['p_reg', 'lap_reg'], "reg_loss must be from ['p_reg', 'lap_reg']"
    else: reg_loss= 'no_reg'

    # define loop parameter
    if mus is not None:
        params, params_len, params_tag = mus, len(mus),'mu'
        assert mu == 0, "mu and mus are given in the same time"
        if show: print(f'Training with different "Mus" (with {reg_loss}):')
    elif seeds is not None:
        params, params_len, params_tag = seeds, len(seeds),'seed'
        if show: print(f'Training with different "Seeds" (with {reg_loss}):')
    elif As is not None: 
        params, params_len, params_tag = As, len(As),'No. Training Nodes'
        assert A is None, "A and As are given in the same time"
        assert B is None, "B and Bs are given in the same time"
        if show: print(f'Training with different "Number Training Nodes" (with {reg_loss}):')
    else:
        raise NotImplemented('The input Combination is not supported')

    # evaluation Storage
    Results = {}
    
    # loop over datasets
    for dataset_name in datasets: 
        # define dataset
        if show: print(f'=========================\n{dataset_name}:')
        if dataset_name == 'Cora':
            if ABDUL_G_DRIVE: dataset = Planetoid(root=f'data/Planetoid', name='Cora', transform=NormalizeFeatures())
            else: dataset = Planetoid(root=f'../../data/Planetoid', name='Cora', transform=NormalizeFeatures())
        elif dataset_name == 'CiteSeer':
            if ABDUL_G_DRIVE: dataset = Planetoid(root=f'data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
            else: dataset = Planetoid(root=f'../../data/Planetoid', name='CiteSeer', transform=NormalizeFeatures())
        elif dataset_name == 'PubMed':
            if ABDUL_G_DRIVE: dataset = Planetoid(root=f'data/Planetoid', name='PubMed', transform=NormalizeFeatures())         
            else: dataset = Planetoid(root=f'../../data/Planetoid', name='PubMed', transform=NormalizeFeatures())         
        else:
            raise NotImplementedError('Only Cora, CiteSeer and PubMed datasets are supported')
        
        # define data
        data = dataset[0]
        data = data.to(device)
        
        # Calculate A_hat as it's training invariant 
        if reg_loss == 'p_reg':
            A_hat, A_hat_mask, N = A_hat_computations(data)

        # define dataset evaluation metrices
        Results[dataset_name+'_acc']  = [],[]   # val, test
        Results[dataset_name+'_rms']  = [],[]
        Results[dataset_name+'_roc']  = [],[]
        Results[dataset_name+'_loss'] = []      # train

        # loop over params
        for i in range(params_len):
            # define model parameters        
            if params_tag == 'mu': 
                mu = mus[i]
            elif params_tag == 'seed': 
                seed = seeds[i]
            elif params_tag == 'No. Training Nodes': 
                A, B = As[i], Bs[i]

            # calc the split
            if (A is not None) & (B is not None):
                train_mask, val_mask, test_mask = random_split(dataset, A, B, seed)
            else:
                train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

            # define MOC
            model = GCN(dataset, hidden_channels=16, seed = seed).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()

            # define epoch evaluation metrices
            val_acc_lst , val_rms_lst , val_roc_lst  = [], [], []
            test_acc_lst, test_rms_lst, test_roc_lst = [], [], []
            loss_lst = []

            # loop over epochs
            # Z = None # can be used for visulizing the embaddings 
            for epoch in range(epochs):
                print(data.train_mask.sum())
                # train
                if reg_loss == 'p_reg':  
                    p_reg_dict = {'A_hat':A_hat, 'A_hat_mask':A_hat_mask, 'N':N, 'phi':phi}
                    loss, _ = train(model, optimizer, criterion, data, train_mask, mu, reg_loss, p_reg_dict)
                else: # handels both no_reg and lap_reg cases 
                    loss, _ = train(model, optimizer, criterion, data, train_mask, mu, reg_loss)

                # evaluate 
                with torch.no_grad():
                  val_acc , val_rms , val_roc , _ = test(model, data, data.val_mask)
                  test_acc, test_rms, test_roc, _ = test(model, data, data.test_mask)

                  # save metrices 
                  val_acc_lst .append(val_acc) 
                  val_rms_lst .append(val_rms) 
                  val_roc_lst .append(val_roc) 
                  test_acc_lst.append(test_acc) 
                  test_rms_lst.append(test_rms) 
                  test_roc_lst.append(test_roc)
                  loss_lst.append(float(loss))

            # save means metrices for val
            Results[dataset_name+'_acc'][0].append(np.mean(val_acc_lst))
            Results[dataset_name+'_rms'][0].append(np.mean(val_rms_lst))
            Results[dataset_name+'_roc'][0].append(np.mean(val_roc_lst))
            # save means metrices for test
            Results[dataset_name+'_acc'][1].append(np.mean(test_acc_lst))
            Results[dataset_name+'_rms'][1].append(np.mean(test_rms_lst))
            Results[dataset_name+'_roc'][1].append(np.mean(test_roc_lst))
            # save means loss for train
            Results[dataset_name+'_loss'].append(np.mean(loss_lst))
            
            # Print 
            condition = ((show) & (i % 2 == 0)) if (len(params) < 10) else ((show) & (i % 5 == 0))
            precent = int(i/len(params)*100)
            if condition:
                if params_tag == 'No. Training Nodes': 
                    print(f'{precent}%: A,B= ({A},{B}), Loss: {np.mean(loss_lst):.4f}')            
                elif params_tag == 'seed': 
                    print(f'{precent}%: seed= {seed}, Loss: {np.mean(loss_lst):.4f}')            
                elif params_tag == 'mu': 
                    print(f'{precent}%: mu= {mu:.4f}, Loss: {np.mean(loss_lst):.4f}')
                print(f'    Val_acc : {np.mean(val_acc_lst):.4f}, Val_rms : {np.mean(val_rms_lst):.4f}, Val_roc : {np.mean(val_roc_lst):.4f}')
                print(f'    Test_acc: {np.mean(test_acc_lst):.4f}, Test_rms: {np.mean(test_rms_lst):.4f}, Test_roc: {np.mean(test_roc_lst):.4f}')

    return Results, params, params_tag, reg_loss


# ==========================================
# Report functions
# ==========================================

# viz report:  metrices vs params for the 3 datasets on test
def report_vis(Results, params, params_tag, reg_loss):
    # Auxiliary function for ordring list w.r.t another list
    def aux_sort(lst,params):
          return [i for _,i in sorted(zip(params,lst))]
    # Set figure params
    datasets  = ['Cora', 'CiteSeer', 'PubMed']
    colors    = ['green', 'blue', 'red']
    tags      = ['acc', 'rms', 'roc']
    titles    = ['Accuracy', 'RMS', 'ROC-AUC']
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    plt.suptitle(f"{params_tag.capitalize()} Vs. Evaluation Metrics",fontsize=18)
    for i_tag,tag in enumerate(tags):    
        for i_data,dataset_name in enumerate(datasets):
          axs[i_tag].plot(sorted(params),aux_sort(Results[f'{dataset_name}_{tag}'][1],params), color=colors[i_data], marker='o', label=f'{dataset_name}')
          axs[i_tag].legend()
          axs[i_tag].set_title(f'{titles[i_tag]}')
          axs[i_tag].set_xlabel(params_tag)
          axs[i_tag].set_xticks(sorted(params))
          axs[i_tag].set_xticklabels(sorted(np.round(params,3) if (type(params)==np.ndarray) else params), rotation='vertical')
          axs[i_tag].set_ylabel(f'{tags[i_tag]}')
          axs[i_tag].grid(True)
    
    caption =  f"The figure shows: {params_tag.capitalize()} Vs. Accuracy, RMS and ROC-AUC for the 3 datasets.\n"+\
                "The model is evaluated with "+ reg_loss +" on test data." 
    plt.figtext(0.5, -0.2, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

# stats report: best params for the 3 datasets on test
def report_stats(Results, params, params_tag):
    # Set stats params
    datasets  = ['Cora', 'CiteSeer', 'PubMed']
    tags      = ['acc', 'rms', 'roc']
    titles    = ['Accuracy', 'RMS', 'ROC-AUC']
    for dataset_name in datasets:
        print(f'========= {dataset_name}:')
        for tag in tags:
            if tag != 'rms':
                best = params[np.argmax(Results[f'{dataset_name}_{tag}'][1])]
                best = np.round(best, 3) if ((type(best) == np.float64) or (type(best) == float)) else best
                print(f"    {tag} best {params_tag} on test: {best}")
            else:
                best = params[np.argmin(Results[f'{dataset_name}_{tag}'][1])]
                best = np.round(best, 3) if ((type(best) == np.float64) or (type(best) == float)) else best
                print(f"    {tag} best {params_tag} on test: {best}")