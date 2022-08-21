import numpy as np
import torch

import torch.nn.functional as F


def acc(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)

    tp_train = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    tp_val = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    tp_test = (pred[data.test_mask] == data.y[data.test_mask]).sum()

    acc_train = int(tp_train) / int(data.train_mask.sum())
    acc_val = int(tp_val) / int(data.val_mask.sum())
    acc_test = int(tp_test) / int(data.test_mask.sum())

    return acc_train, acc_val, acc_test


def icd_apolline_0(model, data):
    model.eval()
    # out = model(data)[0][data.train_mask]
    Z = model(data)[data.test_mask]
    out = F.log_softmax(Z, dim=1)
    pred = out.argmax(dim=1)

    N = len(pred)

    w=0
    class_ids = np.unique(data.y)

    for class_id in class_ids:
        # Sk = nodes in class k
        Sk = (pred == class_id).nonzero(as_tuple=True)[0]

        # Compute ck
        ck = torch.Tensor(np.zeros(len(class_ids)))
        for i in Sk:
            #zi = torch.max(out[i]).item()
            zi = out[i].clone().detach()
            zi = torch.nn.Softmax(dim=0)(zi)
            ck += zi
        ck = ck/len(Sk)

        for i in Sk:
            #zi = torch.max(out[i]).item()
            zi = out[i].clone().detach()
            zi = torch.nn.Softmax(dim=0)(zi)
            w += torch.linalg.norm(zi-ck)
        
        #print(clk, ck, w)

    return (w/N).item()


def icd_apolline_1(model, data):
    model.eval()
    # out = model(data)[0][data.train_mask]
    Z = model(data)[data.test_mask]
    out = F.log_softmax(Z, dim=1)
    pred = out.argmax(dim=1)

    N = len(pred)

    w=0
    class_ids = np.unique(data.y)

    for class_id in class_ids:
        # Sk = nodes in class k
        Sk = (pred == class_id).nonzero(as_tuple=True)[0]

        # Compute ck
        ck = torch.Tensor(np.zeros(len(class_ids)))
        for i in Sk:
            #zi = torch.max(out[i]).item()
            zi = out[i].clone().detach()
            zi = torch.nn.Softmax(dim=0)(zi)
            ck += zi
        ck = ck/len(Sk)

        for i in Sk:
            #zi = torch.max(out[i]).item()
            zi = out[i].clone().detach()
            zi = torch.nn.Softmax(dim=0)(zi)
            w += torch.linalg.norm(zi-ck)**0.5
        
        #print(clk, ck, w)

    return (w/N).item()


def icd_apolline_2(model, data):
    """
    Same as icd_apolline_1 but for all train valid and test masks
    """
    model.eval()
    icds = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        with torch.no_grad():
            Z = model(data)[mask]
        out = F.log_softmax(Z, dim=1)
        pred = out.argmax(dim=1)

        N = len(pred)

        w=0
        class_ids = np.unique(data.y)
        for class_id in class_ids:
            
            # Sk is the list of indeces of nodes that belong to class k
            Sk = (pred == class_id).nonzero(as_tuple=True)[0]

            # compute ck
            ck = torch.Tensor(np.zeros(len(class_ids)))

            for i in Sk:
                zi = out[i]
                zi = torch.nn.Softmax(dim=0)(zi)
                ck += zi
            ck = ck/len(Sk)
            # print(f'apo2 -> ck = {ck}')

            # compute icd per class
            for i in Sk:
                zi = out[i]
                zi = torch.nn.Softmax(dim=0)(zi)
                omega = torch.linalg.norm(zi-ck)**0.5
                w += omega
        icds.append((w/N).item())
        
    return icds


def icd_apolline_3(model, data):
    """
    This implementation does not take the log softmax of Z when computing icds.
    """
    model.eval()
    icds = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        with torch.no_grad():
            Z = model(data)[mask]
        out = F.log_softmax(Z, dim=1)
        pred = out.argmax(dim=1)

        N = len(pred)

        w=0
        class_ids = np.unique(data.y)
        for class_id in class_ids:
            
            # Sk is the list of indeces of nodes that belong to class k
            Sk = (pred == class_id).nonzero(as_tuple=True)[0]

            # compute ck
            ck = torch.Tensor(np.zeros(len(class_ids)))

            for i in Sk:
                # zi = out[i]
                # zi = torch.nn.Softmax(dim=0)(zi)
                zi = Z[i]
                ck += zi
            ck = ck/len(Sk)

            # compute icd per class
            for i in Sk:
                # zi = out[i]
                # zi = torch.nn.Softmax(dim=0)(zi)
                zi = Z[i]
                w += torch.linalg.norm(zi-ck)**0.5
            
        icds.append((w/N).item())
        
    return icds




def icd_apolline_4(model, data):
    """
    This implementation computed icds using softmax(log_softmax(z))
    But on all nodes that belong to class k
    """
    model.eval()
    icds = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        with torch.no_grad():
            Z = model(data)[mask]
        out = F.log_softmax(Z, dim=1)
        pred = out.argmax(dim=1)

        N = len(pred)

        w=0
        class_ids = np.unique(data.y)
        for class_id in class_ids:
            
            # Sk is the list of indeces of nodes that belong to class k
            # Sk = (pred == class_id).nonzero(as_tuple=True)[0]
            Sk = (data.y == class_id).nonzero(as_tuple=True)[0]

            # compute ck
            ck = torch.Tensor(np.zeros(len(class_ids)))

            for i in Sk:
                zi = out[i]
                zi = torch.nn.Softmax(dim=0)(zi)
                ck += zi
            ck = ck/len(Sk)

            # compute icd per class
            for i in Sk:
                zi = out[i]
                zi = torch.nn.Softmax(dim=0)(zi)
                w += torch.linalg.norm(zi-ck)**0.5
            
        icds.append((w/N).item())
        
    return icds



def icd_apolline_5(model, data):
    """
    This implementation computes icds on z
    And on all nodes that belong to class k
    """
    model.eval()
    icds = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        with torch.no_grad():
            Z = model(data)[mask]
        out = F.log_softmax(Z, dim=1)
        pred = out.argmax(dim=1)

        N = len(pred)

        w=0
        class_ids = np.unique(data.y)
        for class_id in class_ids:
            
            # Sk is the list of indeces of nodes that belong to class k
            # Sk = (pred == class_id).nonzero(as_tuple=True)[0]
            Sk = (data.y == class_id).nonzero(as_tuple=True)[0]

            # compute ck
            ck = torch.Tensor(np.zeros(len(class_ids)))

            for i in Sk:
                # zi = out[i]
                # zi = torch.nn.Softmax(dim=0)(zi)
                zi = Z[i]

                ck += zi
            ck = ck/len(Sk)

            # compute icd per class
            for i in Sk:
                # zi = out[i]
                # zi = torch.nn.Softmax(dim=0)(zi)
                zi = Z[i]

                w += torch.linalg.norm(zi-ck)**0.5
            
        icds.append((w/N).item())
        
    return icds


def icd_saf_0(model, data):
    model.eval()
    with torch.no_grad():
        icds = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            Z = model(data)[mask]
            icd_per_class = [] # intra class distances
            for c in data.y.unique().numpy():
                # s_k = Z[data.y[mask] == c]
                s_k = torch.nn.Softmax(dim=0)(Z[data.y[mask] == c])
                icd_per_class.append(s_k.var())
            
            icds.append(np.array(icd_per_class).mean())
        return icds


def icd_saf_1(model, data):
    # different implementation of icd_apolline_2
    model.eval()
    icds = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        with torch.no_grad():
            Z = model(data)[mask]
        out = F.log_softmax(Z, dim=1)
        pred = out.argmax(dim=1)

        N = len(pred)

        w=0
        class_ids = np.unique(data.y)
        for class_id in class_ids:
            Sk = (pred == class_id)
            ck = torch.nn.Softmax(dim=1)(out)[Sk].mean(dim=0)
            zi = torch.nn.Softmax(dim=1)(out)[Sk]

            omega = torch.linalg.norm((zi-ck), dim=1)**.5
            omega = omega.sum()
            w += omega

        icds.append((w/N).item())
    return icds
    
