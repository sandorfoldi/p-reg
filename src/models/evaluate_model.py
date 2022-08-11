import numpy as np
import torch

def evaluate0(model, data):    
    model.eval()
    pred = model(data).argmax(dim=1)
    tp = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(tp) / int(data.test_mask.sum())

    return acc


def evaluate1(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)

    tp_train = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    tp_val = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    tp_test = (pred[data.test_mask] == data.y[data.test_mask]).sum()

    acc_train = int(tp_train) / int(data.train_mask.sum())
    acc_val = int(tp_val) / int(data.val_mask.sum())
    acc_test = int(tp_test) / int(data.test_mask.sum())

    return acc_train, acc_val, acc_test


def evaluate1abdul(model, data):
    model.eval()
    pred = model(data)[0].argmax(dim=1)

    tp_train = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    tp_val = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    tp_test = (pred[data.test_mask] == data.y[data.test_mask]).sum()

    acc_train = int(tp_train) / int(data.train_mask.sum())
    acc_val = int(tp_val) / int(data.val_mask.sum())
    acc_test = int(tp_test) / int(data.test_mask.sum())

    return acc_train, acc_val, acc_test


def icd0(model, data):
    model.eval()
    out = model(data)[data.train_mask]
    pred = out.argmax(dim=1)

    N = len(pred)

    w=0
    class_ids = np.unique(pred)

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

    return w/N

def icd1(model, data):
    model.eval()
    with torch.no_grad():
        Z = model(data)
        icds = [] # intra class distances
        for c in data.y.unique().numpy():
            s_k = Z[data.y == c]
            icds.append(s_k.std()**2)
        
        return np.array(icds).mean()


def icd2(model, data):
    model.eval()
    with torch.no_grad():
        Z = model(data)
        icds = [] # intra class distances
        for c in data.y.unique().numpy():
            s_k = torch.nn.Softmax(dim=0)(Z[data.y == c])
            icds.append(s_k.std()**2)
        
        return np.array(icds).mean()


def icd3(model, data):
    model.eval()
    with torch.no_grad():
        Z = model(data).numpy()
        icds = []
        for c in data.y.unique().numpy():
            s_k = Z[data.y == c]
            icds.append((((s_k.mean() - s_k)**2)**.5).mean())
        
        return np.array(icds).mean()