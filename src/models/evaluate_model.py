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
        for class_id in np.unique(data.y):
            Sk = (pred == class_id)
            ck = torch.nn.Softmax(dim=1)(out)[Sk].mean(dim=0)
            zi = torch.nn.Softmax(dim=1)(out)[Sk]

            omega = torch.linalg.norm((zi-ck), dim=1)**.5
            omega = omega.sum()
            w += omega

        icds.append((w/N).item())
    return icds


def icd_saf_2(model, data):
    """
    instead of softmax(out), simply z is used
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
        for class_id in np.unique(data.y):
            Sk = (pred == class_id)
            ck = Z[Sk].mean(dim=0)
            zi = Z[Sk]

            omega = torch.linalg.norm((zi-ck), dim=1)**.5
            omega = omega.sum()
            w += omega

        icds.append((w/N).item())
    return icds
    

def icd_saf_3(model, data):
    # icd computed over nodes that belong to specific classes
    # and not the ones that are predicted to belong to specific classes
    model.eval()
    icds = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        with torch.no_grad():
            Z = model(data)[mask]
        out = F.log_softmax(Z, dim=1)
        pred = out.argmax(dim=1)

        N = len(pred)

        w=0

        for class_id in np.unique(data.y):
            Sk = (data.y[mask] == class_id)
            ck = torch.nn.Softmax(dim=1)(out)[Sk].mean(dim=0)
            zi = torch.nn.Softmax(dim=1)(out)[Sk]

            omega = torch.linalg.norm((zi-ck), dim=1)**.5
            omega = omega.sum()
            w += omega

        icds.append((w/N).item())
    return icds


def icd_saf_4(model, data):
    # only taking softmax once
    model.eval()
    icds = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        with torch.no_grad():
            Z = model(data)[mask]

        pred = Z.argmax(dim=1)

        N = len(pred)

        w=0
        for class_id in np.unique(data.y):
            Sk = (pred == class_id)
            ck = torch.nn.Softmax(dim=1)(Z)[Sk].mean(dim=0)
            zi = torch.nn.Softmax(dim=1)(Z)[Sk]

            omega = torch.linalg.norm((zi-ck), dim=1)**.5
            omega = omega.sum()
            w += omega

        icds.append((w/N).item())
    return icds