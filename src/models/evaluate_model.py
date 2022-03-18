def evaluate0(model, data):    
    model.eval()
    pred = model(data).argmax(dim=1)
    tp = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(tp) / int(data.test_mask.sum())

    return acc