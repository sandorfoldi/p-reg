def evaluate0(model, data):    
    model.eval()
    pred = model(data).argmax(dim=1)
    tp = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(tp) / int(data.test_mask.sum())

    return acc


def test(model, data, splits):
    train_mask = splits[0].to(data.x.device)
    val_mask = splits[1].to(data.x.device)
    test_mask = splits[2].to(data.x.device)
    model.eval()
    # final output
    logits, accs = model(data), []
    for mask in [train_mask, val_mask, test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs