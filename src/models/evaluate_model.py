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
    tp_valid = (pred[data.valid_mask] == data.y[data.valid_mask]).sum()
    tp_test = (pred[data.test_mask] == data.y[data.test_mask]).sum()

    acc_train = int(tp_train) / int(data.train_mask.sum())
    acc_valid = int(tp_valid) / int(data.valid_mask.sum())
    acc_test = int(tp_test) / int(data.test_mask.sum())

    return acc_train, acc_valid, acc_test


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