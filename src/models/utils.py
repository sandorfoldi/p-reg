import torch
import random

from tqdm import tqdm


def random_splits(data, A, B):
    class_names = torch.unique(data.y)
    class_masks = [(data.y == classname).nonzero(as_tuple=False).numpy().reshape(-1).tolist() for classname in class_names]

    train_indeces = []
    valid_indeces = []
    test_indeces = []
    #print(class_masks[0].shape)

    ind = 0
    
    for class_mask in class_masks:
        class_mask = set(class_mask)
        add_to_train = set(random.sample(class_mask, k=A))
        class_mask -= add_to_train
        add_to_valid = set(random.sample(class_mask, k=B))
        class_mask -= add_to_valid
        add_to_test = class_mask
        train_indeces += add_to_train
        valid_indeces += add_to_valid
        test_indeces += add_to_test
        ind += 1

    train_mask = torch.zeros(len(data.y), dtype=torch.bool)
    for i in train_indeces:
        train_mask[i] = True

    valid_mask = torch.zeros(len(data.y), dtype=torch.bool)
    for i in valid_indeces:
        valid_mask[i] = True

    test_mask = torch.zeros(len(data.y), dtype=torch.bool)
    for i in test_indeces:
        test_mask[i] = True
    
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask
    
    return data



        

    '''embedding = embeddings[i]
    neighbour_idxs = []
    # this is extremely inefficient
    for connection in data.edge_index.T:
        if connection[0] == i:
            neighbour_idxs.append(connection[1])'''
    pass


def propagate(data, embeddings):
    propagated = torch.zeros_like(embeddings)
    for i in tqdm(range(data.x.shape[0])):
        embedding = embeddings[i]
        neighbour_idxs = []
        # this is extremely inefficient
        for connection in data.edge_index.T:
            if connection[0] == i:
                neighbour_idxs.append(connection[1])
        propagated[i] = sum(list(map(lambda l: embeddings[l], neighbour_idxs))) / len(neighbour_idxs)
    
    return propagated
