



# ========================================== 
# Dependencies
# ========================================== 

# pyg
from torch_geometric.utils import to_networkx

# vis
import networkx as nx
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from fa2 import ForceAtlas2

# other
import warnings
warnings.filterwarnings("ignore")






# ========================================== 
# Utils
# ==========================================
# graph viz   
def visualize_graph(G, color):
    """ Spring Layout Graph """
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, 
                    pos=nx.spring_layout(G, seed=42), 
                    with_labels=False,
                    node_color=color, 
                    cmap="Set2")
    plt.show()


# atals graph viz  ********
def visualize_Atlas(h, color):
    """ Atlas Graph """
    G = to_networkx(h, node_attrs=["y"], to_undirected=True)
    #get nodes positions based on fa2
    forceatlas2 = ForceAtlas2(
                            # Behavior alternatives
                            outboundAttractionDistribution=False,  # Dissuade hubs
                            edgeWeightInfluence=1.5,
                            # Performance
                            jitterTolerance=0.1, # Tolerance
                            barnesHutOptimize=True,
                            barnesHutTheta=1,
                            # Tuning
                            scalingRatio=1.,
                            strongGravityMode=False,
                            gravity=0.1,
                            # Log
                            verbose=False)
    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=200)
    #Set node size according to degree (here rescaled by a factor of 3 for better visualization)
    node_size = dict(G.degree())
    plt.figure( figsize=(10,10))
    options = {
        'node_size': [v*3  for v in node_size.values()] ,
        'linewidths': 1,
        'width': 0.5,
    }
    nx.draw_networkx(G, positions, **options, with_labels=False, node_color = color)
    plt.axis('off')
    #Add a caption
    txt="Figure 1.5: Visualization of the network with node positions inferred by Force Atlas 2."
    plt.figtext(0.5, +0.1, txt, wrap=True, horizontalalignment='center', fontsize=20)
    plt.show()


# embedding scatter viz (fast)
def visualize_scatter(h, color, epoch=None, loss=None):
    """ Scatter Embedding """
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


# embedding tsne viz (theory and reliable)  ********* 
def visualize_TSNE(h, color, epoch=None, loss=None):
    """ TSNE Embedding """
    transform = TSNE  # or PCA
    trans = transform(n_components=2, init='pca', learning_rate='auto')
    X_reduced = trans.fit_transform(h.detach().cpu().numpy())

    _, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=color,
        cmap="jet",
        alpha=0.7,
    )
    if epoch is not None and loss is not None:
        title=f"{transform.__name__} visualization of embeddings\nEpoch: {epoch}, Loss: {loss.item():.4f}"
    else:
        title=f"{transform.__name__} visualization of embeddings."
    ax.set(
        aspect="equal",
        xlabel="$X_1$",
        ylabel="$X_2$",
        title= title
    )
    plt.show()


# confusion matrix vis  **************
def visualize_CM(y_true, y_pred, mask, num_classes):
    """ Confusion Matrix for Predictions """
    cm = confusion_matrix(y_true = y_true[mask].to('cpu'), y_pred = y_pred[mask].to('cpu'))
    df_cm = pd.DataFrame(cm, range(num_classes), range(num_classes))
    plt.figure(figsize=(8,8))
    sn.heatmap(df_cm, annot=True, fmt='g', cbar=False)
    plt.title('Confusion matrix')
    plt.show()


# dataset stats print *************
def print_dataset(dataset):
    """ dataset overview """
    print(f'{"Dataset:":<27} {dataset}:')
    print(f'{"Number of graphs:":<27} {len(dataset)}')
    print(f'{"Number of features:":<27} {dataset.num_features}')
    print(f'{"Number of classes:":<27} {dataset.num_classes}')


# data stats print *************
def print_data(data):
    """ data overview """
    print('')
    print(data)
    print('')
    print(f'{"Number of nodes:":<27} {data.num_nodes}')
    print(f'{"Number of edges:":<27} {data.num_edges}')
    print(f'{"Average node degree:":<27} {data.num_edges / data.num_nodes:.2f}')
    print(f'{"Has isolated nodes:":<27} {data.has_isolated_nodes()}')
    print(f'{"Has self-loops:":<27} {data.has_self_loops()}')
    print(f'{"Is undirected:":<27} {data.is_undirected()}')
    print('')
    print(f'{"Number of training nodes:":<27} {data.train_mask.sum()}')
    print(f'{"Number of validation nodes:":<27} {data.val_mask.sum()}')
    print(f'{"Number of test nodes:":<27} {data.test_mask.sum()}')
    print('')
    print(f'{"Training node label rate:":<27} {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'{"Validation node label rate:":<27} {int(data.val_mask.sum()) / data.num_nodes:.2f}')
    print(f'{"Test node label rate:":<27} {int(data.test_mask.sum()) / data.num_nodes:.2f}')