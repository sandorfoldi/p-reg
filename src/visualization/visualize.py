import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable


def gen_fig(filename, x, ys, xlabel, ylabels, ycolors, dpi=300):
    fig, ax = plt.subplots()
    for y, ylabel in zip(ys, ylabels):
        ax.plot(x, y, xlabel=xlabel, ylabel=ylabel)
    plt.legend()
    plt.save_fig(os.path.join('reports', filename), dpi=dpi)


def viz_swipe_preg_on_mu_icd():
    csv_path = 'reports/swipe_preg_on_mu.csv'
    df = pd.read_csv(csv_path)
    seeds = df['seed'].unique()

    arr = np.array([df[df['seed'] == seed]['icd0'] for seed in seeds])
    icd_means = arr.mean(axis=0)
    icd_vars = arr.var(axis=0)

    mus = df[df['seed']==seeds[0]]['mu']

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # ax.set_title('Intra class distance')
    ax.set(
        title='Intra class distance',
        xlabel='Regularization factor ($\mu$)',
        ylabel='Intra class distance ($\omega$)',
        xlim=(0, 2),
        # ylim=(0.26, 0.48),
        xticks=np.linspace(0, 2, num=11),
        # yticks=np.linspace(0.26, 0.48, 12)
        )
    ax.grid()
    plt.tight_layout()

    plt.errorbar(mus, icd_means, yerr=3*icd_vars**.5, elinewidth=2, capsize=4, capthick=2)
    plt.savefig('reports/swipe_preg_on_mu_icd.png', dpi=300)


def viz_swipe_preg_on_mu_acc():
    csv_path = 'reports/swipe_preg_on_mu.csv'

    df = pd.read_csv(csv_path)
    seeds = df['seed'].unique()

    arr_acc_train = np.array([df[df['seed'] == seed]['train_acc'] for seed in seeds])
    acc_train_means = arr_acc_train.mean(axis=0)
    acc_train_vars = arr_acc_train.var(axis=0)

    arr_acc_val = np.array([df[df['seed'] == seed]['val_acc'] for seed in seeds])
    acc_val_means = arr_acc_val.mean(axis=0)
    acc_val_vars = arr_acc_val.var(axis=0)

    arr_acc_test = np.array([df[df['seed'] == seed]['test_acc'] for seed in seeds])
    acc_test_means = arr_acc_test.mean(axis=0)
    acc_test_vars = arr_acc_test.var(axis=0)


    mus = df[df['seed']==seeds[0]]['mu']

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    # ax.set_title('Intra class distance')
    ax.set(
        title='Accuracy',
        xlabel='Regularization factor ($\mu$)',
        ylabel='Accuracy',
        xlim=(0, 2),
        ylim=(0., 1.2),
        # xticks=np.linspace(0, 2, num=11),
        # yticks=np.linspace(0.26, 0.48, 12)
        )
    ax.grid()
    plt.tight_layout()

    plt.errorbar(mus, acc_train_means, yerr=3*acc_train_vars**.5, elinewidth=2, capsize=4, capthick=2, label='Train')

    plt.errorbar(mus, acc_val_means, yerr=3*acc_val_vars**.5, elinewidth=2, capsize=4, capthick=2, label='Validation')

    plt.errorbar(mus, acc_test_means, yerr=3*acc_test_vars**.5, elinewidth=2, capsize=4, capthick=2, label='Test')

    plt.legend(loc='lower left')

    plt.savefig('reports/swipe_preg_on_mu_acc.png', dpi=300)



def viz_swipe_preg_on_train_size():
    csv_path = 'reports/swipe_preg_on_train_size.csv'
    df = pd.read_csv(csv_path)
    seeds = df['seed'].unique()

    train_sizes = df[(df['seed']==seeds[0]) & (df['model'] == 'gcn')]['num_training_nodes']

    arr_gcn = np.array([df[(df['seed'] == seed) & (df['model'] == 'gcn')]['test_acc'] for seed in seeds]) 
    acc_gcn_means = arr_gcn.mean(axis=0)
    acc_gcn_vars = arr_gcn.var(axis=0)

    arr_gat = np.array([df[(df['seed'] == seed) & (df['model'] == 'gat')]['test_acc'] for seed in seeds])
    acc_gat_means = arr_gat.mean(axis=0)
    acc_gat_vars = arr_gat.var(axis=0)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    plt.errorbar(train_sizes, acc_gcn_means, yerr=3*acc_gcn_vars**.5, elinewidth=2, capsize=4, capthick=2, label='GCN')

    plt.errorbar(train_sizes, acc_gat_means, yerr=3*acc_gat_vars**.5, elinewidth=2, capsize=4, capthick=2, label='GAT')


    ax.set(
        title='Relation between training nodes and accuracy',
        xlim=(0,100),
        ylim=(0, 1), 
        xlabel='No. training nodes', 
        ylabel='Test accuracy',
        xticks=np.linspace(0, 100, 11),
        yticks=np.linspace(0, 1, 6),
        )
    ax.grid()
    ax.legend(loc='lower left')
    plt.tight_layout()

    plt.savefig('reports/swipe_preg_on_train_size.png', dpi=300)


def viz_swipe_preg_on_unmask_preg_ratio():
    csv_path = 'reports/swipe_preg_on_unmask_preg_ratio.csv'
    df = pd.read_csv(csv_path)
    seeds = df['seed'].unique()

    unmask_preg_ratios = df[(df['seed']==seeds[0]) & (df['model'] == 'gcn')]['unmask preg ratio']

    arr_gcn = np.array([df[(df['seed'] == seed) & (df['model'] == 'gcn')]['test_acc'] for seed in seeds]) 
    acc_gcn_means = arr_gcn.mean(axis=0)
    acc_gcn_vars = arr_gcn.var(axis=0)

    arr_gat = np.array([df[(df['seed'] == seed) & (df['model'] == 'gat')]['test_acc'] for seed in seeds])
    acc_gat_means = arr_gat.mean(axis=0)
    acc_gat_vars = arr_gat.var(axis=0)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    plt.errorbar(unmask_preg_ratios, acc_gcn_means, yerr=3*acc_gcn_vars**.5, elinewidth=2, capsize=4, capthick=2, label='GCN')

    plt.errorbar(unmask_preg_ratios, acc_gat_means, yerr=3*acc_gat_vars**.5, elinewidth=2, capsize=4, capthick=2, label='GAT')

    ax.set(
        title='Relation between number of nodes used for regularization and accuracy',
        xlim=(0,1),
        # ylim=(0, 1), 
        xlabel='No. training nodes', 
        ylabel='Test accuracy',
        xticks=np.linspace(0, 1, 11),
        # yticks=np.linspace(0, 1, 6),
        )
    ax.grid()
    ax.legend(loc='lower left')
    plt.tight_layout()

    plt.savefig('reports/swipe_preg_on_unmask_preg_ratio.png', dpi=300)



def viz_swipe_preg_and_lapreg_on_mu():
    csv_path = 'reports/swipe_preg_and_lapreg_on_mu.csv'
    df = pd.read_csv(csv_path)
    seeds = df['seed'].unique()

    mus = df[(df['seed']==seeds[0]) & (df['model'] == 'gcn') & (df['reg'] == 'preg_loss')]['mu']

    arr_gcn = np.array([df[(df['seed'] == seed) & (df['model'] == 'gcn') & (df['reg'] == 'preg_loss')]['test_acc'] for seed in seeds]) 
    acc_gcn_means = arr_gcn.mean(axis=0)
    acc_gcn_vars = arr_gcn.var(axis=0)

    arr_gat = np.array([df[(df['seed'] == seed) & (df['model'] == 'gat') & (df['reg'] == 'preg_loss')]['test_acc'] for seed in seeds])
    acc_gat_means = arr_gat.mean(axis=0)
    acc_gat_vars = arr_gat.var(axis=0)

    arr_gcn_lap = np.array([df[(df['seed'] == seed) & (df['model'] == 'gcn') & (df['reg'] == 'lap_loss')]['test_acc'] for seed in seeds]) 
    acc_gcn_lap_means = arr_gcn_lap.mean(axis=0)
    acc_gcn_lap_vars = arr_gcn_lap.var(axis=0)

    arr_gat_lap = np.array([df[(df['seed'] == seed) & (df['model'] == 'gat') & (df['reg'] == 'lap_loss')]['test_acc'] for seed in seeds])
    acc_gat_lap_means = arr_gat_lap.mean(axis=0)
    acc_gat_lap_vars = arr_gat_lap.var(axis=0)
    
    
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    plt.errorbar(mus, acc_gcn_means, fmt='r', yerr=3*acc_gcn_vars**.5, elinewidth=2, capsize=4, capthick=2, label='GCN - PREG')

    plt.errorbar(mus, acc_gat_means, fmt='b', yerr=3*acc_gat_vars**.5, elinewidth=2, capsize=4, capthick=2, label='GAT - PREG')

    plt.errorbar(mus, acc_gcn_lap_means, fmt='--r', yerr=3*acc_gat_vars**.5, elinewidth=2, capsize=4, capthick=2, label='GCN - LAPLOSS')

    plt.errorbar(mus, acc_gat_lap_means, fmt='--b', yerr=3*acc_gat_vars**.5, elinewidth=2, capsize=4, capthick=2, label='GAT - LAPLOSS')

    """ax.set(
        title='Relation between number of nodes used for regularization and accuracy',
        xlim=(0,1),
        # ylim=(0, 1), 
        xlabel='No. training nodes', 
        ylabel='Test accuracy',
        xticks=np.linspace(0, 1, 11),
        # yticks=np.linspace(0, 1, 6),
        )"""
    ax.grid()
    ax.legend(loc='lower left')
    plt.tight_layout()

    plt.savefig('reports/swipe_preg_and_lapreg_on_mu.png', dpi=300)


def _add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')



def viz_swipe_preg_on_mu_and_neurs_icd():
    df = pd.read_csv('reports/swipe_preg_on_mu_and_neurs.csv')
    # df['icd'] = df['icd'].apply(lambda l: float(l.split('(')[-1].split(')')[0]))
    print(df.head())
    # ridiculously slow implementation, but I don't want to figure this out now
    arr = np.zeros((df['hidden_channels'].unique().shape[0], df['mu'].unique().shape[0]))
    for ind_i, i in enumerate(df['hidden_channels'].unique()):
        for ind_j, j in enumerate(df['mu'].unique()):
            # print((df['hidden_channels'] == i) & (df['mu'] == j))
            # print(df[(df['hidden_channels'] == i) & (df['mu'] == j)]['test_acc'])
            # icd0 = list(map(lambda l: l.lstrip('[').rstrip(']').split(' '), df['icd0']))
            # icd0 = list(map(lambda l: float(l), icd0))

            arr[ind_i,ind_j] = df[(df['hidden_channels'] == i) & (df['mu'] == j)]['icd0'].mean()

    # arr = np.nan_to_num(arr)
    fig, ax = plt.subplots()
    mus = df[(df['seed'] == 0) & (df['hidden_channels'] == 1)]['mu']
    hidden_channels = df[(df['seed'] == 0) & (df['mu'] == 0)]
    im = ax.pcolormesh(list(range(0,21, 2)), [1, 2, 4, 8, 16, 32, 64, 128, 256], arr, )
    ax.set_yscale('log')
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_yticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256])
    _add_colorbar(im, fig, ax)

    ax.set(
        title='Effects of changing model complexity and the amont of applied regularization jointly',
        xlabel='Regularization factor', 
        ylabel='No. neurons in hidden layer')
    ax.legend()
    plt.savefig('reports/swipe_preg_on_mu_and_neurs_icd.png', dpi=300)


def viz_swipe_preg_on_mu_and_neurs_acc():
    df = pd.read_csv('reports/swipe_preg_on_mu_and_neurs.csv')
    # df['icd'] = df['icd'].apply(lambda l: float(l.split('(')[-1].split(')')[0]))
    print(df.head())
    # ridiculously slow implementation, but I don't want to figure this out now
    arr = np.zeros((df['hidden_channels'].unique().shape[0], df['mu'].unique().shape[0]))
    for ind_i, i in enumerate(df['hidden_channels'].unique()):
        for ind_j, j in enumerate(df['mu'].unique()):
            # print((df['hidden_channels'] == i) & (df['mu'] == j))
            # print(df[(df['hidden_channels'] == i) & (df['mu'] == j)]['test_acc'])
            arr[ind_i,ind_j] = df[(df['hidden_channels'] == i) & (df['mu'] == j)]['test_acc'].mean()

    # arr = np.nan_to_num(arr)
    fig, ax = plt.subplots()
    im = ax.pcolormesh(list(range(0,21, 2)), [1, 2, 4, 8, 16, 32, 64, 128, 256], arr, )
    ax.set_yscale('log')
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax.set_yticklabels([1, 2, 4, 8, 16, 32, 64, 128, 256])
    _add_colorbar(im, fig, ax)

    ax.set(
        title='Effects of changing model complexity and the amont of applied regularization jointly',
        xlabel='Regularization factor', 
        ylabel='No. neurons in hidden layer')
    ax.legend()
    plt.savefig('reports/swipe_preg_on_mu_and_neurs_acc.png', dpi=300)


if __name__ == '__main__':
    viz_swipe_preg_on_mu_icd()
    viz_swipe_preg_on_mu_acc()
    viz_swipe_preg_on_train_size()
    viz_swipe_preg_on_unmask_preg_ratio()
    viz_swipe_preg_on_mu_and_neurs_acc()
    viz_swipe_preg_on_mu_and_neurs_icd()
    viz_swipe_preg_and_lapreg_on_mu()