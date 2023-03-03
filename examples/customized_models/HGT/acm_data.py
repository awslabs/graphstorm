""" Prepare ACM data for node classification task in GSF
"""

import argparse
import scipy.io
import urllib.request
import os
import dgl
import json
import pickle
import numpy as np
import torch as th
import torch.nn as nn
from dgl.data.utils import save_graphs


def create_acm_graph(dowload_path='/tmp/ACM.mat', output_path=None):
    """Create ACM graph data from URL downloading.
    1. Assign paper nodes with a random 256D feature;
    2. No edge features
    """
    if not os.path.exists(dowload_path):
        data_url = 'https://data.dgl.ai/dataset/ACM.mat'
        urllib.request.urlretrieve(data_url, dowload_path)

    data = scipy.io.loadmat(dowload_path)
    graph_acm = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
    })

    pvc = data['PvsC'].tocsr()
    p_selected = pvc.tocoo()
    # generate labels
    labels = pvc.indices
    labels = th.tensor(labels).long()
    graph_acm.nodes['paper'].data['label'] = labels

    # generate train/val/test split and assign them to the paper nodes
    pid = p_selected.row
    shuffle = np.random.permutation(pid)
    train_idx = th.tensor(shuffle[0:800]).long()
    val_idx = th.tensor(shuffle[800:900]).long()
    test_idx = th.tensor(shuffle[900:]).long()
    train_mask = th.zeros(pid.shape[0]).long()
    train_mask[train_idx] = 1
    val_mask = th.zeros(pid.shape[0]).long()
    val_mask[val_idx] = 1
    test_mask = th.zeros(pid.shape[0]).long()
    test_mask[test_idx] = 1
    graph_acm.nodes['paper'].data['train_mask'] = train_mask
    graph_acm.nodes['paper'].data['val_mask'] = val_mask
    graph_acm.nodes['paper'].data['test_mask'] = test_mask

    # Give all nodes a 256D random values as their feature.
    for n_type in graph_acm.ntypes:
        emb = nn.Parameter(th.Tensor(graph_acm.number_of_nodes(n_type), 256), requires_grad = False)
        nn.init.xavier_uniform_(emb)
        graph_acm.nodes[n_type].data['feat'] = emb

    print(graph_acm)
    print(f'\n Number of classes: {labels.max() + 1}')
    print(f'\n Paper nodes labels: {labels.shape}')
    
    # Save the graph for later partition
    output_file_path = os.path.join(output_path, 'acm.dgl')
    print(f'Saving ACM data to {output_file_path} ......')
    save_graphs(output_file_path, [graph_acm], None)
    print(f'{output_file_path} saved.')
    
    return graph_acm


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Prepare ACM data for using GraphStorm")
    
    parser.add_argument('--download-path', type=str, default='/tmp/ACM.mat',
                        help="The path of folder to store downloaded ACM raw data")
    parser.add_argument('--output-path', type=str, required=True,
                        help="The path of folder to store processed ACM data in the JSON format")

    args = parser.parse_args()
        
    graph = create_acm_graph(dowload_path=args.download_path, 
                             output_path=args.output_path)
    