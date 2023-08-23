import pandas as pd
import numpy as np
import pyarrow.parquet as pq 
import pyarrow as pa

import torch
import dgl
from dgl import AddReverse, Compose, ToSimple

def create_dgl_graph():
    # load dataset
    data = 'MAG'
    df = pd.read_csv('./DATA/{}/edges.csv'.format(data))
    num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    print('num_nodes: ', num_nodes)
    
    # Extract graph structure 
    # We only consider edge types from 100 to 109
    timestamp = df['time'].values
    mask = np.logical_and(timestamp >= 100, timestamp < 110)
    
    timestamp = timestamp[mask]
    src_nodes = df['src'].values[mask]
    dst_nodes = df['dst'].values[mask]
    
    # get the new node mappings
    unique_nodes = np.unique(
        np.concatenate([src_nodes, dst_nodes])
    )
    
    remain_nodes = len(unique_nodes)
    old_to_new_nodes = {}
    for i, j in enumerate(unique_nodes):
        old_to_new_nodes[j] = i
    
    new_src_nodes = np.array([
        old_to_new_nodes[i] for i in src_nodes
    ])
    
    new_dst_nodes = np.array([
        old_to_new_nodes[i] for i in dst_nodes
    ])
    
    # construct dgl graph
    
    etypes = [
        ('paper', 'cite_00', 'paper'),
        ('paper', 'cite_01', 'paper'),
        ('paper', 'cite_02', 'paper'),
        ('paper', 'cite_03', 'paper'),
        ('paper', 'cite_04', 'paper'),
        ('paper', 'cite_05', 'paper'),
        ('paper', 'cite_06', 'paper'),
        ('paper', 'cite_07', 'paper'),
        ('paper', 'cite_08', 'paper'),
        ('paper', 'cite_09', 'paper')
    ]
    
    import dgl
    graph_data = {}
    for etype in etypes:
        t = int(etype[1].split('_')[1])
        edge_mask = timestamp == (100 + t)
        
        graph_data[etype] = (
            new_src_nodes[edge_mask], 
            new_dst_nodes[edge_mask]
        )
    
    g = dgl.heterograph(graph_data)
    transform = Compose([ToSimple(), AddReverse()])
    g = transform(g)
    
    # add node features to dgl graph
    labels_df = pd.read_csv('./DATA/{}/labels.csv'.format(data))
    label_timestamp = labels_df['time'].values 
    
    mask = np.logical_and(label_timestamp >= 100, label_timestamp < 110)
    label_timestamp = label_timestamp[mask]
    node = labels_df['node'].values[mask]
    label = labels_df['label'].values[mask] - 1 # label [0, 1, ..., 151]
    
    # nodes not in train val test set has label -1
    new_label = - np.ones(remain_nodes)
    train_val_test_nodes = []
    
    for i, j in zip(node, label):
        if i in old_to_new_nodes:
            i = old_to_new_nodes[i]
            new_label[i] = j
            train_val_test_nodes.append(i)
    
    # generate data split 80% train 10% valid 10 test
    perm = np.random.permutation(len(train_val_test_nodes))
    train_val_test_nodes = np.array(train_val_test_nodes)[perm]
    train_ids = train_val_test_nodes[: int(0.8 * len(train_val_test_nodes))]
    valid_ids = train_val_test_nodes[int(0.8 * len(train_val_test_nodes)): int(0.9 * len(train_val_test_nodes))]
    test_ids  = train_val_test_nodes[int(0.9 * len(train_val_test_nodes)): ]
    
    train_mask = np.zeros(remain_nodes, np.uint8)
    train_mask[train_ids] = True
    
    valid_mask = np.zeros(remain_nodes, np.uint8)
    valid_mask[valid_ids] = True
    
    test_mask = np.zeros(remain_nodes, np.uint8)
    test_mask[test_ids] = True
    
    # node features
    node_feats = torch.load('./DATA/{}/node_features.pt'.format(data))
    g.nodes['paper'].data['label'] = torch.from_numpy(new_label).int()
    g.nodes['paper'].data['train_mask'] = torch.from_numpy(train_mask).to(torch.uint8)
    g.nodes['paper'].data['val_mask'] = torch.from_numpy(valid_mask).to(torch.uint8)
    g.nodes['paper'].data['test_mask'] = torch.from_numpy(test_mask).to(torch.uint8)
    g.nodes['paper'].data['feat'] = node_feats[unique_nodes].float()
    
    return g


g = create_dgl_graph()
print(g)
print(g.is_homogeneous)
dgl.distributed.partition_graph(g, graph_name='MAG_Temporal', num_parts=1, out_path='./DATA/')