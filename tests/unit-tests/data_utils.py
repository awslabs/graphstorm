"""
Create dummy datasets for unit tests
"""

import os
import dgl
import numpy as np
import torch as th
import dgl.distributed as dist


def generate_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    th_mask = th.tensor(mask, dtype=th.bool)
    return th_mask

def generate_dummy_hetero_graph(size='tiny'):
    """
    generate a dummy heterogeneous graph.
    Parameters
    ----------
    size: the size of dummy graph data, could be one of tiny, small, medium, large, and largest

    :return:
    hg: a heterogeneous graph.
    """
    size_dict = {
        'tiny': 1e+2,
        'small': 1e+4,
        'medium': 1e+6,
        'large': 1e+8,
        'largest': 1e+10
    }

    data_size = int(size_dict[size])

    num_nodes_dict = {
        "n0": data_size,
        "n1": data_size,
    }

    edges = {
        ("n0", "r0", "n1"): (th.randint(data_size, (data_size,)),
                             th.randint(data_size, (data_size,))),
        ("n0", "r1", "n1"): (th.randint(data_size, (2 * data_size,)),
                             th.randint(data_size, (2 * data_size,)))
    }

    hetero_graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

    # set node and edge features
    node_feat = {'n0': th.randn(data_size, 2),
                 'n1': th.randn(data_size, 2)}

    edge_feat = {'r0': th.randn(data_size, 2),
                 'r1': th.randn(2 * data_size, 2)}

    hetero_graph.nodes['n0'].data['feat'] = node_feat['n0']
    hetero_graph.nodes['n1'].data['feat'] = node_feat['n1']
    hetero_graph.nodes['n1'].data['label'] = th.randint(10, (hetero_graph.number_of_nodes('n1'), ))

    hetero_graph.edges['r0'].data['feat'] = edge_feat['r0']
    hetero_graph.edges['r1'].data['feat'] = edge_feat['r1']
    hetero_graph.edges['r1'].data['label'] = th.randint(10, (hetero_graph.number_of_edges('r1'), ))

    # set train/val/test masks for nodes and edges
    target_ntype = ['n1']
    target_etype = [("n0", "r1", "n1"), ("n0", "r0", "n1")]

    node_train_mask = generate_mask([0,1], data_size)
    node_val_mask = generate_mask([2,3], data_size)
    node_test_mask = generate_mask([4,5], data_size)

    edge_train_mask = generate_mask([0,1], 2 * data_size)
    edge_val_mask = generate_mask([2,3], 2 * data_size)
    edge_test_mask = generate_mask([4,5], 2 * data_size)

    edge_train_mask2 = generate_mask([i for i in range(data_size//2)], data_size)
    edge_val_mask2 = generate_mask([2,3], data_size)
    edge_test_mask2 = generate_mask([4,5], data_size)

    hetero_graph.nodes[target_ntype[0]].data['train_mask'] = node_train_mask
    hetero_graph.nodes[target_ntype[0]].data['val_mask'] = node_val_mask
    hetero_graph.nodes[target_ntype[0]].data['test_mask'] = node_test_mask

    hetero_graph.edges[target_etype[0]].data['train_mask'] = edge_train_mask
    hetero_graph.edges[target_etype[0]].data['val_mask'] = edge_val_mask
    hetero_graph.edges[target_etype[0]].data['test_mask'] = edge_test_mask

    hetero_graph.edges[target_etype[1]].data['train_mask'] = edge_train_mask2
    hetero_graph.edges[target_etype[1]].data['val_mask'] = edge_val_mask2
    hetero_graph.edges[target_etype[1]].data['test_mask'] = edge_test_mask2

    return hetero_graph


def partion_and_load_distributed_graph(hetero_graph, dirname, graph_name='dummy'):
    """
    Partition a heterogeneous graph into a temporal directory, and reload it as a distributed graph
    Parameters
    ----------
    hetero_graph: a DGL heterogeneous graph
    dirname : the directory where the graph will be partitioned and stored.
    graph_name: string as a name

    Returns
    -------
    dist_graph: a DGL distributed graph
    part_config : the path of the partition configuration file.
    """

    if not isinstance(hetero_graph, dgl.DGLGraph):
        raise Exception('Must have a valid DGL heterogeneous graph')

    print(f'Create a temporary folder \'{dirname}\' for output of distributed graph data')
    dist.partition_graph(hetero_graph, graph_name=graph_name, num_parts=1,
                         out_path=dirname, part_method='metis')

    dist.initialize('')
    part_config = os.path.join(dirname, graph_name+'.json')
    dist_graph = dist.DistGraph(graph_name=graph_name, part_config=part_config)

    return dist_graph, part_config


def generate_dummy_dist_graph(dirname, size='tiny', graph_name='dummy'):
    """
    Generate a dummy DGL distributed graph with the given size
    Parameters
    ----------
    dirname : the directory where the graph will be partitioned and stored.
    size: the size of dummy graph data, could be one of tiny, small, medium, large, and largest
    graph_name: string as a name

    Returns
    -------
    dist_graph: a DGL distributed graph
    part_config : the path of the partition configuration file.
    """
    hetero_graph = generate_dummy_hetero_graph(size=size)
    return partion_and_load_distributed_graph(hetero_graph=hetero_graph, dirname=dirname,
                                              graph_name=graph_name)


""" For self tests"""
if __name__ == '__main__':
    dist_graph = generate_dummy_dist_graph('small')
    print(dist_graph.num_edges('r1'))
