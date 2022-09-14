"""
Create dummy datasets for unit tests
"""

import os
import tempfile
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

    hetero_graph.edges['r0'].data['feat'] = edge_feat['r0']
    hetero_graph.edges['r1'].data['feat'] = edge_feat['r1']

    # set train/val/test masks for nodes and edges
    target_ntype = ['n0']
    target_etype = [("n0", "r1", "n1")]

    node_train_mask = generate_mask([0,1], data_size)
    node_val_mask = generate_mask([2,3], data_size)
    node_test_mask = generate_mask([4,5], data_size)

    edge_train_mask = generate_mask([0,1], 2 * data_size)
    edge_val_mask = generate_mask([2,3], 2 * data_size)
    edge_test_mask = generate_mask([4,5], 2 * data_size)

    hetero_graph.nodes[target_ntype[0]].data['train_mask'] = node_train_mask
    hetero_graph.nodes[target_ntype[0]].data['val_mask'] = node_val_mask
    hetero_graph.nodes[target_ntype[0]].data['test_mask'] = node_test_mask

    hetero_graph.edges[target_etype[0]].data['train_mask'] = edge_train_mask
    hetero_graph.edges[target_etype[0]].data['val_mask'] = edge_val_mask
    hetero_graph.edges[target_etype[0]].data['test_mask'] = edge_test_mask

    return hetero_graph


def partion_and_load_distributed_graph(hetero_graph=None, graph_name='dummy'):
    """
    Partition a heterogeneous graph into a temporal directory, and reload it as a distributed graph
    Parameters
    ----------
    hetero_graph: a DGL heterogeneous graph
    graph_name: string as a name

    Returns
    -------
    dist_graph: a DGL distributed graph
    """

    if not isinstance(hetero_graph, dgl.DGLGraph):
        raise Exception('Must have a valid DGL heterogeneous graph')

    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f'Create a temporary folder \'{tmpdirname}\' for output of distributed graph data')
        dist.partition_graph(hetero_graph, graph_name=graph_name, num_parts=1,
                             out_path=tmpdirname, part_method='metis')

        dist.initialize('')
        dist_graph = dist.DistGraph(graph_name=graph_name,
                                    part_config=os.path.join(tmpdirname, graph_name+'.json'))

    return dist_graph


def generate_dummy_dist_graph(size='tiny'):
    """
    Generate a dummy DGL distributed graph with the given size
    Parameters
    ----------
    size: the size of dummy graph data, could be one of tiny, small, medium, large, and largest

    Returns
    -------

    """
    hetero_graph = generate_dummy_hetero_graph(size=size)
    return partion_and_load_distributed_graph(hetero_graph=hetero_graph)


""" For self tests"""
if __name__ == '__main__':
    dist_graph = generate_dummy_dist_graph('small')
    print(dist_graph.num_edges('r1'))
