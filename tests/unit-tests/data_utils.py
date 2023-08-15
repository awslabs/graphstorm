"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Create dummy datasets for unit tests
"""

import os
import dgl
import numpy as np
import torch as th
import dgl.distributed as dist
import tempfile

from transformers import AutoTokenizer
from graphstorm import get_feat_size
from graphstorm.model.lm_model import TOKEN_IDX, ATT_MASK_IDX, VALID_LEN
from util import create_tokens


def generate_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    th_mask = th.tensor(mask, dtype=th.bool)
    return th_mask

def generate_dummy_hetero_graph(size='tiny', gen_mask=True):
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
    if gen_mask:
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


def generate_dummy_homo_graph(size='tiny', gen_mask=True):
    """
    generate a dummy homogeneous graph.
    Parameters
    ----------
    size: the size of dummy graph data, could be one of tiny, small, medium, large, and largest

    :return:
    hg: a homogeneous graph in one node type and one edge type.
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
        "_N": data_size,
    }

    edges = {
        ("_N", "_E", "_N"): (th.randint(data_size, (2 * data_size,)),
                             th.randint(data_size, (2 * data_size,)))
    }

    hetero_graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

    # set node and edge features
    node_feat = {'_N': th.randn(data_size, 2)}

    edge_feat = {'_E': th.randn(2 * data_size, 2)}

    hetero_graph.nodes['_N'].data['feat'] = node_feat['_N']
    hetero_graph.nodes['_N'].data['label'] = th.randint(10, (hetero_graph.number_of_nodes('_N'), ))

    hetero_graph.edges['_E'].data['feat'] = edge_feat['_E']
    hetero_graph.edges['_E'].data['label'] = th.randint(10, (hetero_graph.number_of_edges('_E'), ))

    # set train/val/test masks for nodes and edges
    if gen_mask:
        target_ntype = ['_N']
        target_etype = [("_N", "_E", "_N")]

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


def generate_dummy_dist_graph(dirname, size='tiny', graph_name='dummy', gen_mask=True, is_homo=False):
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
    if not is_homo:
        hetero_graph = generate_dummy_hetero_graph(size=size, gen_mask=gen_mask)
    else:
        hetero_graph = generate_dummy_homo_graph(size=size, gen_mask=gen_mask)
    return partion_and_load_distributed_graph(hetero_graph=hetero_graph, dirname=dirname,
                                              graph_name=graph_name)

def create_lm_graph(tmpdirname):
    """ Create a graph with textual feaures
        Only n0 has a textual feature.
        n1 does not have textual feature.
    """
    bert_model_name = "bert-base-uncased"
    max_seq_length = 8
    lm_config = [{"lm_type": "bert",
                  "model_name": bert_model_name,
                  "gradient_checkpoint": True,
                  "node_types": ["n0"]}]
    # get the test dummy distributed graph
    g, part_config = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_feat_size(g, {'n0' : ['feat']})
    input_text = ["Hello world!"]
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    input_ids, valid_len, attention_mask, _ = \
        create_tokens(tokenizer=tokenizer,
                      input_text=input_text,
                      max_seq_length=max_seq_length,
                      num_node=g.number_of_nodes('n0'))

    g.nodes['n0'].data[TOKEN_IDX] = input_ids
    g.nodes['n0'].data[VALID_LEN] = valid_len

    return lm_config, feat_size, input_ids, attention_mask, g, part_config

def create_lm_graph2(tmpdirname):
    """ Create a graph with textual feaures
        Both n0 and n1 have textual features.
    """
    bert_model_name = "bert-base-uncased"
    max_seq_length = 8
    lm_config = [{"lm_type": "bert",
                  "model_name": bert_model_name,
                  "gradient_checkpoint": True,
                  "attention_probs_dropout_prob": 0,
                  "hidden_dropout_prob":0,
                  "node_types": ["n0", "n1"]}]

    # get the test dummy distributed graph
    g, create_lm_graph = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_feat_size(g, {'n0' : ['feat']})
    input_text = ["Hello world!"]
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    input_ids0, valid_len0, attention_mask0, _ = \
        create_tokens(tokenizer=tokenizer,
                      input_text=input_text,
                      max_seq_length=max_seq_length,
                      num_node=g.number_of_nodes('n0'))

    g.nodes['n0'].data[TOKEN_IDX] = input_ids0
    g.nodes['n0'].data[ATT_MASK_IDX] = valid_len0

    input_text = ["Hello Aamzon!"]
    input_ids1, valid_len1, attention_mask1, _ = \
        create_tokens(tokenizer=tokenizer,
                      input_text=input_text,
                      max_seq_length=max_seq_length,
                      num_node=g.number_of_nodes('n1'))
    g.nodes['n1'].data[TOKEN_IDX] = input_ids1
    g.nodes['n1'].data[VALID_LEN] = valid_len1

    return lm_config, feat_size, input_ids0, attention_mask0, \
        input_ids1, attention_mask1, g, create_lm_graph

""" For self tests"""
if __name__ == '__main__':
    dist_graph = generate_dummy_dist_graph('small')
    print(dist_graph.num_edges('r1'))
