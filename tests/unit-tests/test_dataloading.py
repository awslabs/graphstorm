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

    Test functions and classes in the dataloading.py
"""
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock

import torch as th
import dgl
import pytest
from data_utils import generate_dummy_dist_graph, generate_dummy_dist_graph_reconstruct

import graphstorm as gs
from graphstorm.dataloading import GSgnnNodeTrainData, GSgnnNodeInferData
from graphstorm.dataloading import GSgnnEdgeTrainData, GSgnnEdgeInferData
from graphstorm.dataloading import GSgnnAllEtypeLinkPredictionDataLoader
from graphstorm.dataloading import GSgnnNodeDataLoader, GSgnnEdgeDataLoader
from graphstorm.dataloading import (GSgnnLinkPredictionDataLoader,
                                   FastGSgnnLinkPredictionDataLoader)
from graphstorm.dataloading import GSgnnLinkPredictionTestDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionJointTestDataLoader
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER

from graphstorm.dataloading.dataset import (prepare_batch_input,
                                            prepare_batch_edge_input)
from graphstorm.dataloading.utils import modify_fanout_for_target_etype
from graphstorm.dataloading.utils import trim_data

from numpy.testing import assert_equal

def get_nonzero(mask):
    mask = mask[0:len(mask)]
    return th.nonzero(mask, as_tuple=True)[0]

def test_GSgnnEdgeData_wo_test_mask():
    for file in os.listdir("/dev/shm/"):
        shutil.rmtree(file, ignore_errors=True)
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    va_etypes = [("n0", "r1", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        dist_graph, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                            dirname=os.path.join(tmpdirname, 'dummy'),
                                                            gen_mask=False)

        ev_data_nomask = GSgnnEdgeInferData(graph_name='dummy', part_config=part_config,
                                            eval_etypes=va_etypes)
        ev_data_nomask2 = GSgnnEdgeInferData(graph_name='dummy', part_config=part_config,
                                             eval_etypes=None)

    assert ev_data_nomask.eval_etypes == va_etypes
    assert ev_data_nomask2.eval_etypes == dist_graph.canonical_etypes

    # eval graph without test mask
    # all edges in the eval etype are treated as target edges
    assert len(ev_data_nomask.infer_idxs) == len(va_etypes)
    for etype in va_etypes:
        assert th.all(ev_data_nomask.infer_idxs[etype] == th.arange(dist_graph.num_edges(etype)))

    assert len(ev_data_nomask2.infer_idxs) == len(dist_graph.canonical_etypes)
    for etype in dist_graph.canonical_etypes:
        assert th.all(ev_data_nomask2.infer_idxs[etype] == th.arange(dist_graph.num_edges(etype)))

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_GSgnnNodeData_wo_test_mask():
    for file in os.listdir("/dev/shm/"):
        shutil.rmtree(file, ignore_errors=True)
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    va_ntypes = ["n1"]
    with tempfile.TemporaryDirectory() as tmpdirname:
        dist_graph, part_config = generate_dummy_dist_graph(graph_name='dummy2',
                                                            dirname=os.path.join(tmpdirname, 'dummy2'),
                                                            gen_mask=False)
        infer_data_nomask = GSgnnNodeInferData(graph_name='dummy2', part_config=part_config,
                                            eval_ntypes=va_ntypes)
    assert infer_data_nomask.eval_ntypes == va_ntypes
    # eval graph without test mask
    # all nodes in the eval ntype are treated as target nodes
    assert len(infer_data_nomask.infer_idxs) == len(va_ntypes)
    for ntype in va_ntypes:
        assert th.all(infer_data_nomask.infer_idxs[ntype] == th.arange(dist_graph.num_nodes(ntype)))

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_GSgnnEdgeData():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    tr_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    tr_single_etype = [("n0", "r1", "n1")]
    va_etypes = [("n0", "r1", "n1")]
    ts_etypes = [("n0", "r1", "n1")]

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        dist_graph, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                            dirname=os.path.join(tmpdirname, 'dummy'))
        tr_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=tr_etypes, eval_etypes=va_etypes,
                                     label_field='label')
        tr_data1 = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                      train_etypes=tr_etypes)
        # pass train etypes as None
        tr_data2 = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=None,
                                     label_field='label')
        # train etypes does not cover all etypes.
        tr_data3 = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=tr_single_etype,
                                     label_field='label')
        ev_data = GSgnnEdgeInferData(graph_name='dummy', part_config=part_config,
                                     eval_etypes=va_etypes)
        # pass eval etypes as None
        ev_data2 = GSgnnEdgeInferData(graph_name='dummy', part_config=part_config,
                                      eval_etypes=None)

    # successful initialization with default setting
    assert tr_data.train_etypes == tr_etypes
    assert tr_data.eval_etypes == va_etypes
    assert tr_data1.train_etypes == tr_etypes
    assert tr_data1.eval_etypes == tr_etypes
    assert ev_data.eval_etypes == va_etypes
    assert tr_data2.train_etypes == tr_data2.eval_etypes
    assert tr_data2.train_etypes == dist_graph.canonical_etypes
    assert tr_data3.train_etypes == tr_single_etype
    assert tr_data3.eval_etypes == tr_single_etype
    assert ev_data2.eval_etypes == dist_graph.canonical_etypes

    # sucessfully split train/val/test idxs
    assert len(tr_data.train_idxs) == len(tr_etypes)
    for etype in tr_etypes:
        assert th.all(tr_data.train_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))
    assert len(ev_data.train_idxs) == 0

    assert len(tr_data.val_idxs) == len(va_etypes)
    for etype in va_etypes:
        assert th.all(tr_data.val_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))
    assert len(tr_data1.val_idxs) == len(tr_etypes)
    for etype in tr_etypes:
        assert th.all(tr_data1.val_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))
    assert len(ev_data.val_idxs) == 0

    assert len(tr_data.test_idxs) == len(ts_etypes)
    for etype in ts_etypes:
        assert th.all(tr_data.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
    assert len(tr_data1.test_idxs) == len(tr_etypes)
    for etype in tr_etypes:
        assert th.all(tr_data1.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
    assert len(ev_data.test_idxs) == len(va_etypes)
    for etype in va_etypes:
        assert th.all(ev_data.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

    # pass train etypes as None
    assert len(tr_data2.train_idxs) == len(dist_graph.canonical_etypes)
    for etype in tr_etypes:
        assert th.all(tr_data2.train_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))
    for etype in tr_etypes:
        assert th.all(tr_data2.val_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))
    for etype in tr_etypes:
        assert th.all(tr_data2.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

    # train etypes does not cover all etypes.
    assert len(tr_data3.train_idxs) == len(tr_single_etype)
    for etype in tr_single_etype:
        assert th.all(tr_data3.train_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))
    for etype in tr_single_etype:
        assert th.all(tr_data3.val_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))
    for etype in tr_single_etype:
        assert th.all(tr_data3.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

    # pass eval etypes as None
    assert len(ev_data2.test_idxs) == 2
    for etype in dist_graph.canonical_etypes:
        assert th.all(ev_data2.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

    labels = tr_data.get_labels({('n0', 'r1', 'n1'): [0, 1]})
    assert len(labels.keys()) == 1
    assert ('n0', 'r1', 'n1') in labels
    try:
        labels = tr_data.get_labels({('n0', 'r0', 'n1'): [0, 1]})
        no_label = False
    except:
        no_label = True
    assert no_label
    try:
        labels = tr_data1.get_labels({('n0', 'r1', 'n1'): [0, 1]})
        no_label = False
    except:
        no_label = True
    assert no_label

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_GSgnnNodeData():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    tr_ntypes = ["n1"]
    va_ntypes = ["n1"]
    ts_ntypes = ["n1"]

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        dist_graph, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                            dirname=tmpdirname)
        tr_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                     train_ntypes=tr_ntypes, eval_ntypes=va_ntypes,
                                     label_field='label')
        tr_data1 = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                      train_ntypes=tr_ntypes)
        ev_data = GSgnnNodeInferData(graph_name='dummy', part_config=part_config,
                                     eval_ntypes=va_ntypes)

    # successful initialization with default setting
    assert tr_data.train_ntypes == tr_ntypes
    assert tr_data.eval_ntypes == va_ntypes
    assert tr_data1.train_ntypes == tr_ntypes
    assert tr_data1.eval_ntypes == tr_ntypes
    assert ev_data.eval_ntypes == va_ntypes

    # sucessfully split train/val/test idxs
    assert len(tr_data.train_idxs) == len(tr_ntypes)
    for ntype in tr_ntypes:
        assert th.all(tr_data.train_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['train_mask']))
    assert len(ev_data.train_idxs) == 0

    assert len(tr_data.val_idxs) == len(va_ntypes)
    for ntype in va_ntypes:
        assert th.all(tr_data.val_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['val_mask']))
    assert len(tr_data1.val_idxs) == len(tr_ntypes)
    for ntype in tr_ntypes:
        assert th.all(tr_data1.val_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['val_mask']))
    assert len(ev_data.val_idxs) == 0

    assert len(tr_data.test_idxs) == len(ts_ntypes)
    for ntype in ts_ntypes:
        assert th.all(tr_data.test_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['test_mask']))
    assert len(tr_data1.test_idxs) == len(tr_ntypes)
    for ntype in tr_ntypes:
        assert th.all(tr_data1.test_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['test_mask']))
    assert len(ev_data.test_idxs) == len(va_ntypes)
    for ntype in va_ntypes:
        assert th.all(ev_data.test_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['test_mask']))

    labels = tr_data.get_labels({'n1': [0, 1]})
    assert len(labels.keys()) == 1
    assert 'n1' in labels
    try:
        labels = tr_data.get_labels({'n0': [0, 1]})
        no_label = False
    except:
        no_label = True
    assert no_label
    try:
        labels = tr_data1.get_labels({'n1': [0, 1]})
        no_label = False
    except:
        no_label = True
    assert no_label

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

@pytest.mark.parametrize("batch_size", [1, 10, 128])
def test_GSgnnAllEtypeLinkPredictionDataLoader(batch_size):
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    tr_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        lp_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=tr_etypes, label_field='label')

    # successful initialization with default setting
    assert lp_data.train_etypes == tr_etypes
    dataloader = GSgnnAllEtypeLinkPredictionDataLoader(
        lp_data,
        target_idx=lp_data.train_idxs,
        fanout=[],
        batch_size=batch_size,
        num_negative_edges=4,
        device='cuda:0',
        exclude_training_targets=False)

    for input_nodes, pos_graph, neg_graph, blocks in dataloader:
        assert "n0" in input_nodes
        assert "n1" in input_nodes

        etypes = pos_graph.canonical_etypes
        assert ("n0", "r1", "n1") in etypes
        assert ("n0", "r0", "n1") in etypes

        etypes = neg_graph.canonical_etypes
        assert ("n0", "r1", "n1") in etypes
        assert ("n0", "r0", "n1") in etypes
    th.distributed.destroy_process_group()

def test_node_dataloader():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        np_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                     train_ntypes=['n1'], label_field='label')

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {'n1': th.arange(np_data.g.number_of_nodes('n1'))}
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=False)
    all_nodes = []
    for input_nodes, seeds, blocks in dataloader:
        assert 'n1' in seeds
        all_nodes.append(seeds['n1'])
    all_nodes = th.cat(all_nodes)
    assert_equal(all_nodes.numpy(), target_idx['n1'])

    # With data shuffling, the seed nodes should have different orders
    # whenever the data loader is called.
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=True)
    all_nodes1 = []
    for input_nodes, seeds, blocks in dataloader:
        assert 'n1' in seeds
        all_nodes1.append(seeds['n1'])
    all_nodes1 = th.cat(all_nodes1)
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=True)
    all_nodes2 = []
    for input_nodes, seeds, blocks in dataloader:
        assert 'n1' in seeds
        all_nodes2.append(seeds['n1'])
    all_nodes2 = th.cat(all_nodes2)
    assert not np.all(all_nodes1.numpy() == all_nodes2.numpy())

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_node_dataloader_reconstruct():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph_reconstruct(graph_name='dummy',
                                                               dirname=tmpdirname)
        np_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                     train_ntypes=['n0'], label_field='label',
                                     node_feat_field={'n0': ['feat'], 'n4': ['feat']})

    feat_sizes = gs.gsf.get_feat_size(np_data.g, {'n0': 'feat', 'n4': 'feat'})
    target_idx = {'n0': th.arange(np_data.g.number_of_nodes('n0'))}
    # Test the case that we cannot construct all node features.
    try:
        dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10, 'cuda:0',
                                         train_task=False, construct_feat_ntype=['n1', 'n2'])
        assert False
    except:
        pass

    # Test the case that we construct node features for one-layer GNN.
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=False, construct_feat_ntype=['n2'])
    all_nodes = []
    rel_names_for_reconstruct = gs.gsf.get_rel_names_for_reconstruct(np_data.g,
                                                                     ['n1', 'n2'], feat_sizes)
    for input_nodes, seeds, blocks in dataloader:
        assert 'n0' in seeds
        assert len(blocks) == 2
        for etype in blocks[0].canonical_etypes:
            if etype in rel_names_for_reconstruct:
                assert blocks[0].number_of_edges(etype) > 0
            else:
                assert blocks[0].number_of_edges(etype) == 0
        for ntype in blocks[1].srctypes:
            assert ntype in input_nodes
            nids = blocks[1].srcnodes[ntype].data[dgl.NID].numpy()
            assert len(nids) <= len(input_nodes[ntype])
            nodes = input_nodes[ntype].numpy()
            assert np.all(nodes[0:len(nids)] == nids)
        all_nodes.append(seeds['n0'])
    all_nodes = th.cat(all_nodes)
    assert_equal(all_nodes.numpy(), target_idx['n0'])

    # Test the case that we construct node features for two-layer GNN.
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10, 10], 10, 'cuda:0',
                                     train_task=False, construct_feat_ntype=['n3'])
    all_nodes = []
    rel_names_for_reconstruct = gs.gsf.get_rel_names_for_reconstruct(np_data.g,
                                                                     ['n3'], feat_sizes)
    for input_nodes, seeds, blocks in dataloader:
        assert 'n0' in seeds
        assert len(blocks) == 3
        for etype in blocks[0].canonical_etypes:
            if etype in rel_names_for_reconstruct:
                assert blocks[0].number_of_edges(etype) > 0
            else:
                assert blocks[0].number_of_edges(etype) == 0
        for ntype in blocks[1].srctypes:
            assert ntype in input_nodes
            nids = blocks[1].srcnodes[ntype].data[dgl.NID].numpy()
            assert len(nids) <= len(input_nodes[ntype])
            nodes = input_nodes[ntype].numpy()
            assert np.all(nodes[0:len(nids)] == nids)
        all_nodes.append(seeds['n0'])
    all_nodes = th.cat(all_nodes)
    assert_equal(all_nodes.numpy(), target_idx['n0'])

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_edge_dataloader():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=[('n0', 'r1', 'n1')], label_field='label')

    ################### Test train_task #######################

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=False, remove_target_edge_type=False)
    all_edges = []
    for input_nodes, batch_graph, blocks in dataloader:
        assert len(batch_graph.etypes) == 1
        assert 'r1' in batch_graph.etypes
        all_edges.append(batch_graph.edata[dgl.EID])
    all_edges = th.cat(all_edges)
    assert_equal(all_edges.numpy(), target_idx[('n0', 'r1', 'n1')])

    # With data shuffling, the seed edges should have different orders
    # whenever the data loader is called.
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=True, remove_target_edge_type=False)
    all_edges1 = []
    for input_nodes, batch_graph, blocks in dataloader:
        assert len(batch_graph.etypes) == 1
        assert 'r1' in batch_graph.etypes
        all_edges1.append(batch_graph.edata[dgl.EID])
    all_edges1 = th.cat(all_edges1)
    all_edges2 = []
    for input_nodes, batch_graph, blocks in dataloader:
        assert len(batch_graph.etypes) == 1
        assert 'r1' in batch_graph.etypes
        all_edges2.append(batch_graph.edata[dgl.EID])
    all_edges2 = th.cat(all_edges2)
    assert not np.all(all_edges1.numpy() == all_edges2.numpy())

    ################### Test removing target edges #######################
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10, 'cuda:0',
                                     train_task=False, remove_target_edge_type=True,
                                     reverse_edge_types_map={('n0', 'r1', 'n1'): ('n0', 'r0', 'n1')})
    all_edges = []
    for input_nodes, batch_graph, blocks in dataloader:
        # All edge types are excluded, so the block doesn't have any edges.
        assert blocks[0].number_of_edges() == 0

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_lp_dataloader():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=[('n0', 'r1', 'n1')])

    ################### Test train_task #######################

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], 10, num_negative_edges=2,
                                               device='cuda:0', train_task=False)
    all_edges = []
    for input_nodes, pos_graph, neg_graph, blocks in dataloader:
        assert len(pos_graph.etypes) == 1
        assert 'r1' in pos_graph.etypes
        all_edges.append(pos_graph.edata[dgl.EID])
    all_edges = th.cat(all_edges)
    assert_equal(all_edges.numpy(), target_idx[('n0', 'r1', 'n1')])

    # With data shuffling, the seed edges should have different orders
    # whenever the data loader is called.
    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], 10, num_negative_edges=2,
                                               device='cuda:0', train_task=True)
    all_edges1 = []
    for input_nodes, pos_graph, neg_graph, blocks in dataloader:
        assert len(pos_graph.etypes) == 1
        assert 'r1' in pos_graph.etypes
        all_edges1.append(pos_graph.edata[dgl.EID])
    all_edges1 = th.cat(all_edges1)
    all_edges2 = []
    for input_nodes, pos_graph, neg_graph, blocks in dataloader:
        assert len(pos_graph.etypes) == 1
        assert 'r1' in pos_graph.etypes
        all_edges2.append(pos_graph.edata[dgl.EID])
    all_edges2 = th.cat(all_edges2)
    assert not np.all(all_edges1.numpy() == all_edges2.numpy())

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

# initialize the torch distributed environment
@pytest.mark.parametrize("batch_size", [1, 10, 128])
@pytest.mark.parametrize("num_negative_edges", [1, 16, 128])
def test_GSgnnLinkPredictionTestDataLoader(batch_size, num_negative_edges):
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    test_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        lp_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=test_etypes, label_field='label')
        g = lp_data.g

        dataloader = GSgnnLinkPredictionTestDataLoader(
            lp_data,
            target_idx=lp_data.train_idxs, # use train edges as val or test edges
            batch_size=batch_size,
            num_negative_edges=num_negative_edges)

        total_edges = {etype: len(lp_data.train_idxs[etype]) for etype in test_etypes}
        num_pos_edges = {etype: 0 for etype in test_etypes}
        for pos_neg_tuple, sample_type in dataloader:
            assert sample_type == BUILTIN_LP_UNIFORM_NEG_SAMPLER
            assert isinstance(pos_neg_tuple, dict)
            assert len(pos_neg_tuple) == 1
            for canonical_etype, pos_neg in pos_neg_tuple.items():
                assert len(pos_neg) == 4
                pos_src, neg_src, pos_dst, neg_dst = pos_neg
                assert pos_src.shape == pos_dst.shape
                assert pos_src.shape[0] == batch_size \
                    if num_pos_edges[canonical_etype] + batch_size < total_edges[canonical_etype] \
                    else total_edges[canonical_etype] - num_pos_edges[canonical_etype]
                eid = lp_data.train_idxs[canonical_etype][num_pos_edges[canonical_etype]: \
                    num_pos_edges[canonical_etype]+batch_size] \
                    if num_pos_edges[canonical_etype]+batch_size < total_edges[canonical_etype] \
                    else lp_data.train_idxs[canonical_etype] \
                        [num_pos_edges[canonical_etype]:]
                src, dst = g.find_edges(eid, etype=canonical_etype)

                assert_equal(pos_src.numpy(), src.numpy())
                assert_equal(pos_dst.numpy(), dst.numpy())
                num_pos_edges[canonical_etype] += batch_size
                assert neg_dst.shape[0] == pos_src.shape[0]
                assert neg_dst.shape[1] == num_negative_edges
                assert th.all(neg_dst < g.number_of_nodes(canonical_etype[2]))

                assert neg_src.shape[0] == pos_src.shape[0]
                assert neg_src.shape[1] == num_negative_edges
                assert th.all(neg_src < g.number_of_nodes(canonical_etype[0]))

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

# initialize the torch distributed environment
@pytest.mark.parametrize("batch_size", [1, 10, 128])
@pytest.mark.parametrize("num_negative_edges", [1, 16, 128])
def test_GSgnnLinkPredictionJointTestDataLoader(batch_size, num_negative_edges):
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    test_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        lp_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=test_etypes, label_field='label')
        g = lp_data.g

        dataloader = GSgnnLinkPredictionJointTestDataLoader(
            lp_data,
            target_idx=lp_data.train_idxs, # use train edges as val or test edges
            batch_size=batch_size,
            num_negative_edges=num_negative_edges)

        total_edges = {etype: len(lp_data.train_idxs[etype]) for etype in test_etypes}
        num_pos_edges = {etype: 0 for etype in test_etypes}
        for pos_neg_tuple, sample_type in dataloader:
            assert sample_type == BUILTIN_LP_JOINT_NEG_SAMPLER
            assert isinstance(pos_neg_tuple, dict)
            assert len(pos_neg_tuple) == 1
            for canonical_etype, pos_neg in pos_neg_tuple.items():
                assert len(pos_neg) == 4
                pos_src, neg_src, pos_dst, neg_dst = pos_neg
                assert pos_src.shape == pos_dst.shape
                assert pos_src.shape[0] == batch_size \
                    if num_pos_edges[canonical_etype] + batch_size < total_edges[canonical_etype] \
                    else total_edges[canonical_etype] - num_pos_edges[canonical_etype]
                eid = lp_data.train_idxs[canonical_etype][num_pos_edges[canonical_etype]: \
                    num_pos_edges[canonical_etype]+batch_size] \
                    if num_pos_edges[canonical_etype]+batch_size < total_edges[canonical_etype] \
                    else lp_data.train_idxs[canonical_etype] \
                        [num_pos_edges[canonical_etype]:]
                src, dst = g.find_edges(eid, etype=canonical_etype)
                assert_equal(pos_src.numpy(), src.numpy())
                assert_equal(pos_dst.numpy(), dst.numpy())
                num_pos_edges[canonical_etype] += batch_size
                assert len(neg_dst.shape) == 1
                assert neg_dst.shape[0] == num_negative_edges
                assert th.all(neg_dst < g.number_of_nodes(canonical_etype[2]))

                assert len(neg_src.shape) == 1
                assert neg_src.shape[0] == num_negative_edges
                assert th.all(neg_src < g.number_of_nodes(canonical_etype[0]))

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_prepare_input():
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    test_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        lp_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=test_etypes, label_field='label')
        lp_data.g.nodes['n1'].data['feat2'] = \
            lp_data.g.nodes['n1'].data['feat'][th.arange(lp_data.g.num_nodes('n1'))] * 2
        lp_data.g.edges['r0'].data['feat2'] = \
            lp_data.g.edges['r0'].data['feat'][th.arange(lp_data.g.num_edges('r0'))] * 2
        g = lp_data.g

        # single ntype/edge, single feat
        input_nodes = {
            "n0": th.randint(g.num_nodes("n0"), (10,))
        }
        input_edges = {
            ("n0", "r1", "n1"): th.randint(g.num_edges(("n0", "r1", "n1")), (20,)),
        }

        node_feat = prepare_batch_input(g, input_nodes, feat_field='feat')
        edge_feat = prepare_batch_edge_input(g, input_edges, feat_field='feat')
        assert len(node_feat) == 1
        assert len(edge_feat) == 1
        assert_equal(node_feat["n0"].numpy(),
                     g.nodes["n0"].data["feat"][input_nodes["n0"]].numpy())
        assert_equal(edge_feat[("n0", "r1", "n1")].numpy(),
                     g.edges[("n0", "r1", "n1")].data["feat"][
                         input_edges[("n0", "r1", "n1")]].numpy())

        # multiple ntype/edge, single feat
        input_nodes = {
            "n0": th.randint(g.num_nodes("n0"), (10,)),
            "n1": th.randint(g.num_nodes("n1"), (20,)),
        }
        input_edges = {
            ("n0", "r1", "n1"): th.randint(g.num_edges(("n0", "r1", "n1")), (20,)),
            ("n0", "r0", "n1"): th.randint(g.num_edges(("n0", "r0", "n1")), (10,)),
        }

        node_feat = prepare_batch_input(g, input_nodes, feat_field='feat')
        edge_feat = prepare_batch_edge_input(g, input_edges, feat_field='feat')
        assert len(node_feat) == 2
        assert len(edge_feat) == 2
        assert_equal(node_feat["n0"].numpy(),
                     g.nodes["n0"].data["feat"][input_nodes["n0"]].numpy())
        assert_equal(node_feat["n1"].numpy(),
                     g.nodes["n1"].data["feat"][input_nodes["n1"]].numpy())
        assert_equal(edge_feat[("n0", "r1", "n1")].numpy(),
                     g.edges[("n0", "r1", "n1")].data["feat"][
                         input_edges[("n0", "r1", "n1")]].numpy())
        assert_equal(edge_feat[("n0", "r0", "n1")].numpy(),
                     g.edges[("n0", "r0", "n1")].data["feat"][
                         input_edges[("n0", "r0", "n1")]].numpy())

        # multiple ntype/edge, multiple feat
        input_nodes = {
            "n0": th.randint(g.num_nodes("n0"), (10,)),
            "n1": th.randint(g.num_nodes("n1"), (20,)),
        }
        input_edges = {
            ("n0", "r1", "n1"): th.randint(g.num_edges(("n0", "r1", "n1")), (20,)),
            ("n0", "r0", "n1"): th.randint(g.num_edges(("n0", "r0", "n1")), (10,)),
        }

        node_feat = prepare_batch_input(g, input_nodes,
                                        feat_field={"n0":["feat"],
                                                    "n1":["feat", "feat2"]})
        edge_feat = prepare_batch_edge_input(g, input_edges,
                                             feat_field={
                                                 ("n0", "r1", "n1"): ["feat"],
                                                 ("n0", "r0", "n1"): ["feat", "feat2"]})
        assert len(node_feat) == 2
        assert len(edge_feat) == 2
        assert_equal(node_feat["n0"].numpy(),
                     g.nodes["n0"].data["feat"][input_nodes["n0"]].numpy())
        assert_equal(node_feat["n1"].numpy(),
                     th.cat([g.nodes["n1"].data["feat"][input_nodes["n1"]],
                             g.nodes["n1"].data["feat2"][input_nodes["n1"]]], dim=-1).numpy())
        assert_equal(edge_feat[("n0", "r1", "n1")].numpy(),
                     g.edges[("n0", "r1", "n1")].data["feat"][
                         input_edges[("n0", "r1", "n1")]].numpy())
        assert_equal(edge_feat[("n0", "r0", "n1")].numpy(),
                     th.cat([g.edges[("n0", "r0", "n1")].data["feat"][
                                 input_edges[("n0", "r0", "n1")]],
                             g.edges[("n0", "r0", "n1")].data["feat2"][
                                 input_edges[("n0", "r0", "n1")]]], dim=-1).numpy())

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_modify_fanout_for_target_etype():
    data_dict = {
        ('user', 'follows', 'user'): (th.tensor([0, 1]), th.tensor([1, 2])),
        ('user', 'follows', 'topic'): (th.tensor([1, 1]), th.tensor([1, 2])),
        ('user', 'plays', 'game'): (th.tensor([0, 3]), th.tensor([3, 4]))
    }
    g = dgl.heterograph(data_dict)
    fanout = [10,5]
    target_etypes = [('user', 'follows', 'user')]
    new_fanout = modify_fanout_for_target_etype(g, fanout, target_etypes)
    assert len(new_fanout) == 2
    assert new_fanout[0][('user', 'follows', 'user')] == 0
    assert new_fanout[0][('user', 'follows', 'topic')] == 10
    assert new_fanout[0][('user', 'plays', 'game')] == 10
    assert new_fanout[1][('user', 'follows', 'user')] == 0
    assert new_fanout[1][('user', 'follows', 'topic')] == 5
    assert new_fanout[1][('user', 'plays', 'game')] == 5

    fanout = [{("user","follows","user"):20,
               ("user","follows","topic"):10,
               ("user","plays","game"):5},
              {("user","follows","user"):3,
               ("user","follows","topic"):2,
               ("user","plays","game"):1}]
    new_fanout = modify_fanout_for_target_etype(g, fanout, target_etypes)
    assert len(new_fanout) == 2
    assert new_fanout[0][('user', 'follows', 'user')] == 0
    assert new_fanout[0][('user', 'follows', 'topic')] == 10
    assert new_fanout[0][('user', 'plays', 'game')] == 5
    assert new_fanout[1][('user', 'follows', 'user')] == 0
    assert new_fanout[1][('user', 'follows', 'topic')] == 2
    assert new_fanout[1][('user', 'plays', 'game')] == 1

@pytest.mark.parametrize("dataloader", [GSgnnNodeDataLoader])
def test_np_dataloader_trim_data(dataloader):
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        np_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                     train_ntypes=['n1'], label_field='label')

        target_idx = {'n1': th.arange(np_data.g.number_of_nodes('n1'))}
        @patch("graphstorm.dataloading.dataloading.trim_data")
        def check_dataloader_trim(mock_trim_data):
            mock_trim_data.side_effect = [
                target_idx["n1"][:len(target_idx["n1"])-2],
                target_idx["n1"][:len(target_idx["n1"])-2],
            ]

            loader = dataloader(np_data, dict(target_idx),
                                [10], 10, 'cpu',
                                train_task=True)
            assert len(loader.dataloader.collator.nids) == 1
            assert len(loader.dataloader.collator.nids["n1"]) == np_data.g.number_of_nodes('n1') - 2

            loader = dataloader(np_data, dict(target_idx),
                                [10], 10, 'cpu',
                                train_task=False)
            assert len(loader.dataloader.collator.nids) == 1
            assert len(loader.dataloader.collator.nids["n1"]) == np_data.g.number_of_nodes('n1')

        check_dataloader_trim()

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()


@pytest.mark.parametrize("dataloader", [GSgnnAllEtypeLinkPredictionDataLoader,
                                        GSgnnLinkPredictionDataLoader,
                                        FastGSgnnLinkPredictionDataLoader])
def test_edge_dataloader_trim_data(dataloader):
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    test_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        lp_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=test_etypes, label_field='label')
        g = lp_data.g

        @patch("graphstorm.dataloading.dataloading.trim_data")
        def check_dataloader_trim(mock_trim_data):
            mock_trim_data.side_effect = [
                lp_data.train_idxs[("n0", "r1", "n1")][:len(lp_data.train_idxs[("n0", "r1", "n1")])-1],
                lp_data.train_idxs[("n0", "r0", "n1")][:len(lp_data.train_idxs[("n0", "r0", "n1")])-1],

                lp_data.train_idxs[("n0", "r1", "n1")][:len(lp_data.train_idxs[("n0", "r1", "n1")])-1],
                lp_data.train_idxs[("n0", "r0", "n1")][:len(lp_data.train_idxs[("n0", "r0", "n1")])-1],
            ]

            loader = dataloader(
                lp_data,
                fanout=[],
                target_idx=dict(lp_data.train_idxs), # test train_idxs
                batch_size=16,
                num_negative_edges=4)

            assert len(loader.dataloader.collator.eids) == len(lp_data.train_idxs)
            for etype in lp_data.train_idxs.keys():
                assert len(loader.dataloader.collator.eids[etype]) == len(lp_data.train_idxs[etype]) - 1

            # test task, trim_data should not be called.
            loader = dataloader(
                lp_data,
                target_idx=dict(lp_data.train_idxs), # use train edges as val or test edges
                fanout=[],
                batch_size=16,
                num_negative_edges=4,
                train_task=False)

            assert len(loader.dataloader.collator.eids) == len(lp_data.train_idxs)
            for etype in lp_data.train_idxs.keys():
                assert len(loader.dataloader.collator.eids[etype]) == len(lp_data.train_idxs[etype])

        check_dataloader_trim()

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()


if __name__ == '__main__':
    test_np_dataloader_trim_data(GSgnnNodeDataLoader)
    test_edge_dataloader_trim_data(GSgnnLinkPredictionDataLoader)
    test_edge_dataloader_trim_data(FastGSgnnLinkPredictionDataLoader)
    test_GSgnnEdgeData_wo_test_mask()
    test_GSgnnNodeData_wo_test_mask()
    test_GSgnnEdgeData()
    test_GSgnnNodeData()
    test_lp_dataloader()
    test_edge_dataloader()
    test_node_dataloader()
    test_node_dataloader_reconstruct()
    test_GSgnnAllEtypeLinkPredictionDataLoader(10)
    test_GSgnnAllEtypeLinkPredictionDataLoader(1)
    test_GSgnnLinkPredictionTestDataLoader(1, 1)
    test_GSgnnLinkPredictionTestDataLoader(10, 20)
    test_GSgnnLinkPredictionJointTestDataLoader(1, 1)
    test_GSgnnLinkPredictionJointTestDataLoader(10, 20)

    test_prepare_input()
    test_modify_fanout_for_target_etype()
