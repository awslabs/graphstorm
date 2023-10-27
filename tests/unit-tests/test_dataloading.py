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
import multiprocessing as mp
import torch.distributed as dist
from unittest.mock import patch, MagicMock

import torch as th
import dgl
import pytest
from data_utils import (
    generate_dummy_dist_graph,
    generate_dummy_dist_graph_reconstruct,
    create_distill_data,
)

import graphstorm as gs
from graphstorm.utils import setup_device
from graphstorm.dataloading import GSgnnNodeTrainData, GSgnnNodeInferData
from graphstorm.dataloading import GSgnnEdgeTrainData, GSgnnEdgeInferData
from graphstorm.dataloading import GSgnnAllEtypeLinkPredictionDataLoader
from graphstorm.dataloading import (GSgnnNodeDataLoader,
                                    GSgnnEdgeDataLoader,
                                    GSgnnNodeSemiSupDataLoader)
from graphstorm.dataloading import (GSgnnLinkPredictionDataLoader,
                                    GSgnnLPJointNegDataLoader,
                                    GSgnnLPLocalUniformNegDataLoader,
                                    GSgnnLPLocalJointNegDataLoader,
                                    FastGSgnnLinkPredictionDataLoader,
                                    FastGSgnnLPJointNegDataLoader,
                                    FastGSgnnLPLocalUniformNegDataLoader,
                                    FastGSgnnLPLocalJointNegDataLoader)
from graphstorm.dataloading import GSgnnLinkPredictionTestDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionJointTestDataLoader
from graphstorm.dataloading import DistillDataloaderGenerator, DistillDataManager
from graphstorm.dataloading import DistributedFileSampler
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER

from graphstorm.dataloading.dataset import (prepare_batch_input,
                                            prepare_batch_edge_input)
from graphstorm.dataloading.utils import modify_fanout_for_target_etype
from graphstorm.dataloading.utils import trim_data

from numpy.testing import assert_equal
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

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
    assert len(ev_data.infer_idxs) == len(ev_data.test_idxs)
    for etype in va_etypes:
        assert th.all(ev_data.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
        assert th.all(ev_data.infer_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

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
    assert len(ev_data2.infer_idxs) == 2
    for etype in dist_graph.canonical_etypes:
        assert th.all(ev_data2.test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
        assert th.all(ev_data2.infer_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

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
    assert len(ev_data.infer_idxs) == len(va_ntypes)
    for ntype in va_ntypes:
        assert th.all(ev_data.test_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['test_mask']))
        assert th.all(ev_data.infer_idxs[ntype] == get_nonzero(dist_graph.nodes[ntype].data['test_mask']))

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

def get_dist_sampler_attributes(rank, world_size, num_files):
    """
    Assign a slice window of file index to each worker.
    The slice window of each worker is specified
    by self.global_start and self.global_end
    """
    if world_size > num_files:
        # If num of workers is greater than num of files,
        # the slice windows are same across all workers,
        # which covers all files.
        remainder = world_size % num_files
        global_start = 0
        global_end = num_files
        part_len = global_end
    else:
        # If num of workers is smaller than num of files,
        # the slice windows are different for each worker.
        # In the case where the files cannot be evenly distributed,
        # the remainder will be assigned to one or multiple workers evenly.
        part_len = num_files // world_size
        remainder = num_files % world_size
        global_start = part_len * rank + min(rank, remainder)
        global_end = global_start + part_len + (rank < remainder)
        part_len = global_end - global_start
    return global_start, global_end, part_len, remainder

@pytest.mark.parametrize("num_files", [3, 7, 8])
def test_distill_sampler_get_file(num_files):
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_distill_data(tmpdirname, num_files)
        file_list = os.listdir(tmpdirname)
        tracker = {
            "global_start": [],
            "global_end": [],
            "part_len": [],
        }
        for rank in range(4):
            dist_sampler = DistributedFileSampler(
                        dataset_path=tmpdirname,
                        shuffle=False,
                        local_rank=rank,
                        world_size=4,
                        is_train=True,
                        infinite=True,
                    )

            # test DistributedFileSampler._file_index_distribute
            global_start, global_end, part_len, remainder = \
                get_dist_sampler_attributes(rank, 4, num_files)
            assert global_start == dist_sampler.global_start
            assert global_end == dist_sampler.global_end
            assert part_len == dist_sampler.part_len
            assert remainder == dist_sampler.remainder
            tracker["global_start"].append(dist_sampler.global_start)
            tracker["global_end"].append(dist_sampler.global_end)
            tracker["part_len"].append(dist_sampler.part_len)
            if rank == 0:
                tracker["remainder"] = dist_sampler.remainder

            # test DistributedFileSampler.get_file
            if num_files == 3:
                if rank == 0 or rank == 3:
                    target_index = [0, 2, 1]
                elif rank == 1:
                    target_index = [1, 0, 2]
                elif rank == 2:
                    target_index = [2, 1, 0]

            if num_files == 7:
                if rank == 0:
                    target_index = [0, 1]
                elif rank == 1:
                    target_index = [2, 3]
                elif rank == 2:
                    target_index = [4, 5]
                elif rank == 3:
                    target_index = [6]

            if num_files == 8:
                if rank == 0:
                    target_index = [0, 1]
                elif rank == 1:
                    target_index = [2, 3]
                elif rank == 2:
                    target_index = [4, 5]
                elif rank == 3:
                    target_index = [6, 7]
            for offset in range(2*num_files):
                assert os.path.join(tmpdirname, file_list[target_index[offset%len(target_index)]]) == \
                    dist_sampler.get_file(offset)

        # test relative relation
        if num_files >= 4:
            assert tracker["global_end"][0] == tracker["global_start"][1]
            assert tracker["global_end"][1] == tracker["global_start"][2]
            assert tracker["global_end"][2] == tracker["global_start"][3]
            total_len = 0
            for rank in range(4):
                assert tracker["part_len"][rank] == \
                    tracker["global_end"][rank] - tracker["global_start"][rank]
                total_len += tracker["part_len"][rank]
            assert total_len == num_files
        else:
            for rank in range(4):
                assert tracker["part_len"][rank] == num_files
                assert tracker["global_start"][rank] == 0
                assert tracker["global_end"][rank] == num_files
                assert tracker["remainder"] == 4 % num_files

@pytest.mark.parametrize("num_files", [3, 7, 8])
@pytest.mark.parametrize("is_train", [True, False])
@pytest.mark.parametrize("infinite", [False, True])
@pytest.mark.parametrize("shuffle", [True, False])
def test_DistillDistributedFileSampler(num_files, is_train,
    infinite, shuffle):
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_distill_data(tmpdirname, num_files)
        global_sampled_files = []

        # set world_size to 4
        # test 1) when num_files < world size.
        # 2) when num_files >= world size and can be evenly divided.
        # 3) when num_files > world size and cannot be evenly divided.

        for rank in range(4):
            device = setup_device(rank)

            file_sampler = DistributedFileSampler(
                    dataset_path=tmpdirname,
                    shuffle=shuffle,
                    local_rank=rank,
                    world_size=4,
                    is_train=is_train,
                    infinite=infinite,
                )
            if is_train and num_files >= 4:
                assert file_sampler.part_len >= (num_files // 4)
                assert file_sampler.part_len <= (num_files // 4) + 1
            else:
                assert file_sampler.part_len == num_files

            file_sampler_iter = iter(file_sampler)
            if is_train and infinite:
                local_sampled_files = []
                for i, data_file in enumerate(file_sampler_iter):
                    if i == file_sampler.part_len:
                        assert data_file.split("/")[-1] in local_sampled_files, \
                            "Infinite sampler doesn't sample evenly."
                        break
                    global_sampled_files.append(data_file.split("/")[-1])
                    local_sampled_files.append(data_file.split("/")[-1])
            else:
                for i, data_file in enumerate(file_sampler_iter):
                    if data_file is None:
                        break
                    assert i < file_sampler.part_len, \
                        "Non-infinite sampler doesn't exit."
                    global_sampled_files.append(data_file.split("/")[-1])
        assert set(global_sampled_files) == set(os.listdir(tmpdirname))


def run_distill_dist_data(worker_rank, world_size,
    backend, tmpdirname, num_files, is_train):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=worker_rank)
    th.cuda.set_device(worker_rank)
    device = setup_device(worker_rank)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    dataloader_generator = DistillDataloaderGenerator(
        tokenizer=tokenizer,
        max_seq_len=8,
        device=device,
        batch_size=4,
    )
    data_mgr = DistillDataManager(
        dataloader_generator,
        dataset_path=tmpdirname,
        local_rank=worker_rank,
        world_size=world_size,
        is_train=is_train,
    )

    if is_train and num_files >= 4:
        assert len(data_mgr) >= num_files // 4
        assert len(data_mgr) <= num_files // 4 + 1
    else:
        assert len(data_mgr) == num_files

    dataset_iterator = data_mgr.get_iterator()
    assert isinstance(dataset_iterator, DataLoader)
    batch = next(iter(dataset_iterator))
    assert len(batch) == 3

    data_mgr.refresh_manager()

    if is_train:
        train_idx = 0
        while True:
            dataset_iterator = data_mgr.get_iterator()
            assert isinstance(dataset_iterator, DataLoader)
            num_batches = th.tensor(len(dataset_iterator), \
                dtype=th.int64, device=device)
            dist.all_reduce(num_batches, op=dist.ReduceOp.MIN)
            min_size = num_batches.item()
            assert int(min_size) == int(num_batches)
            if train_idx == 2 * num_files:
                break
            train_idx += 1
    else:
        for i, dataset_iterator in enumerate(data_mgr):
            if i < len(data_mgr):
                assert isinstance(dataset_iterator, DataLoader)
            if dataset_iterator is None:
                assert i == len(data_mgr)
                break
            assert i < len(data_mgr), \
                "DistillDataManager doesn't exit."

@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("num_files", [3, 7, 8])
@pytest.mark.parametrize("is_train", [True, False])
def test_DistillDataloaderGenerator(backend, num_files, is_train):
    # test DistillDataloaderGenerator
    with tempfile.TemporaryDirectory() as tmpdirname:
        create_distill_data(tmpdirname, num_files)
        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_distill_dist_data,
                        args=(0, 4, backend, tmpdirname, num_files, is_train))
        p1 = ctx.Process(target=run_distill_dist_data,
                        args=(1, 4, backend, tmpdirname, num_files, is_train))
        p2 = ctx.Process(target=run_distill_dist_data,
                        args=(2, 4, backend, tmpdirname, num_files, is_train))
        p3 = ctx.Process(target=run_distill_dist_data,
                        args=(3, 4, backend, tmpdirname, num_files, is_train))

        p0.start()
        p1.start()
        p2.start()
        p3.start()
        p0.join()
        p1.join()
        p2.join()
        p3.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0
        assert p2.exitcode == 0
        assert p3.exitcode == 0

@pytest.mark.parametrize("batch_size", [10, 11])
def test_np_dataloader_len(batch_size):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        np_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                     train_ntypes=['n1'], label_field='label')

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {'n1': th.arange(np_data.g.number_of_nodes('n1'))}
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], batch_size, 'cuda:0',
                                     train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], batch_size, 'cuda:0',
                                     train_task=True)
    assert len(dataloader) == len(list(dataloader))

    target_idx_2 = {'n1': th.arange(np_data.g.number_of_nodes('n1')//2)}
    # target_idx > unlabeled_idx
    dataloader = GSgnnNodeSemiSupDataLoader(np_data, target_idx, target_idx_2, [10], batch_size, 'cuda:0',
                                     train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnNodeSemiSupDataLoader(np_data, target_idx, target_idx_2, [10], batch_size, 'cuda:0',
                                     train_task=True)
    assert len(dataloader) == len(list(dataloader))

    # target_idx < unlabeled_idx
    dataloader = GSgnnNodeSemiSupDataLoader(np_data, target_idx_2, target_idx, [10], batch_size, 'cuda:0',
                                     train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnNodeSemiSupDataLoader(np_data, target_idx_2, target_idx, [10], batch_size, 'cuda:0',
                                     train_task=True)
    assert len(dataloader) == len(list(dataloader))

@pytest.mark.parametrize("batch_size", [10, 11])
def test_ep_dataloader_len(batch_size):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=[('n0', 'r1', 'n1')], label_field='label')

    ################### Test train_task #######################

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], batch_size, 'cuda:0',
                                     train_task=False, remove_target_edge_type=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], batch_size, 'cuda:0',
                                     train_task=True, remove_target_edge_type=False)
    assert len(dataloader) == len(list(dataloader))

@pytest.mark.parametrize("batch_size", [10, 11])
def test_lp_dataloader_len(batch_size):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=[('n0', 'r1', 'n1')])

    ################### Test train_task #######################

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPLocalUniformNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPLocalUniformNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPLocalJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPLocalJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPLocalUniformNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPLocalUniformNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPLocalJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPLocalJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               device='cuda:0', train_task=True)
    assert len(dataloader) == len(list(dataloader))

if __name__ == '__main__':
    test_np_dataloader_len(11)
    test_ep_dataloader_len(11)
    test_lp_dataloader_len(11)

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

    test_distill_sampler_get_file(num_files=7)
    test_DistillDistributedFileSampler(num_files=7, is_train=True, \
        infinite=False, shuffle=True)
    test_DistillDataloaderGenerator("gloo", 7, True)
