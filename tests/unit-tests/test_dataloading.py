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
    generate_dummy_dist_graph_homogeneous_failure_graph,
    generate_dummy_dist_graph_multi_task,
    create_distill_data,
    SIZE_DICT
)

import graphstorm as gs
from graphstorm.config import (TaskInfo,
                               BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION,
                               BUILTIN_TASK_EDGE_CLASSIFICATION,
                               BUILTIN_TASK_LINK_PREDICTION)
from graphstorm.utils import setup_device, get_device
from graphstorm.dataloading import GSgnnData
from graphstorm.dataloading import GSgnnAllEtypeLinkPredictionDataLoader
from graphstorm.dataloading import (GSgnnNodeDataLoader,
                                    GSgnnEdgeDataLoader,
                                    GSgnnNodeSemiSupDataLoader)
from graphstorm.dataloading import (GSgnnLinkPredictionDataLoaderBase,
                                    GSgnnLinkPredictionDataLoader,
                                    GSgnnLPJointNegDataLoader,
                                    GSgnnLPInBatchJointNegDataLoader,
                                    GSgnnLPLocalUniformNegDataLoader,
                                    GSgnnLPLocalJointNegDataLoader,
                                    FastGSgnnLinkPredictionDataLoader,
                                    FastGSgnnLPJointNegDataLoader,
                                    FastGSgnnLPLocalUniformNegDataLoader,
                                    FastGSgnnLPLocalJointNegDataLoader)
from graphstorm.dataloading import (GSgnnLinkPredictionTestDataLoader,
                                    GSgnnLinkPredictionJointTestDataLoader,
                                    GSgnnLinkPredictionPredefinedTestDataLoader)
from graphstorm.dataloading import DistillDataloaderGenerator, DistillDataManager
from graphstorm.dataloading import DistributedFileSampler
from graphstorm.dataloading import GSgnnMultiTaskDataLoader
from graphstorm.dataloading import (BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                                    BUILTIN_LP_JOINT_NEG_SAMPLER,
                                    BUILTIN_LP_FIXED_NEG_SAMPLER)

from graphstorm.dataloading.sampler import InbatchJointUniform
from graphstorm.dataloading.sampler import GlobalUniform
from graphstorm.dataloading.sampler import (GSHardEdgeDstNegativeSampler,
                                             GSFixedEdgeDstNegativeSampler)

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

def test_GSgnnData():
    # Create dist graph without reverse edges
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    tr_ntypes = ['n1']
    va_ntypes = ['n1']
    ts_ntypes = ['n1']
    in_ntypes = ['n1']

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        dist_graph, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                            dirname=tmpdirname, add_reverse=False)
        gdata = GSgnnData(part_config=part_config)

        assert gdata.graph_name == 'dummy'
        assert gdata.node_feat_field == None
        assert gdata.edge_feat_field == None
        assert gdata.has_node_feats('n1') is False
        assert gdata.has_node_feats(('n0', 'r0', 'n1')) is False
        assert gdata.has_node_lm_feats('n1') is False
        assert gdata.has_edge_lm_feats(('n0', 'r0', 'n1')) is False

        # test node train/val/test idxs
        for ntype in tr_ntypes:
            assert th.all(gdata.get_node_train_set(ntype)[ntype] == get_nonzero(dist_graph.nodes[ntype].data['train_mask']))
        for ntype in va_ntypes:
            assert th.all(gdata.get_node_val_set(ntype)[ntype] == get_nonzero(dist_graph.nodes[ntype].data['val_mask']))
        for ntype in ts_ntypes:
            assert th.all(gdata.get_node_test_set(ntype)[ntype] == get_nonzero(dist_graph.nodes[ntype].data['test_mask']))
        for ntype in in_ntypes:
            # with test mask
            assert th.all(gdata.get_node_infer_set(ntype)[ntype] == get_nonzero(dist_graph.nodes[ntype].data['test_mask']))
        # without test mask
        assert th.all(gdata.get_node_infer_set('n0')['n0'] == th.arange(dist_graph.num_nodes('n0')))

        # check node label
        labels = gdata.get_node_feats({'n1': th.tensor([0, 1])}, 'label')
        assert len(labels.keys()) == 1
        assert 'n1' in labels
        try:
            labels = gdata.get_node_feats({'n0': th.tensor([0, 1])}, 'label')
            no_label = False
        except:
            no_label = True
        assert no_label

        tr_etypes = [('n0', 'r1', 'n1'), ('n0', 'r0', 'n1')]
        va_etypes = [('n0', 'r1', 'n1'), ('n0', 'r0', 'n1')]
        ts_etypes = [('n0', 'r1', 'n1'), ('n0', 'r0', 'n1')]
        if_etypes = [('n0', 'r1', 'n1'), ('n0', 'r0', 'n1')]
        # test edge train/val/test idxs
        train_idxs = gdata.get_edge_train_set()
        assert len(train_idxs) == 2
        for etype in tr_etypes:
            assert th.all(gdata.get_edge_train_set(etype)[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))
            assert th.all(train_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))

        val_idxs = gdata.get_edge_val_set()
        assert len(val_idxs) == 2
        for etype in va_etypes:
            assert th.all(gdata.get_edge_val_set(etype)[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))
            assert th.all(val_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))

        test_idxs = gdata.get_edge_test_set()
        for etype in ts_etypes:
            assert th.all(gdata.get_edge_test_set(etype)[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
            assert th.all(test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

        infer_idxs = gdata.get_edge_infer_set()
        assert len(infer_idxs) == 2
        for etype in if_etypes:
            assert th.all(gdata.get_edge_infer_set(etype)[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
            assert th.all(infer_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

        # check edge label
        labels = gdata.get_edge_feats({('n0', 'r1', 'n1'): th.tensor([0, 1])}, 'label')
        assert len(labels.keys()) == 1
        assert ('n0', 'r1', 'n1') in labels
        try:
            labels = gdata.get_edge_feats({('n0', 'r0', 'n1'): th.tensor([0, 1])}, 'label')
            no_label = False
        except:
            no_label = True
        assert no_label

        ##################################
        # provide node feat and edge feat
        gdata = GSgnnData(part_config=part_config, node_feat_field='feat', edge_feat_field='feat', lm_feat_ntypes=['n0'], lm_feat_etypes=[('n0', 'r0', 'n1')])

        assert gdata.graph_name == 'dummy'
        assert gdata.node_feat_field == 'feat'
        assert gdata.edge_feat_field == 'feat'
        assert gdata.has_node_feats('n0') is True
        assert gdata.has_node_feats(('n0', 'r0', 'n1')) is True
        assert gdata.has_node_feats('n1') is True
        assert gdata.has_node_feats(('n0', 'r1', 'n1')) is True
        assert gdata.has_node_lm_feats('n0') is True
        assert gdata.has_edge_lm_feats(('n0', 'r0', 'n1')) is True
        assert gdata.has_node_lm_feats('n1') is False
        assert gdata.has_edge_lm_feats(('n0', 'r1', 'n1')) is False

        feat = gdata.get_node_feats({'n0': th.tensor([0, 1])}, 'feat')
        assert th.all(feat['n0'] == dist_graph.nodes['n0'].data['feat'][th.tensor([0, 1])])
        feat = gdata.get_node_feats({'n0': th.tensor([0, 1])}, {'n0':['feat', 'feat1']})
        assert th.all(feat['n0'] == \
                      th.cat([dist_graph.nodes['n0'].data['feat'][th.tensor([0, 1])],
                             dist_graph.nodes['n0'].data['feat1'][th.tensor([0, 1])]], dim=1))
        try:
            feat = gdata.get_node_feats({'n0': th.tensor([0, 1])}, "feat2")
            no_feat = False
        except:
            no_feat = True
        assert no_feat

        feat = gdata.get_node_feats({'n0': th.tensor([0, 1]),
                                     'n1': th.tensor([0, 1])},
                                     {'n0':['feat', 'feat1'],
                                      'n1':['feat1']})
        assert len(feat) == 2
        assert th.all(feat['n0'] == \
            th.cat([dist_graph.nodes['n0'].data['feat'][th.tensor([0, 1])],
                   dist_graph.nodes['n0'].data['feat1'][th.tensor([0, 1])]], dim=1))
        assert th.all(feat['n1'] == dist_graph.nodes['n1'].data['feat1'][th.tensor([0, 1])])

        feat = gdata.get_edge_feats({('n0', 'r0', 'n1'): th.tensor([0, 1])}, 'feat')
        assert th.all(feat[('n0', 'r0', 'n1')] == dist_graph.edges[('n0', 'r0', 'n1')].data['feat'][th.tensor([0, 1])])
        feat = gdata.get_edge_feats({('n0', 'r0', 'n1'): th.tensor([0, 1])}, {('n0', 'r0', 'n1'): ['feat']})
        assert th.all(feat[('n0', 'r0', 'n1')] == dist_graph.edges[('n0', 'r0', 'n1')].data['feat'][th.tensor([0, 1])])
        try:
            feat = gdata.get_edge_feats({('n0', 'r0', 'n1'): th.tensor([0, 1])}, "fea2")
            no_feat = False
        except:
            no_feat = True
        assert no_feat
        feat = gdata.get_edge_feats({('n0', 'r0', 'n1'): th.tensor([0, 1]),
                                     ('n0', 'r1', 'n1'): th.tensor([0, 1])},
                                     'feat')
        assert len(feat) == 2
        assert th.all(feat[('n0', 'r0', 'n1')] == dist_graph.edges[('n0', 'r0', 'n1')].data['feat'][th.tensor([0, 1])])
        assert th.all(feat[('n0', 'r1', 'n1')] == dist_graph.edges[('n0', 'r1', 'n1')].data['feat'][th.tensor([0, 1])])
        try:
            feat = gdata.get_edge_feats({('n0', 'r0', 'n1'): th.tensor([0, 1]),
                                         ('n0', 'r1', 'n1'): th.tensor([0, 1])},
                                        {('n0', 'r0', 'n1'):'feat',
                                         ('n0', 'r1', 'n1'):'feat1'})
            no_feat = False
        except:
            no_feat = True
        assert no_feat

        ########################
        # Test internal functions
        # test _check_ntypes
        ntypes = None
        try:
            ntypes = gdata._check_ntypes(ntypes)
            err_check = False
        except:
            err_check = True
        assert err_check
        ntypes = 'n0'
        ntypes = gdata._check_ntypes(ntypes)
        assert len(ntypes) == 1
        assert ntypes[0] == 'n0'
        ntypes = ['n0', 'n1']
        ntypes = gdata._check_ntypes(ntypes)
        assert len(ntypes) == 2
        assert ntypes[0] == 'n0'
        assert ntypes[1] == 'n1'
        ntypes = {'n1':0}
        try:
            ntypes = gdata._check_ntypes(ntypes)
            err_check = False
        except:
            err_check = True
        assert err_check

        # test _check_node_mask
        ntypes = 'n0'
        masks = gdata._check_node_mask(ntypes, 'train_m')
        assert len(masks) == 1
        assert masks[0] == 'train_m'
        ntypes = ['n0', 'n1']
        masks = gdata._check_node_mask(ntypes, 'train_m')
        assert len(masks) == 2
        assert masks[0] == 'train_m'
        assert masks[1] == 'train_m'
        ntypes = ['n0', 'n1']
        masks = gdata._check_node_mask(ntypes, ['train_m0', 'train_m1'])
        assert len(masks) == 2
        assert masks[0] == 'train_m0'
        assert masks[1] == 'train_m1'
        ntypes = ['n0', 'n1']
        try:
            masks = gdata._check_node_mask(ntypes, ['train_m0'])
            err_mask = False
        except:
            err_mask = True
        assert err_mask

        # test _check_etypes
        etypes = None
        try:
            etypes = gdata._check_etypes(etypes)
            err_check = False
        except:
            err_check = True
        assert err_check
        etypes = ('n0', 'r0', 'n1')
        etypes = gdata._check_etypes(etypes)
        assert len(etypes) == 1
        assert etypes[0] == ('n0', 'r0', 'n1')
        etypes = [('n0', 'r0', 'n1'), ('n0', 'r0', 'n2')]
        etypes = gdata._check_etypes(etypes)
        assert len(etypes) == 2
        assert etypes[0] == ('n0', 'r0', 'n1')
        assert etypes[1] == ('n0', 'r0', 'n2')
        etypes = {('n0', 'r0', 'n2'):0}
        try:
            etypes = gdata._check_etypes(etypes)
            err_check = False
        except:
            err_check = True
        assert err_check

        # test _check_edge_mask
        etypes = ('n0', 'r0', 'n1')
        masks = gdata._check_edge_mask(etypes, 'train_m')
        assert len(masks) == 1
        assert masks[0] == 'train_m'
        etypes = [('n0', 'r0', 'n1'), ('n0', 'r0', 'n2')]
        masks = gdata._check_edge_mask(etypes, 'train_m')
        assert len(masks) == 2
        assert masks[0] == 'train_m'
        assert masks[1] == 'train_m'
        etypes = [('n0', 'r0', 'n1'), ('n0', 'r0', 'n2')]
        masks = gdata._check_edge_mask(etypes, ['train_m0', 'train_m1'])
        assert len(masks) == 2
        assert masks[0] == 'train_m0'
        assert masks[1] == 'train_m1'
        etypes = [('n0', 'r0', 'n1'), ('n0', 'r0', 'n2')]
        try:
            masks = gdata._check_edge_mask(etypes, ['train_m0'])
            err_mask = False
        except:
            err_mask = True
        assert err_mask

        # test _exclude_reverse_etype
        etypes = [('n0', 'r0', 'n1'), ('n0', 'r0', 'n2'), ('n1', 'r2', 'n0'), ('n2', 'r3', 'n0')]
        new_etypes = gdata._exclude_reverse_etype(etypes)
        assert len(new_etypes) == 4
        assert set(etypes) == set(new_etypes)

        new_etypes = gdata._exclude_reverse_etype(etypes, reverse_edge_types_map={('n0', 'r0', 'n2'): ('n2', 'r3', 'n0')})
        assert len(new_etypes) == 3
        assert set(etypes[:3]) == set(new_etypes)

        new_etypes = gdata._exclude_reverse_etype(etypes, reverse_edge_types_map={('n2', 'r3', 'n0'): ('n0', 'r0', 'n2'),
            ('n1', 'r2', 'n0'):('n0', 'r0', 'n1')})
        assert len(new_etypes) == 2
        assert set(etypes[2:]) == set(new_etypes)

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_GSgnnData2():
    # Create dist graph with reverse edges
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        dist_graph, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                            dirname=tmpdirname, add_reverse=True)
        # there will be three etypes:
        # ('n0', 'r1', 'n1'), ('n0', 'r0', 'n1'), ("n1", "r2", "n0")
        gdata = GSgnnData(part_config=part_config)

        # There are reverse edges
        tr_etypes = [('n0', 'r1', 'n1'), ('n0', 'r0', 'n1')]
        va_etypes = [('n0', 'r1', 'n1'), ('n0', 'r0', 'n1')]
        ts_etypes = [('n0', 'r1', 'n1'), ('n0', 'r0', 'n1')]
        if_etypes = [('n0', 'r1', 'n1'), ('n0', 'r0', 'n1')]
        extra_etype = ("n1", "r2", "n0")
        # test edge train/val/test idxs
        train_idxs = gdata.get_edge_train_set()
        assert len(train_idxs) == 3
        for etype in tr_etypes:
            assert th.all(gdata.get_edge_train_set(etype)[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))
            assert th.all(train_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))
        # extra_etype does not have train_mask
        assert th.all(gdata.get_edge_train_set(extra_etype)[extra_etype] == \
            th.arange(dist_graph.num_edges(extra_etype)))
        assert th.all(train_idxs[extra_etype] == \
            th.arange(dist_graph.num_edges(extra_etype)))

        # extra_etype does not have val_mask
        # will skip it.
        val_idxs = gdata.get_edge_val_set()
        assert len(val_idxs) == 2
        for etype in va_etypes:
            assert th.all(gdata.get_edge_val_set(etype)[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))
            assert th.all(val_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['val_mask']))

        # extra_etype does not have test_mask
        # will skip it.
        test_idxs = gdata.get_edge_test_set()
        for etype in ts_etypes:
            assert th.all(gdata.get_edge_test_set(etype)[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
            assert th.all(test_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

        infer_idxs = gdata.get_edge_infer_set()
        assert len(infer_idxs) == 3
        for etype in if_etypes:
            assert th.all(gdata.get_edge_infer_set(etype)[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
            assert th.all(infer_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))
        # extra_etype does not have test_mask
        assert th.all(gdata.get_edge_infer_set(extra_etype)[extra_etype] == \
            th.arange(dist_graph.num_edges(extra_etype)))
        assert th.all(infer_idxs[extra_etype] ==
            th.arange(dist_graph.num_edges(extra_etype)))

        # with reverse edges
        train_idxs = gdata.get_edge_train_set(reverse_edge_types_map={
            ('n0', 'r1', 'n1'): ("n1", "r2", "n0")
        })
        assert len(train_idxs) == 2
        for etype in tr_etypes:
            assert th.all(train_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['train_mask']))
        infer_idxs = gdata.get_edge_infer_set(reverse_edge_types_map={
            ('n0', 'r1', 'n1'): ("n1", "r2", "n0")
        })
        assert len(infer_idxs) == 2
        for etype in if_etypes:
            assert th.all(infer_idxs[etype] == get_nonzero(dist_graph.edges[etype[1]].data['test_mask']))

        train_idxs = gdata.get_edge_train_set(reverse_edge_types_map={
            ('n0', 'r1', 'n1'): ("n1", "r2", "n0"),
            ('n0', 'r0', 'n1'): ('n0', 'r0', 'n1') # fake reverse etype
        })
        assert len(train_idxs) == 1
        val_idxs = gdata.get_edge_val_set(reverse_edge_types_map={
            ('n0', 'r1', 'n1'): ("n1", "r2", "n0"),
            ('n0', 'r0', 'n1'): ('n0', 'r0', 'n1') # fake reverse etype
        })
        assert len(val_idxs) == 1
        test_idxs = gdata.get_edge_test_set(reverse_edge_types_map={
            ('n0', 'r1', 'n1'): ("n1", "r2", "n0"),
            ('n0', 'r0', 'n1'): ('n0', 'r0', 'n1') # fake reverse etype
        })
        assert len(test_idxs) == 1
        infer_idxs = gdata.get_edge_infer_set(reverse_edge_types_map={
            ('n0', 'r1', 'n1'): ("n1", "r2", "n0"),
            ('n0', 'r0', 'n1'): ('n0', 'r0', 'n1') # fake reverse etype
        })
        assert len(infer_idxs) == 1

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
        lp_data = GSgnnData(part_config=part_config)

    # successful initialization with default setting
    train_idxs = lp_data.get_edge_train_set(tr_etypes)
    dataloader = GSgnnAllEtypeLinkPredictionDataLoader(
        lp_data,
        target_idx=train_idxs,
        fanout=[],
        batch_size=batch_size,
        num_negative_edges=4,
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
        np_data = GSgnnData(part_config=part_config)

    setup_device(0)
    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {'n1': th.arange(np_data.g.number_of_nodes('n1'))}
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10,
                                     label_field='label',
                                     node_feats='feat',
                                     edge_feats='feat',
                                     train_task=False)
    all_nodes = []
    for input_nodes, seeds, blocks in dataloader:
        assert 'n1' in seeds
        all_nodes.append(seeds['n1'])
    all_nodes = th.cat(all_nodes)
    assert_equal(all_nodes.numpy(), target_idx['n1'])
    assert dataloader.node_feat_fields == 'feat'
    assert dataloader.edge_feat_fields == 'feat'

    # With data shuffling, the seed nodes should have different orders
    # whenever the data loader is called.
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10,
                                     label_field='label',
                                     train_task=True)
    all_nodes1 = []
    for input_nodes, seeds, blocks in dataloader:
        assert 'n1' in seeds
        all_nodes1.append(seeds['n1'])
    all_nodes1 = th.cat(all_nodes1)
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10,
                                     label_field='label',
                                     train_task=True)
    assert dataloader.node_feat_fields == None
    assert dataloader.edge_feat_fields == None

    all_nodes2 = []
    for input_nodes, seeds, blocks in dataloader:
        assert 'n1' in seeds
        all_nodes2.append(seeds['n1'])
    all_nodes2 = th.cat(all_nodes2)
    assert not np.all(all_nodes1.numpy() == all_nodes2.numpy())

    # fail to create GSgnnNodeDataLoader when label field is set to None.
    try:
        dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10,
                                         label_field=None,
                                         train_task=True)
        init_loader_fail = False
    except:
        init_loader_fail = True
    assert init_loader_fail

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
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field={'n0': ['feat'], 'n4': ['feat']})
    setup_device(0)
    feat_sizes = gs.get_node_feat_size(np_data.g, {'n0': 'feat', 'n4': 'feat'})
    target_idx = {'n0': th.arange(np_data.g.number_of_nodes('n0'))}
    # Test the case that we cannot construct all node features.
    try:
        dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10, label_field='label',
                                         train_task=False,
                                         node_feats={'n0': ['feat'], 'n4': ['feat']},
                                         construct_feat_ntype=['n1', 'n2'])
        assert False
    except:
        pass

    # Test the case that we construct node features for one-layer GNN.
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10, label_field='label',
                                     train_task=False,
                                     node_feats={'n0': ['feat'], 'n4': ['feat']},
                                     construct_feat_ntype=['n2'])
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
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10, 10], 10, label_field='label',
                                     train_task=False,
                                     node_feats={'n0': ['feat'], 'n4': ['feat']},
                                     construct_feat_ntype=['n3'])
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
        ep_data = GSgnnData(part_config=part_config)

    ################### Test train_task #######################

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10,
                                     label_field='label',
                                     train_task=False, remove_target_edge_type=False)
    all_edges = []
    for _, batch_graph, blocks in dataloader:
        assert len(batch_graph.etypes) == 1
        assert 'r1' in batch_graph.etypes
        all_edges.append(batch_graph.edata[dgl.EID])
    all_edges = th.cat(all_edges)
    assert_equal(all_edges.numpy(), target_idx[('n0', 'r1', 'n1')])

    # With data shuffling, the seed edges should have different orders
    # whenever the data loader is called.
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10,
                                     label_field='label',
                                     train_task=True, remove_target_edge_type=False)
    all_edges1 = []
    for _, batch_graph, blocks in dataloader:
        assert len(batch_graph.etypes) == 1
        assert 'r1' in batch_graph.etypes
        all_edges1.append(batch_graph.edata[dgl.EID])
    all_edges1 = th.cat(all_edges1)
    all_edges2 = []
    for _, batch_graph, blocks in dataloader:
        assert len(batch_graph.etypes) == 1
        assert 'r1' in batch_graph.etypes
        all_edges2.append(batch_graph.edata[dgl.EID])
    all_edges2 = th.cat(all_edges2)
    assert not np.all(all_edges1.numpy() == all_edges2.numpy())

    ################### Test removing target edges #######################
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10,
                                     label_field='label',
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
        ep_data = GSgnnData(part_config=part_config)

    ################### Test train_task #######################

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], 10,
                                               num_negative_edges=2, train_task=False)
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
                                               train_task=True)
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
        lp_data = GSgnnData(part_config=part_config)
        g = lp_data.g
        train_idxs = lp_data.get_edge_train_set(test_etypes)
        dataloader = GSgnnLinkPredictionTestDataLoader(
            lp_data,
            target_idx=train_idxs, # use train edges as val or test edges
            batch_size=batch_size,
            num_negative_edges=num_negative_edges)

        total_edges = {etype: len(train_idxs[etype]) for etype in test_etypes}
        num_pos_edges = {etype: 0 for etype in test_etypes}
        num_samples = 0
        for pos_neg_tuple, sample_type in dataloader:
            num_samples += 1
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
                eid = train_idxs[canonical_etype][num_pos_edges[canonical_etype]: \
                    num_pos_edges[canonical_etype]+batch_size] \
                    if num_pos_edges[canonical_etype]+batch_size < total_edges[canonical_etype] \
                    else train_idxs[canonical_etype] \
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
        assert len(dataloader) == num_samples

        # The target idx size for ("n0", "r1", "n1") is 2
        # The target idx size for ("n0", "r0", "n1") is 50
        fixed_test_size = 10
        dataloader = GSgnnLinkPredictionTestDataLoader(
            lp_data,
            target_idx=train_idxs, # use train edges as val or test edges
            batch_size=batch_size,
            num_negative_edges=num_negative_edges,fixed_test_size=fixed_test_size)
        num_samples = 0
        num_pos_pairs = 0
        for pos_neg_tuple, sample_type in dataloader:
            num_samples += 1
            assert isinstance(pos_neg_tuple, dict)
            assert len(pos_src) > 0
            assert len(pos_neg_tuple) == 1
            for _, pos_neg in pos_neg_tuple.items():
                pos_src, _, pos_dst, _ = pos_neg
                assert len(pos_src) <= batch_size
                num_pos_pairs += len(pos_src)

        expected_samples = 0
        expected_pos_pairs = 0
        for _, t_idx in train_idxs.items():
            expected_idx_len = fixed_test_size if fixed_test_size < len(t_idx) else len(t_idx)
            expected_samples += -(-expected_idx_len // batch_size)
            expected_pos_pairs += expected_idx_len

        assert num_samples == expected_samples
        assert len(dataloader) == num_samples
        assert num_pos_pairs == expected_pos_pairs

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

@pytest.mark.parametrize("batch_size", [1, 10, 128])
def test_GSgnnLinkPredictionPredefinedTestDataLoader(batch_size):
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    test_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        lp_data = GSgnnData(part_config=part_config)
        g = lp_data.g
        g.edges[("n0", "r1", "n1")].data["neg"] = th.randint(g.num_nodes("n1"),
                                                             (g.num_edges(("n0", "r1", "n1")), 10))
        g.edges[("n0", "r0", "n1")].data["neg"] = th.randint(g.num_nodes("n1"),
                                                             (g.num_edges(("n0", "r0", "n1")), 10))

        infer_idxs = lp_data.get_edge_infer_set(test_etypes)
        dataloader = GSgnnLinkPredictionPredefinedTestDataLoader(
            lp_data,
            target_idx=infer_idxs, # use train edges as val or test edges
            batch_size=batch_size,
            fixed_edge_dst_negative_field="neg")

        total_edges = {etype: len(infer_idxs[etype]) for etype in test_etypes}
        num_pos_edges = {etype: 0 for etype in test_etypes}
        for pos_neg_tuple, sample_type in dataloader:
            assert sample_type == BUILTIN_LP_FIXED_NEG_SAMPLER
            assert isinstance(pos_neg_tuple, dict)
            assert len(pos_neg_tuple) == 1
            for canonical_etype, pos_neg in pos_neg_tuple.items():
                assert len(pos_neg) == 4
                pos_src, _, pos_dst, neg_dst = pos_neg
                assert pos_src.shape == pos_dst.shape
                assert pos_src.shape[0] == batch_size \
                    if num_pos_edges[canonical_etype] + batch_size < total_edges[canonical_etype] \
                    else total_edges[canonical_etype] - num_pos_edges[canonical_etype]
                eid = infer_idxs[canonical_etype][num_pos_edges[canonical_etype]: \
                    num_pos_edges[canonical_etype]+batch_size] \
                    if num_pos_edges[canonical_etype]+batch_size < total_edges[canonical_etype] \
                    else infer_idxs[canonical_etype] \
                        [num_pos_edges[canonical_etype]:]
                src, dst = g.find_edges(eid, etype=canonical_etype)
                assert_equal(pos_src.numpy(), src.numpy())
                assert_equal(pos_dst.numpy(), dst.numpy())
                assert len(neg_dst.shape) == 2
                assert neg_dst.shape[1] == 10
                assert_equal(neg_dst.numpy(), g.edges[canonical_etype].data["neg"][eid].numpy())

                num_pos_edges[canonical_etype] += batch_size
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
        lp_data = GSgnnData(part_config=part_config)
        g = lp_data.g
        train_idxs = lp_data.get_edge_train_set(test_etypes)
        dataloader = GSgnnLinkPredictionJointTestDataLoader(
            lp_data,
            target_idx=train_idxs, # use train edges as val or test edges
            batch_size=batch_size,
            num_negative_edges=num_negative_edges)

        total_edges = {etype: len(train_idxs[etype]) for etype in test_etypes}
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
                eid = train_idxs[canonical_etype][num_pos_edges[canonical_etype]: \
                    num_pos_edges[canonical_etype]+batch_size] \
                    if num_pos_edges[canonical_etype]+batch_size < total_edges[canonical_etype] \
                    else train_idxs[canonical_etype] \
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
        lp_data = GSgnnData(part_config=part_config)
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
        np_data = GSgnnData(part_config=part_config)

        target_idx = {'n1': th.arange(np_data.g.number_of_nodes('n1'))}
        @patch("graphstorm.dataloading.dataloading.trim_data")
        def check_dataloader_trim(mock_trim_data):
            mock_trim_data.side_effect = [
                target_idx["n1"][:len(target_idx["n1"])-2],
                target_idx["n1"][:len(target_idx["n1"])-2],
            ]

            loader = dataloader(np_data, dict(target_idx),
                                [10], 10, label_field='label',
                                train_task=True)
            assert len(loader.dataloader.collator.nids) == 1
            assert len(loader.dataloader.collator.nids["n1"]) == np_data.g.number_of_nodes('n1') - 2

            loader = dataloader(np_data, dict(target_idx),
                                [10], 10, label_field='label',
                                train_task=False)
            assert len(loader.dataloader.collator.nids) == 1
            assert len(loader.dataloader.collator.nids["n1"]) == np_data.g.number_of_nodes('n1')

        check_dataloader_trim()

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

@pytest.mark.parametrize("dataloader", [GSgnnNodeDataLoader])
@pytest.mark.parametrize("backend", ['gloo', 'nccl'])
def test_np_dataloader_trim_data_device(dataloader, backend):
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend=backend,
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    setup_device(0)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        np_data = GSgnnData(part_config=part_config)

        target_idx = {'n1': th.arange(np_data.g.number_of_nodes('n1'))}
        @patch("graphstorm.dataloading.dataloading.trim_data")
        def check_dataloader_trim(mock_trim_data):
            mock_trim_data.side_effect = [
                target_idx["n1"][:len(target_idx["n1"])-2],
                target_idx["n1"][:len(target_idx["n1"])-2],
            ]

            dataloader(np_data, dict(target_idx), [10], 10,
                       label_field='label', train_task=True)
            dataloader(np_data, dict(target_idx), [10], 10,
                       label_field='label', train_task=False)

            args, _ = mock_trim_data.call_args
            assert args[1] == th.device('cpu') if backend == 'gloo' else th.device(0)
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
        lp_data = GSgnnData(part_config=part_config)
        train_idxs = lp_data.get_edge_train_set(test_etypes)
        @patch("graphstorm.dataloading.dataloading.trim_data")
        def check_dataloader_trim(mock_trim_data):
            mock_trim_data.side_effect = [
                train_idxs[("n0", "r1", "n1")][:len(train_idxs[("n0", "r1", "n1")])-1],
                train_idxs[("n0", "r0", "n1")][:len(train_idxs[("n0", "r0", "n1")])-1],

                train_idxs[("n0", "r1", "n1")][:len(train_idxs[("n0", "r1", "n1")])-1],
                train_idxs[("n0", "r0", "n1")][:len(train_idxs[("n0", "r0", "n1")])-1],
            ]

            loader = dataloader(
                lp_data,
                fanout=[],
                target_idx=dict(train_idxs), # test train_idxs
                batch_size=16,
                num_negative_edges=4)

            assert len(loader.dataloader.collator.eids) == len(train_idxs)
            for etype in train_idxs.keys():
                assert len(loader.dataloader.collator.eids[etype]) == len(train_idxs[etype]) - 1

            # test task, trim_data should not be called.
            loader = dataloader(
                lp_data,
                target_idx=dict(train_idxs), # use train edges as val or test edges
                fanout=[],
                batch_size=16,
                num_negative_edges=4,
                train_task=False)

            assert len(loader.dataloader.collator.eids) == len(train_idxs)
            for etype in train_idxs.keys():
                assert len(loader.dataloader.collator.eids[etype]) == len(train_idxs[etype])

        check_dataloader_trim()

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

@pytest.mark.parametrize("dataloader", [GSgnnAllEtypeLinkPredictionDataLoader,
                                        GSgnnLinkPredictionDataLoader,
                                        FastGSgnnLinkPredictionDataLoader,
                                        GSgnnEdgeDataLoader])
@pytest.mark.parametrize("backend", ['gloo', 'nccl'])
def test_edge_dataloader_trim_data_device(dataloader, backend):
     # initialize the torch distributed environment
    th.distributed.init_process_group(backend=backend,
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    setup_device(0)
    test_etypes = [("n0", "r1", "n1"), ("n0", "r0", "n1")]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        edge_data = GSgnnData(part_config=part_config)
        train_idxs = edge_data.get_edge_train_set([("n0", "r1", "n1"), ("n0", "r0", "n1")])

        @patch("graphstorm.dataloading.dataloading.trim_data")
        def check_dataloader_trim(mock_trim_data):
            mock_trim_data.side_effect = [
                train_idxs[("n0", "r1", "n1")][:len(train_idxs[("n0", "r1", "n1")])-1],
                train_idxs[("n0", "r0", "n1")][:len(train_idxs[("n0", "r0", "n1")])-1],

                train_idxs[("n0", "r1", "n1")][:len(train_idxs[("n0", "r1", "n1")])-1],
                train_idxs[("n0", "r0", "n1")][:len(train_idxs[("n0", "r0", "n1")])-1],
            ]

            if issubclass(dataloader, GSgnnLinkPredictionDataLoaderBase):
                dataloader(
                    edge_data,
                    fanout=[],
                    target_idx=dict(train_idxs), # test train_idxs
                    num_negative_edges=4,
                    batch_size=16)
            else:
                dataloader(
                    edge_data,
                    fanout=[],
                    target_idx=dict(train_idxs), # test train_idxs
                    batch_size=16,
                    label_field='label',
                    remove_target_edge_type=False)

            args, _ = mock_trim_data.call_args
            assert args[1] == th.device('cpu') if backend == 'gloo' else th.device(0)

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
        file_list.sort()
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
    setup_device(worker_rank)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    dataloader_generator = DistillDataloaderGenerator(
        tokenizer=tokenizer,
        max_seq_len=8,
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
        np_data = GSgnnData(part_config=part_config)

    setup_device(0)
    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {'n1': th.arange(np_data.g.number_of_nodes('n1'))}
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], batch_size,
                                     label_field='label',
                                     train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], batch_size,
                                     label_field='label',
                                     train_task=True)
    assert len(dataloader) == len(list(dataloader))

    target_idx_2 = {'n1': th.arange(np_data.g.number_of_nodes('n1')//2)}
    # target_idx > unlabeled_idx
    dataloader = GSgnnNodeSemiSupDataLoader(np_data, target_idx, target_idx_2, [10],
                                            batch_size, label_field='label',
                                            train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnNodeSemiSupDataLoader(np_data, target_idx, target_idx_2, [10],
                                            batch_size, label_field='label',
                                            train_task=True)
    assert len(dataloader) == len(list(dataloader))

    # target_idx < unlabeled_idx
    dataloader = GSgnnNodeSemiSupDataLoader(np_data, target_idx_2, target_idx, [10],
                                            batch_size, label_field='label',
                                            train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnNodeSemiSupDataLoader(np_data, target_idx_2, target_idx, [10],
                                            batch_size, label_field='label',
                                            train_task=True)
    assert len(dataloader) == len(list(dataloader))

@pytest.mark.parametrize("batch_size", [10, 11])
def test_ep_dataloader_len(batch_size):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnData(part_config=part_config)

    ################### Test train_task #######################

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], batch_size,
                                     label_field='label',
                                     train_task=False, remove_target_edge_type=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], batch_size,
                                     label_field='label',
                                     train_task=True, remove_target_edge_type=False)
    assert len(dataloader) == len(list(dataloader))

@pytest.mark.parametrize("batch_size", [10, 11])
def test_lp_dataloader_len(batch_size):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnData(part_config=part_config)

    ################### Test train_task #######################

    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                           train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                           train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPLocalUniformNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                  train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPLocalUniformNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                  train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPInBatchJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                  train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPLocalJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = GSgnnLPLocalJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                   train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLinkPredictionDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                   train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                               train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPLocalUniformNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                      train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPLocalUniformNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                      train_task=True)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPLocalJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                    train_task=False)
    assert len(dataloader) == len(list(dataloader))

    dataloader = FastGSgnnLPLocalJointNegDataLoader(ep_data, target_idx, [10], batch_size, num_negative_edges=2,
                                                    train_task=True)
    assert len(dataloader) == len(list(dataloader))

def _create_hard_neg_graph(num_nodes, num_negs):
    etype0 = ("n0", "r0", "n1")
    etype1 = ("n0", "r0", "n2")
    etype2 = ("n0", "r0", "n3")
    src = th.arange(num_nodes)
    dst = th.arange(num_nodes)
    # each edge has 4 pre-defined hard negatives
    hard0 = th.randint(num_nodes, (num_nodes, 4))
    # each edge has 10 pre-defined hard negatives
    hard1 = th.randint(num_nodes, (num_nodes, num_negs))
    # each edge has 20 pre-defined hard negatives
    hard2 = th.randint(num_nodes, (num_nodes, num_negs*2)) # more hard negatives than num neg
    g = dgl.heterograph({
        etype0: (src, dst),
        etype1: (src, dst),
        etype2: (src, dst),
    })
    g.edges[etype0].data["hard_negative"] = hard0
    g.edges[etype1].data["hard_negative"] = hard1
    g.edges[etype2].data["hard_negative"] = hard2

    return etype0, etype1, etype2, hard0, hard1, hard2, src, dst, g

def test_hard_edge_dst_negative_sample_generate_complex_case():
    # test GSHardEdgeDstNegativeSampler._generate with slow track when not all the pos edges have enough hard negatives defined
    num_nodes = 1000
    # test GSHardEdgeDstNegativeSampler._generate when all some pos edges do not have enough hard negatives defined
    num_negs = 10
    etype0, etype1, etype2, hard0, hard1, hard2, src, _, g = _create_hard_neg_graph(num_nodes, num_negs)

    # not enough predefined hard negatives
    # for hard0[0] and hard0[1]
    hard0[0] = th.randperm(num_nodes)[:4]
    hard0[0][-1] = -1
    hard0[0][-2] = -1
    hard0[1][-1] = -1

    # not enough predefined hard negatives
    # for hard0[0] and hard0[1]
    hard1[0] = th.randperm(num_nodes)[:num_negs]
    hard1[1] = th.randperm(num_nodes)[:num_negs]
    hard1[0][-1] = -1
    hard1[0][-2] = -1
    hard1[1][-1] = -1

    # not enough predefined hard negatives
    # for hard0[0] and hard0[1]
    hard2[0] = th.randperm(num_nodes)[:num_negs*2]
    hard2[1] = th.randperm(num_nodes)[:num_negs*2]
    hard2[0][-1] = -1
    hard2[0][-2] = -1
    hard2[1][-1] = -1

    num_edges = 10
    eids = th.arange(num_edges)
    def test_missing_hard_negs(neg_dst, num_hard_neg, hard_neg_data):
        # hardx[0][-1] and hardx[0][-2] is -1,
        # which means hardx[0] does not enough predefined negatives
        # Random sample will be applied to -1s.
        hard_neg_dst = neg_dst[0][:num_hard_neg]
        hard_neg_rand_0 = hard_neg_dst[-1]
        hard_neg_rand_1 = hard_neg_dst[-2]
        hard_neg_dst = set(hard_neg_dst[:-2].tolist())
        rand_neg_dst = neg_dst[0][num_hard_neg:]
        rand_neg_dst = set(rand_neg_dst.tolist())
        hard_neg_set = set(hard_neg_data[0].tolist())
        assert hard_neg_dst.issubset(hard_neg_set)
        assert len(rand_neg_dst) == 0  or \
            rand_neg_dst.issubset(hard_neg_set) is False

        rand_0_check = hard_neg_rand_0 not in hard_neg_set
        rand_1_check = hard_neg_rand_1 not in hard_neg_set

        # hardx[1][-1] is -1,
        # which means hardx[0] does not enough predefined negatives
        # Random sample will be applied to -1s.
        hard_neg_dst = neg_dst[1][:num_hard_neg]
        hard_neg_rand_2 = hard_neg_dst[-1]
        hard_neg_dst = set(hard_neg_dst[:-1].tolist())
        rand_neg_dst = neg_dst[1][num_hard_neg:]
        rand_neg_dst = set(rand_neg_dst.tolist())
        hard_neg_set = set(hard_neg_data[1].tolist())
        assert hard_neg_dst.issubset(hard_neg_set)
        assert len(rand_neg_dst) == 0  or \
            rand_neg_dst.issubset(hard_neg_set) is False

        rand_2_check = hard_neg_rand_2 not in hard_neg_set
        # The chance is very to to have rand_0_check,
        # rand_1_check and rand_2_check be true at the same time
        # The change is (4/1000)^3
        assert rand_0_check or rand_1_check or rand_2_check

    def check_less_hard_negs(hard_neg_sampler, target_etype, hard_neg_data,
                             num_hard_neg, check_missing_hard_neg):
        neg_src, neg_dst = hard_neg_sampler._generate(g, eids, target_etype)
        assert len(neg_src) == num_edges * num_negs
        assert len(neg_dst) == num_edges * num_negs
        assert_equal(th.repeat_interleave(src[:10], num_negs, 0).numpy(), neg_src.numpy())
        neg_dst = neg_dst.reshape(num_edges, num_negs)

        if check_missing_hard_neg:
            test_missing_hard_negs(neg_dst, num_hard_neg, hard_neg_data)

        start = 2 if check_missing_hard_neg else 0
        for i in range(start, num_edges):
            hard_neg_dst = neg_dst[i][:num_hard_neg]
            hard_neg_dst = set(hard_neg_dst.tolist())
            rand_neg_dst = neg_dst[i][num_hard_neg:]
            rand_neg_dst = set(rand_neg_dst.tolist())
            hard_neg_set = set(hard_neg_data[i].tolist())
            assert hard_neg_dst.issubset(hard_neg_set)
            assert rand_neg_dst.issubset(hard_neg_set) is False

    # case 1:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs > number of hard negatives required (2)
    # 4. num_negs > total number of hard negatives
    # provided (hard0 has 4 negatives for each node)
    # 5. Each edge has enough hard negative even though some edges do not have enough (10) predefined negatives
    # 6. slow track (-1 exists in hard neg feature)
    #
    # expected behavior:
    #   1. Only 2 hard negatives are returned
    #   2. Others will be random negatives
    sampler = GlobalUniform(num_negs)
    hard_sampler = GSHardEdgeDstNegativeSampler(num_negs, "hard_negative", sampler, num_hard_negs=2)
    check_less_hard_negs(hard_sampler, etype0, hard0, 2, check_missing_hard_neg=False)

    # Case 2:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs > number of hard negatives required (2)
    # 4. num_negs == total number of hard negatives
    #    provided (hard1 has 10 negatives for each node)
    # 5. Each edge has enough hard negative even though some edges do not have enough (8) predefined negatives
    # 6. slow track (-1 exists in hard neg feature)
    #
    # expected behavior:
    #   1. Only 2 hard negatives are returned
    #   2. Others will be random negatives
    check_less_hard_negs(hard_sampler, etype1, hard1, 2, check_missing_hard_neg=False)

    # Case 3:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs > number of hard negatives required (8)
    # 4. number of hard negatives required (8) > number of hard negatives
    #    provided (hard0 has only 4 negatives for each node)
    # 5.slow track (-1 exists in hard neg feature)
    #
    # expected behavior:
    #   1. Only 4 hard negatives are returned
    #   2. Others will be random negatives
    #   3. eid 0 will have 2 more random negatives
    #      and eid 1 will have 1 more random negatives
    hard_sampler = GSHardEdgeDstNegativeSampler(num_negs, "hard_negative", sampler, num_hard_negs=8)
    check_less_hard_negs(hard_sampler, etype0, hard0, 4, check_missing_hard_neg=True)

    # Case 4:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs == number of hard negatives required (10)
    # 4. number of hard negatives required (8) == number of hard negatives
    #    provided (hard1 has 10 negatives for each node)
    # 5.slow track (-1 exists in hard neg feature)
    #
    # expected behavior:
    #   1. Equal negatives
    def check_enough_hard_negs(hard_neg_sampler, target_etype, hard_neg_data):
        neg_src, neg_dst = hard_neg_sampler._generate(g, eids, target_etype)
        assert len(neg_src) == num_edges * num_negs
        assert len(neg_dst) == num_edges * num_negs
        assert_equal(th.repeat_interleave(src[:10], num_negs, 0).numpy(), neg_src.numpy())
        neg_dst = neg_dst.reshape(num_edges, num_negs)

        test_missing_hard_negs(neg_dst, num_negs, hard_neg_data)

        for i in range(2, num_edges):
            hard_neg_dst = set(neg_dst[i].tolist())
            hard_neg_set = set(hard_neg_data[i].tolist())
            assert hard_neg_dst == hard_neg_set

    hard_sampler = GSHardEdgeDstNegativeSampler(num_negs, "hard_negative", sampler, num_hard_negs=num_negs)
    check_enough_hard_negs(hard_sampler, etype1, hard1)

    # Case 5:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs == number of hard negatives required (10)
    # 4. number of hard negatives required (8) < number of hard negatives
    #    provided (hard2 has 20 negatives for each node)
    # 5.slow track (-1 exists in hard neg feature)
    #
    # expected behavior:
    #   1. hard negatives will be a subset of hard2
    def check_more_hard_negs(hard_neg_sampler, target_etype, hard_neg_data):
        neg_src, neg_dst = hard_neg_sampler._generate(g, eids, target_etype)
        assert len(neg_src) == num_edges * num_negs
        assert len(neg_dst) == num_edges * num_negs
        assert_equal(th.repeat_interleave(src[:10], num_negs, 0).numpy(), neg_src.numpy())
        neg_dst = neg_dst.reshape(num_edges, num_negs)
        for i in range(num_edges):
            hard_neg_dst = set(neg_dst[i].tolist())
            hard_neg_set = set(hard_neg_data[i].tolist())
            assert hard_neg_dst.issubset(hard_neg_set)
    check_more_hard_negs(hard_sampler, etype2, hard2)

def test_hard_edge_dst_negative_sample_generate():
    # test GSHardEdgeDstNegativeSampler._generate with fast track when all pos edges have enough hard negatives defined
    num_nodes = 100
    num_negs = 10
    etype0, etype1, etype2, hard0, hard1, hard2, src, _, g = _create_hard_neg_graph(num_nodes, num_negs)

    num_edges = 10
    eids = th.arange(num_edges)
    def check_less_hard_negs(hard_neg_sampler, target_etype, hard_neg_data, num_hard_neg):
        neg_src, neg_dst = hard_neg_sampler._generate(g, eids, target_etype)
        assert len(neg_src) == num_edges * num_negs
        assert len(neg_dst) == num_edges * num_negs
        assert_equal(th.repeat_interleave(src[:10], num_negs, 0).numpy(), neg_src.numpy())
        neg_dst = neg_dst.reshape(num_edges, num_negs)
        for i in range(num_edges):
            hard_neg_dst = neg_dst[i][:num_hard_neg]
            hard_neg_dst = set(hard_neg_dst.tolist())
            rand_neg_dst = neg_dst[i][num_hard_neg:]
            rand_neg_dst = set(rand_neg_dst.tolist())
            hard_neg_set = set(hard_neg_data[i].tolist())
            assert hard_neg_dst.issubset(hard_neg_set)
            assert rand_neg_dst.issubset(hard_neg_set) is False

    # case 1:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs > number of hard negatives required (2)
    # 4. num_negs > total number of hard negatives
    # provided (hard0 has 4 negatives for each node)
    # 5. fast track
    #
    # expected behavior:
    #   1. Only 2 hard negatives are returned
    #   2. Others will be random negatives
    sampler = GlobalUniform(num_negs)
    hard_sampler = GSHardEdgeDstNegativeSampler(num_negs, "hard_negative", sampler, num_hard_negs=2)
    check_less_hard_negs(hard_sampler, etype0, hard0, 2)

    # Case 2:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs > number of hard negatives required (2)
    # 4. num_negs == total number of hard negatives
    #    provided (hard1 has 10 negatives for each node)
    # 5. fast track
    #
    # expected behavior:
    #   1. Only 2 hard negatives are returned
    #   2. Others will be random negatives
    check_less_hard_negs(hard_sampler, etype1, hard1, 2)

    # Case 3:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs > number of hard negatives required (8)
    # 4. number of hard negatives required (8) > number of hard negatives
    #    provided (hard0 has only 4 negatives for each node)
    # 5.fast track
    #
    # expected behavior:
    #   1. Only 4 hard negatives are returned
    #   2. Others will be random negatives
    hard_sampler = GSHardEdgeDstNegativeSampler(num_negs, "hard_negative", sampler, num_hard_negs=8)
    check_less_hard_negs(hard_sampler, etype0, hard0, 4)

    # Case 4:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs == number of hard negatives required (10)
    # 4. number of hard negatives required (10) == number of hard negatives
    #    provided (hard1 has 10 negatives for each node)
    # 5.fast track
    #
    # expected behavior:
    #   1. Equal negatives
    def check_enough_hard_negs(hard_neg_sampler, target_etype, hard_neg_data):
        neg_src, neg_dst = hard_neg_sampler._generate(g, eids, target_etype)
        assert len(neg_src) == num_edges * num_negs
        assert len(neg_dst) == num_edges * num_negs
        assert_equal(th.repeat_interleave(src[:10], num_negs, 0).numpy(), neg_src.numpy())
        neg_dst = neg_dst.reshape(num_edges, num_negs)
        for i in range(num_edges):
            hard_neg_dst = set(neg_dst[i].tolist())
            hard_neg_set = set(hard_neg_data[i].tolist())
            assert hard_neg_dst == hard_neg_set
    hard_sampler = GSHardEdgeDstNegativeSampler(num_negs, "hard_negative", sampler, num_hard_negs=num_negs)
    check_enough_hard_negs(hard_sampler, etype1, hard1)

    # Case 5:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs == number of hard negatives required (10)
    # 4. number of hard negatives required (8) < number of hard negatives
    #    provided (hard2 has 20 negatives for each node)
    # 5.fast track
    #
    # expected behavior:
    #   1. hard negatives will be a subset of hard2
    def check_more_hard_negs(hard_neg_sampler, target_etype, hard_neg_data):
        neg_src, neg_dst = hard_neg_sampler._generate(g, eids, target_etype)
        assert len(neg_src) == num_edges * num_negs
        assert len(neg_dst) == num_edges * num_negs
        assert_equal(th.repeat_interleave(src[:10], num_negs, 0).numpy(), neg_src.numpy())
        neg_dst = neg_dst.reshape(num_edges, num_negs)
        for i in range(num_edges):
            hard_neg_dst = set(neg_dst[i].tolist())
            hard_neg_set = set(hard_neg_data[i].tolist())
            assert hard_neg_dst.issubset(hard_neg_set)
    check_more_hard_negs(hard_sampler, etype2, hard2)

    # Case 6:
    # hard_negative field is dict
    # num_hard_neg is dict
    # 3. num_negs > number of hard negatives required (2)
    # 4. num_negs > total number of hard negatives
    # provided (hard0 has 4 negatives for each node)
    # 5. fast track
    #
    # expected behavior:
    #   1. Only 2 hard negatives are returned
    #   2. Others will be random negatives
    hard_sampler = GSHardEdgeDstNegativeSampler(
        num_negs,
        {etype0: "hard_negative",
         etype1: "hard_negative",
         etype2: "hard_negative"},
        sampler,
        {etype0: 2,
         etype1: 2,
         etype2: 10})
    check_less_hard_negs(hard_sampler, etype0, hard0, 2)

    # Case 7:
    # hard_negative field is dict
    # num_hard_neg is dict
    # 3. num_negs > number of hard negatives required (2)
    # 4. num_negs == total number of hard negatives
    #    provided (hard1 has 10 negatives for each node)
    # 5. fast track
    #
    # expected behavior:
    #   1. Only 2 hard negatives are returned
    #   2. Others will be random negatives
    check_less_hard_negs(hard_sampler, etype1, hard1, 2)

    # Case 8:
    # 1. hard_negative field is string
    # 2. num_hard_neg is int
    # 3. num_negs == number of hard negatives required (10)
    # 4. number of hard negatives required (8) < number of hard negatives
    #    provided (hard2 has 20 negatives for each node)
    # 5.fast track
    #
    # expected behavior:
    #   1. hard negatives will be a subset of hard2
    check_more_hard_negs(hard_sampler, etype2, hard2)

    def check_none_hard_negs(hard_neg_sampler, target_etype, hard_neg_data):
        neg_src, neg_dst = hard_neg_sampler._generate(g, eids, target_etype)
        assert len(neg_src) == num_edges * num_negs
        assert len(neg_dst) == num_edges * num_negs
        assert_equal(th.repeat_interleave(src[:10], num_negs, 0).numpy(), neg_src.numpy())
        neg_dst = neg_dst.reshape(num_edges, num_negs)
        for i in range(num_edges):
            hard_neg_dst = set(neg_dst[i].tolist())
            hard_neg_set = set(hard_neg_data[i].tolist())
            assert hard_neg_dst.issubset(hard_neg_set) is False
    # Case 9:
    # dst_negative_field is not provided
    hard_sampler = GSHardEdgeDstNegativeSampler(
        num_negs, {}, sampler, 2)
    check_none_hard_negs(hard_sampler, etype2, hard2)

    # Case 10:
    # num_hard_negs is not provided
    hard_sampler = GSHardEdgeDstNegativeSampler(
        num_negs, "hard_negative", sampler, {})
    check_none_hard_negs(hard_sampler, etype2, hard2)

def test_edge_fixed_dst_negative_sample_gen_neg_pairs():
    # test GSHardEdgeDstNegativeSampler.gen_neg_pairs with fast track when all edges have enough predefined negatives
    num_nodes = 1000
    # test GSHardEdgeDstNegativeSampler._generate when all some pos edges do not have enough hard negatives defined
    num_negs = 10
    etype0, etype1, etype2, hard0, hard1, hard2, src, dst, g = _create_hard_neg_graph(num_nodes, num_negs)

    num_edges = 10

    def check_fixed_negs(pos_neg_tuple, etype, hard_neg_data):
        neg_src, _, pos_dst, neg_dst = pos_neg_tuple[etype]

        assert len(neg_src) == num_edges
        assert len(pos_dst) == num_edges
        assert neg_dst.shape[0] == num_edges
        assert_equal(src[:10].numpy(), neg_src.numpy())
        assert_equal(dst[:10].numpy(), pos_dst.numpy())

        assert_equal(hard_neg_data[:10].numpy(), neg_dst.numpy())

    hard_sampler = GSFixedEdgeDstNegativeSampler("hard_negative")
    pos_neg_tuple = hard_sampler.gen_etype_neg_pairs(g, etype0, th.arange(10))
    check_fixed_negs(pos_neg_tuple, etype0, hard0)
    pos_neg_tuple = hard_sampler.gen_etype_neg_pairs(g, etype1, th.arange(10))
    check_fixed_negs(pos_neg_tuple, etype1, hard1)
    pos_neg_tuple = hard_sampler.gen_etype_neg_pairs(g, etype2, th.arange(10))
    check_fixed_negs(pos_neg_tuple, etype2, hard2)

    hard_sampler = GSFixedEdgeDstNegativeSampler({etype0: "hard_negative",
                                          etype1: "hard_negative",
                                          etype2: "hard_negative"})
    pos_neg_tuple = hard_sampler.gen_etype_neg_pairs(g, etype0, th.arange(10))
    check_fixed_negs(pos_neg_tuple, etype0, hard0)
    pos_neg_tuple = hard_sampler.gen_etype_neg_pairs(g, etype1, th.arange(10))
    check_fixed_negs(pos_neg_tuple, etype1, hard1)
    pos_neg_tuple = hard_sampler.gen_etype_neg_pairs(g, etype2, th.arange(10))
    check_fixed_negs(pos_neg_tuple, etype2, hard2)

    # each positive edge should have enough fixed negatives
    hard0[0][-1] = -1
    fail = False
    try:
        pos_neg_tuple = hard_sampler.gen_neg_pairs(g, etype0, th.arange(10))
    except:
        fail = True
    assert fail


@pytest.mark.parametrize("num_pos", [2, 10])
@pytest.mark.parametrize("num_neg", [5, 20])
def test_inbatch_joint_neg_sampler(num_pos, num_neg):
    src = th.arange(num_pos)
    dst = th.arange(num_pos)
    g = dgl.heterograph({
        ("n0", "r0", "n1"): (src, dst),
    })
    sampler = InbatchJointUniform(num_neg)
    src, dst = sampler._generate(g, th.arange(num_pos), ("n0", "r0", "n1"))
    # In batch joint negative includes
    # uniform negatives + in-batch negatives.
    assert len(src) == num_pos * num_neg +  num_pos * (num_pos - 1)
    assert len(dst) == num_pos * num_neg + num_pos * (num_pos - 1)
    in_batch_src = src[-num_pos * (num_pos - 1):]
    in_batch_dst = dst[-num_pos * (num_pos - 1):]

    for i in range(num_pos):
        assert_equal(in_batch_src[i*(num_pos-1):(i+1)*(num_pos-1)].numpy(), np.repeat(i, (num_pos-1)))
        tmp_idx = np.ones(num_pos, dtype=bool)
        tmp_idx[i] = False
        assert_equal(in_batch_dst[i*(num_pos-1):(i+1)*(num_pos-1)].numpy(),
                     np.arange(num_pos)[tmp_idx])

def test_GSgnnTrainData_homogeneous():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    tr_ntypes = ["_N"]
    va_ntypes = ["_N"]

    with tempfile.TemporaryDirectory() as tmpdirname:
        # generate the test dummy homogeneous distributed graph and
        # test if it is possible to create GSgnnData on homogeneous graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                            dirname=tmpdirname,
                                                            is_homo=True)
        data = GSgnnData(part_config=part_config)
        _ = data.get_node_train_set(tr_ntypes)
        _ = data.get_node_val_set(va_ntypes)
        _ = data.get_node_test_set(va_ntypes)
        _ = data.get_node_infer_set(va_ntypes)

        # generate the test dummy distributed graph with "_N" node type. As it is expected to be
        # a homogeneous graph with "_N" as node type and ("_N", "_E", "_N") as edge type.
        # It should throw an error to clarify that.
        _, part_config = generate_dummy_dist_graph_homogeneous_failure_graph(graph_name='dummy',
                                                            dirname=tmpdirname)
        data = GSgnnData(part_config=part_config)
        try:
            _ = data.get_node_train_set(tr_ntypes)
            assert False, "expected Error raised for non-homogeneous graph input"
        except AssertionError as _:
            pass
        try:
            _ = data.get_node_val_set(va_ntypes)
            assert False, "expected Error raised for non-homogeneous graph input"
        except AssertionError as _:
            pass
        try:
            _ = data.get_node_test_set(va_ntypes)
            assert False, "expected Error raised for non-homogeneous graph input"
        except AssertionError as _:
            pass
        try:
            _ = data.get_node_infer_set(va_ntypes)
            assert False, "expected Error raised for non-homogeneous graph input"
        except AssertionError as _:
            pass

    tr_etypes = [("_N", "_E", "_N")]
    va_etypes = [("_N", "_E", "_N")]

    with tempfile.TemporaryDirectory() as tmpdirname:
        # generate the test dummy homogeneous distributed graph and
        # test if it is possible to create GSgnnData on homogeneous graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                            dirname=os.path.join(tmpdirname, 'dummy'),
                                                            is_homo=True)
        data = GSgnnData(part_config=part_config)
        _ = data.get_edge_train_set(tr_etypes)
        _ = data.get_edge_val_set(va_etypes)
        _ = data.get_edge_test_set(va_etypes)
        _ = data.get_edge_infer_set(va_etypes)

        # generate the test dummy distributed graph with "_N" node type. As it is expected to be
        # a homogeneous graph with "_N" as node type and ("_N", "_E", "_N") as edge type.
        # It should throw an error to clarify that.
        dist_graph, part_config = generate_dummy_dist_graph_homogeneous_failure_graph(graph_name='dummy',
                                                            dirname=os.path.join(tmpdirname, 'dummy'))
        data = GSgnnData(part_config=part_config)
        try:
            _ = data.get_edge_train_set(tr_etypes)
            assert False, "expected Error raised for non-homogeneous graph input"
        except AssertionError as _:
            pass
        try:
            _ = data.get_edge_val_set(va_etypes)
            assert False, "expected Error raised for non-homogeneous graph input"
        except AssertionError as _:
            pass
        try:
            _ = data.get_edge_test_set(va_etypes)
            assert False, "expected Error raised for non-homogeneous graph input"
        except AssertionError as _:
            pass
        try:
            _ = data.get_edge_infer_set(va_etypes)
            assert False, "expected Error raised for non-homogeneous graph input"
        except AssertionError as _:
            pass

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

def test_GSgnnMultiTaskDataLoader():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph_multi_task(graph_name='dummy',
                                                              dirname=tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # n0: train_mask: 2, val_mask: 2, test_maks: 4
        #     train_mask1: 5, val_mask1: 3, test_mask1: 7
        #     train_mask2: 50, val_mask2: 25, test_mask2: 25
        # n1: train_mask: 50, val_mask: 2, tset_mask:2
        #
        # ("n0", "r1", "n1"): train_mask: 2, val_mask: 2, test_maks: 4
        # ("n0", "r1", "n1"): train_mask1: 5, val_mask1: 3, test_maks2: 7
        # ("n0", "r1", "n1"): train_mask2: 2, val_mask2: 2, test_mask2: 4
        # ("n0", "r0", "n1"): train_mask: 50, val_mask: 2, test_maks: 4
        tast_info_n0_0 = TaskInfo(task_type=BUILTIN_TASK_NODE_CLASSIFICATION,
                                  task_id='tast_info_n0_0',
                                  task_config=None)
        tast_info_n0_1 = TaskInfo(task_type=BUILTIN_TASK_NODE_REGRESSION,
                                  task_id='tast_info_n0_1',
                                  task_config=None)
        tast_info_n0_2 = TaskInfo(task_type=BUILTIN_TASK_NODE_CLASSIFICATION,
                                  task_id='tast_info_n0_2',
                                  task_config=None)
        task_info_n1_0 = TaskInfo(task_type=BUILTIN_TASK_NODE_CLASSIFICATION,
                                  task_id='task_info_n1_0',
                                  task_config=None)
        task_info_edge_0 = TaskInfo(task_type=BUILTIN_TASK_LINK_PREDICTION,
                                  task_id='task_info_edge_0',
                                  task_config=None)
        task_info_edge_1 = TaskInfo(task_type=BUILTIN_TASK_EDGE_CLASSIFICATION,
                                  task_id='task_info_edge_1',
                                  task_config=None)
        task_info_edge_2 = TaskInfo(task_type=BUILTIN_TASK_EDGE_CLASSIFICATION,
                                  task_id='task_info_edge_2',
                                  task_config=None)
        task_infos = [tast_info_n0_0, tast_info_n0_1,
                      tast_info_n0_2, task_info_n1_0,
                      task_info_edge_0, task_info_edge_1,
                      task_info_edge_2]

        task_n0_0_dataloader = GSgnnNodeDataLoader(gdata,
                                                   gdata.get_node_train_set("n0", "train_mask"),
                                                   fanout=[10],
                                                   batch_size=2,
                                                   label_field="label",
                                                   train_task=True)
        task_n0_1_dataloader = GSgnnNodeDataLoader(gdata,
                                                   gdata.get_node_train_set("n0", "train_mask1"),
                                                   fanout=[10],
                                                   batch_size=2,
                                                   label_field="label",
                                                   train_task=True)
        task_n0_2_dataloader = GSgnnNodeDataLoader(gdata,
                                                   gdata.get_node_train_set("n0", "train_mask2"),
                                                   fanout=[10],
                                                   batch_size=5,
                                                   label_field="label",
                                                   train_task=True)
        task_n1_0_dataloader = GSgnnNodeDataLoader(gdata,
                                                   gdata.get_node_train_set("n1", "train_mask"),
                                                   fanout=[10],
                                                   batch_size=10,
                                                   label_field="label",
                                                   train_task=True)

        task_edge_0_dataloader = GSgnnLinkPredictionDataLoader(
            gdata,
            gdata.get_edge_train_set(etypes=[("n0", "r1", "n1"), ("n0", "r0", "n1")]),
            fanout=[10],
            batch_size=10,
            num_negative_edges=2,
            train_task=True
        )

        task_edge_1_dataloader = GSgnnEdgeDataLoader(
            gdata,
            gdata.get_edge_train_set(etypes=[("n0", "r1", "n1")],
                                     mask="train_mask1"),
            fanout=[10],
            batch_size=2,
            label_field="label",
            train_task=True,
            remove_target_edge_type=False
        )

        task_edge_2_dataloader = GSgnnEdgeDataLoader(
            gdata,
            gdata.get_edge_train_set(etypes=[("n0", "r1", "n1")],
                                     mask="train_mask2"),
            fanout=[10],
            batch_size=20,
            label_field="label",
            train_task=True,
            remove_target_edge_type=False
        )

        dataloaders = [task_n0_0_dataloader, task_n0_1_dataloader,
                       task_n0_2_dataloader, task_n1_0_dataloader,
                       task_edge_0_dataloader, task_edge_1_dataloader,
                       task_edge_2_dataloader]
        multi_dataloader = GSgnnMultiTaskDataLoader(gdata, task_infos, dataloaders)
        len_n0_0 = len(task_n0_0_dataloader)
        assert len_n0_0 == 1
        len_n0_1 = len(task_n0_1_dataloader)
        assert len_n0_1 == 3
        len_n0_2 = len(task_n0_2_dataloader)
        assert len_n0_2 == 10
        len_n1_0 = len(task_n1_0_dataloader)
        assert len_n1_0 == 5
        len_edge0_0 = len(task_edge_0_dataloader)
        assert len_edge0_0 == 6
        len_edge1_0 = len(task_edge_1_dataloader)
        assert len_edge1_0 == 3
        len_edge2_0 = len(task_edge_2_dataloader)
        assert len_edge2_0 == 5

        assert len(multi_dataloader) == 10

        len(multi_dataloader.dataloaders) == 7
        len(multi_dataloader.task_infos) == 7

        dataloaders = multi_dataloader.dataloaders
        task_infos = multi_dataloader.task_infos
        for dataloader, task_info in zip(dataloaders, task_infos):
            if task_info.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
                assert isinstance(dataloader, GSgnnNodeDataLoader)

            if task_info.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION]:
                assert isinstance(dataloader, GSgnnEdgeDataLoader)
            if task_info.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
                assert isinstance(dataloader, GSgnnLinkPredictionDataLoader)

        iter_num = 0
        n0_1_seeds_cnt = th.tensor([4] * 5) # check whether n0_1 datalaoder iterates the whole datasets 3 times.
        edge0_seeds_cnt = th.tensor([2] * 50) # check whether edge0 datalaoder iterates the whole datasets 1 and half times.
        for mini_batches in multi_dataloader:
            assert len(mini_batches) == 7
            for task_info, mini_batch in mini_batches:
                if task_info.task_id == "tast_info_n0_0":
                    assert task_info.dataloader == task_n0_0_dataloader
                    _, seeds, blocks = mini_batch
                    assert len(seeds["n0"]) == 2
                    assert len(blocks) == 1
                    assert set(seeds["n0"].tolist()).issubset({0,1})
                if task_info.task_id == "tast_info_n0_1":
                    assert task_info.dataloader == task_n0_1_dataloader
                    _, seeds, blocks = mini_batch
                    assert len(seeds["n0"]) == 1 if (iter_num % 3 == 2) else 2
                    assert len(blocks) == 1
                    assert set(seeds["n0"].tolist()).issubset({0,1,2,3,4})
                    n0_1_seeds_cnt[seeds["n0"]] =  n0_1_seeds_cnt[seeds["n0"]] - 1
                if task_info.task_id == "tast_info_n0_2":
                    assert task_info.dataloader == task_n0_2_dataloader
                    _, seeds, blocks = mini_batch
                    assert len(seeds["n0"]) == 5
                    assert len(blocks) == 1
                    assert np.all(seeds["n0"].numpy() >= SIZE_DICT['tiny'] // 2)
                if task_info.task_id == "task_info_n1_0":
                    assert task_info.dataloader == task_n1_0_dataloader
                    _, seeds, blocks = mini_batch
                    assert len(seeds["n1"]) == 10
                    assert len(blocks) == 1
                    assert np.all(seeds["n1"].numpy() < SIZE_DICT['tiny'] // 2)
                if task_info.task_id == "task_info_edge_0":
                    assert task_info.dataloader == task_edge_0_dataloader
                    _, pos_graph, _, blocks = mini_batch
                    assert ("n0", "r1", "n1") in pos_graph.canonical_etypes or \
                        ("n0", "r0", "n1") in pos_graph.canonical_etypes
                    assert len(blocks) == 1
                    num_edges = 0
                    if ("n0", "r1", "n1") in pos_graph.canonical_etypes:
                        eids = pos_graph.edges[("n0", "r1", "n1")].data[dgl.EID]
                        assert set(eids.tolist()).issubset({0,1})
                        num_edges += len(eids)
                    eids = pos_graph.edges[("n0", "r0", "n1")].data[dgl.EID]
                    num_edges += len(eids)
                    assert num_edges == 10 if iter_num != 5 else 2
                    assert np.any(eids.numpy() < SIZE_DICT['tiny'] // 2)
                    edge0_seeds_cnt[eids] = edge0_seeds_cnt[eids] - 1
                if task_info.task_id == "task_info_edge_1":
                    assert task_info.dataloader == task_edge_1_dataloader
                    _, batch_graph, blocks = mini_batch
                    assert len(batch_graph.canonical_etypes) == 1
                    assert ("n0", "r1", "n1") in batch_graph.canonical_etypes
                    assert batch_graph.num_edges(("n0", "r1", "n1")) == 1 if (iter_num % 3 == 2) else 2
                    assert len(blocks) == 1
                    eids = batch_graph.edges[("n0", "r1", "n1")].data[dgl.EID]
                    assert set(eids.tolist()).issubset({0,1,2,3,4})
                if task_info.task_id == "task_info_edge_2":
                    assert task_info.dataloader == task_edge_2_dataloader
                    _, batch_graph, blocks = mini_batch
                    assert len(batch_graph.canonical_etypes) == 1
                    assert ("n0", "r1", "n1") in batch_graph.canonical_etypes
                    assert batch_graph.num_edges(("n0", "r1", "n1")) == 20
                    assert len(blocks) == 1
                    eids = batch_graph.edges[("n0", "r1", "n1")].data[dgl.EID]
                    assert np.any(eids.numpy() >= SIZE_DICT['tiny'])
            iter_num += 1
        assert iter_num == 9
        assert np.any(n0_1_seeds_cnt.numpy() <= 1)
        assert np.any(n0_1_seeds_cnt.numpy() >= 0)
        assert np.any(edge0_seeds_cnt.numpy() <= 1)
        assert np.any(edge0_seeds_cnt.numpy() >= 0)

def run_dist_small_val_test(part_config, worker_rank, world_size):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=worker_rank)
    dgl.distributed.initialize('')
    gs_data = GSgnnData(part_config=part_config)

    @patch("dgl.distributed.edge_split")
    @patch("dgl.distributed.node_split")
    def check_val_test(mock_node_split, mock_edge_split):
        #######
        # Node mask
        total_idx = th.tensor([1,2,3,4])
        split_ret = total_idx[:2] if worker_rank == 0 else total_idx[2:]
        mock_node_split.side_effect = [split_ret, split_ret]
        # Mocked val has 4 labeled data
        # each worker will get 2
        ntype = 'n1'
        idx = gs_data.get_node_val_set(ntype, mask="val_mask")
        assert len(idx[ntype]) == 2
        # Mocked test has 4 labeled data
        # each worker will get 2
        idx = gs_data.get_node_test_set(ntype, mask="test_mask")
        assert len(idx[ntype]) == 2

        total_idx = th.tensor([1])
        split_ret = total_idx[:1] if worker_rank == 0 else total_idx[1:1]
        mock_node_split.side_effect = [split_ret, split_ret]
        # Mocked val has only 1 labeled data
        # Thus, both worker 0 and worker 1 will
        # take the same validation set.
        idx = gs_data.get_node_val_set(ntype, mask="val_mask2")
        assert len(idx[ntype]) == 1
        assert th.nonzero(gs_data._g.nodes[ntype].data['val_mask2'][th.arange(gs_data._g.num_nodes(ntype))]).reshape(-1,)[0].item() == idx[ntype][0].item()

        # Mocked test has only 1 labeled data
        # Thus, both worker 0 and worker 1 will
        # take the same test set.
        idx = gs_data.get_node_test_set(ntype, mask="test_mask2")
        assert len(idx[ntype]) == 1
        assert th.nonzero(gs_data._g.nodes[ntype].data['test_mask2'][th.arange(gs_data._g.num_nodes(ntype))]).reshape(-1,)[0] == idx[ntype][0]

        #######
        # Edge mask
        etype = ("n0", "r1", "n1")
        total_idx = th.tensor([1,2,3,4])
        split_ret = total_idx[:2] if worker_rank == 0 else total_idx[2:]
        mock_edge_split.side_effect = [split_ret, split_ret]

        # Mocked val has two labeled data
        # each worker will get 1
        idx = gs_data.get_edge_val_set(etype, mask="val_mask")
        assert len(idx[etype]) == 2
        # Mocked test has two labeled data
        # each worker will get 1
        idx = gs_data.get_edge_test_set(etype, mask="test_mask")
        assert len(idx[etype]) == 2

        total_idx = th.tensor([1])
        split_ret = total_idx[:1] if worker_rank == 0 else total_idx[1:1]
        mock_edge_split.side_effect = [split_ret, split_ret]
        # Mocked val has only 1 labeled data
        # Thus, both worker 0 and worker 1 will
        # take the same validation set.
        idx = gs_data.get_edge_val_set(etype, mask="val_mask2")
        assert len(idx[etype]) == 1
        assert th.nonzero(gs_data._g.edges[etype].data['val_mask2'][th.arange(gs_data._g.num_edges(etype))]).reshape(-1,)[0].item() == idx[etype][0].item()

        # mocked test has only 1 labeled data
        # Thus, both worker 0 and worker 1 will
        # take the same test set.
        idx = gs_data.get_edge_test_set(etype, mask="test_mask2")
        assert len(idx[etype]) == 1
        assert th.nonzero(gs_data._g.edges[etype].data['test_mask2'][th.arange(gs_data._g.num_edges(etype))]).reshape(-1,)[0] == idx[etype][0]

    check_val_test()

    if worker_rank == 0:
        th.distributed.destroy_process_group()

def test_GSgnnTranData_small_val_test():
    with tempfile.TemporaryDirectory() as tmpdirname:
        _, part_config = generate_dummy_dist_graph(tmpdirname)

        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_dist_small_val_test,
                            args=(part_config, 0, 2))
        p1 = ctx.Process(target=run_dist_small_val_test,
                        args=(part_config, 1, 2))

        p0.start()
        p1.start()
        p0.join()
        p1.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0

if __name__ == '__main__':
    test_GSgnnTranData_small_val_test()
    test_GSgnnLinkPredictionTestDataLoader(1, 1)
    test_GSgnnLinkPredictionTestDataLoader(10, 20)
    test_GSgnnMultiTaskDataLoader()
    test_GSgnnLinkPredictionPredefinedTestDataLoader(1)
    test_GSgnnLinkPredictionPredefinedTestDataLoader(10)
    test_edge_fixed_dst_negative_sample_gen_neg_pairs()
    test_hard_edge_dst_negative_sample_generate_complex_case()
    test_hard_edge_dst_negative_sample_generate()
    test_inbatch_joint_neg_sampler(10, 20)

    test_np_dataloader_len(11)
    test_ep_dataloader_len(11)
    test_lp_dataloader_len(11)

    test_np_dataloader_trim_data(GSgnnNodeDataLoader)
    test_edge_dataloader_trim_data(GSgnnLinkPredictionDataLoader)
    test_edge_dataloader_trim_data(FastGSgnnLinkPredictionDataLoader)
    test_GSgnnData()
    test_GSgnnData2()
    test_lp_dataloader()
    test_edge_dataloader()
    test_node_dataloader()
    test_node_dataloader_reconstruct()
    test_GSgnnAllEtypeLinkPredictionDataLoader(10)
    test_GSgnnAllEtypeLinkPredictionDataLoader(1)
    test_GSgnnLinkPredictionJointTestDataLoader(1, 1)
    test_GSgnnLinkPredictionJointTestDataLoader(10, 20)

    test_prepare_input()
    test_modify_fanout_for_target_etype()

    test_distill_sampler_get_file(num_files=7)
    test_DistillDistributedFileSampler(num_files=7, is_train=True, \
        infinite=False, shuffle=True)
    test_DistillDataloaderGenerator("gloo", 7, True)

    test_GSgnnTrainData_homogeneous()
    test_np_dataloader_trim_data_device(GSgnnNodeDataLoader, 'gloo')
    test_np_dataloader_trim_data_device(GSgnnNodeDataLoader, 'nccl')
    test_edge_dataloader_trim_data_device(GSgnnLinkPredictionDataLoader, 'gloo')
    test_edge_dataloader_trim_data_device(GSgnnEdgeDataLoader, 'nccl')
