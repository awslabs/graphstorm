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
"""

from pathlib import Path
import os
import yaml
import tempfile
import dgl
from argparse import Namespace
import torch as th
import numpy as np
from unittest.mock import patch

from graphstorm.config import (GSConfig, TaskInfo)
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                                BUILTIN_TASK_NODE_REGRESSION,
                                BUILTIN_TASK_EDGE_CLASSIFICATION,
                                BUILTIN_TASK_EDGE_REGRESSION,
                                BUILTIN_TASK_LINK_PREDICTION,
                                BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                                BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT)
from graphstorm.dataloading import GSgnnData, GSgnnMultiTaskDataLoader
from graphstorm.eval.evaluator import GSgnnLPRankingEvalInterface, GSgnnMultiTaskEvaluator
from graphstorm.tracker import GSSageMakerTaskTracker
from graphstorm import (create_builtin_node_gnn_model,
                        create_builtin_edge_gnn_model,
                        create_builtin_lp_gnn_model)
from graphstorm.eval import GSgnnClassificationEvaluator, GSgnnLPEvaluator
from graphstorm.utils import setup_device, get_device
from graphstorm.trainer.mt_trainer import (GSgnnMultiTaskLearningTrainer,
                                           prepare_node_mini_batch,
                                           prepare_edge_mini_batch,
                                           prepare_link_predict_mini_batch,
                                           prepare_reconstruct_node_feat,
                                           prepare_reconstruct_edge_feat)
from graphstorm.trainer import (GSgnnTrainer,
                                GSgnnNodePredictionTrainer,
                                GSgnnEdgePredictionTrainer,
                                GSgnnLinkPredictionTrainer)
from graphstorm.dataloading import (GSgnnNodeDataLoader,
                                    GSgnnEdgeDataLoader,
                                    GSgnnLinkPredictionDataLoader,
                                    GSgnnLinkPredictionTestDataLoader)
from graphstorm.model import GSgnnMultiTaskModelInterface, GSgnnModel
from numpy.testing import assert_equal, assert_raises

from util import (DummyGSgnnEncoderModel,
                  DummyGSgnnMTModel,
                  DummyGSgnnNodeDataLoader,
                  DummyGSgnnEdgeDataLoader,
                  DummyGSgnnLinkPredictionDataLoader)
from data_utils import generate_dummy_dist_graph


def create_nc_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "node_feat_name": ["feat"],
                "model_encoder_type": "rgat",
            },
            "gnn": {
                "num_layers": 1,
                "hidden_size": 4,
                "lr": 0.001,
                "norm": "layer"
            },
            "input": {},
            "output": {},
            "rgat": {
            },
            "node_classification": {
                "num_classes": 2,
                "target_ntype": "n0",
            },
        }
    }
    
    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

def test_trainer_setup_evaluator():

    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        create_nc_config(Path(tmpdirname), 'gnn_nc.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_builtin_node_gnn_model(g, config, True)
    trainer = GSgnnTrainer(model)

    # case 1: by default trainer has no task_tracker
    assert trainer.task_tracker is None

    evaluator = GSgnnClassificationEvaluator(config.eval_frequency,
                                             config.eval_metric,
                                             config.multilabel,
                                             config.use_early_stop)

    # case 2: evaluator has no task_tracker by default
    assert evaluator.task_tracker is None

    # case 3: when setup an evaluator that has no task_tracker and train has no task tracker
    #         eitehr, create a new task_tracker and set it to the evaluator.
    trainer.setup_evaluator(evaluator)

    assert trainer.task_tracker is not None
    assert evaluator.eval_frequency == trainer.task_tracker.log_report_frequency
    assert evaluator.task_tracker == trainer.task_tracker

    # case 4: when setup an evaluator that has no task_tracker, but train has a task tracker,
    #         use the trainer's task_tracker to setup the evaluator.
    trainer.setup_task_tracker(GSSageMakerTaskTracker(10))
    evaluator.setup_task_tracker(None)
    trainer.setup_evaluator(evaluator)

    assert evaluator.task_tracker == trainer.task_tracker
    assert evaluator.task_tracker.log_report_frequency == 10

    # case 5: when setup an evaluator that has a task_tracker, no change of the evaluator.
    evaluator.setup_task_tracker(GSSageMakerTaskTracker(100))
    trainer.setup_evaluator(evaluator)

    assert evaluator.task_tracker != trainer.task_tracker
    assert evaluator.task_tracker.log_report_frequency == 100
    assert trainer.task_tracker.log_report_frequency != 100

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

class DummyGSgnnMultiTaskSharedEncoderModel(GSgnnModel, GSgnnMultiTaskModelInterface):
    """ Dummy GSgnnMultiTaskSharedEncoderModel for testing
    """
    def __init__(self, task_id, task_type, input_nodes, labels, node_feats, expected_loss):
        self.task_id = task_id
        self.task_type = task_type
        self.input_nodes = input_nodes
        self.labels = labels
        self.node_feats = node_feats
        self.expected_loss = expected_loss

    def forward(self, task_id, mini_batch):
        assert task_id == self.task_id
        assert len(mini_batch) == 2
        encoder_data, decoder_data = mini_batch

        if self.task_type == BUILTIN_TASK_NODE_CLASSIFICATION:
            assert len(encoder_data) == 4
            blocks, node_feats, _, input_nodes = encoder_data
            lbl = decoder_data
            assert blocks is None
            assert_equal(lbl.numpy(), self.labels.numpy())
            for ntype, idx in input_nodes.items():
                assert_equal(idx.numpy(), self.input_nodes[ntype].numpy())

            for ntype, feats in node_feats.items():
                assert_equal(feats.numpy(), self.node_feats[ntype].numpy())

            return self.expected_loss
        if self.task_type == BUILTIN_TASK_EDGE_REGRESSION:
            assert len(encoder_data) == 4
            blocks, node_feats, _, input_nodes = encoder_data
            assert blocks is None
            for ntype, idx in input_nodes.items():
                assert_equal(idx.numpy(), self.input_nodes[ntype].numpy())

            for ntype, feats in node_feats.items():
                assert_equal(feats.numpy(), self.node_feats[ntype].numpy())
            assert len(decoder_data) == 3
            batch_graph, edge_decoder_feats, lbl = decoder_data
            assert batch_graph is None
            assert edge_decoder_feats is None
            assert_equal(lbl.numpy(), self.labels.numpy())

            return self.expected_loss
        if self.task_type == BUILTIN_TASK_LINK_PREDICTION:
            assert len(encoder_data) == 4
            blocks, node_feats, _, input_nodes = encoder_data
            assert blocks is None
            for ntype, idx in input_nodes.items():
                assert_equal(idx.numpy(), self.input_nodes[ntype].numpy())

            for ntype, feats in node_feats.items():
                assert_equal(feats.numpy(), self.node_feats[ntype].numpy())

            pos_graph, neg_graph, pos_graph_feats, _ = decoder_data
            assert pos_graph is None
            assert neg_graph is None
            assert pos_graph_feats is None

            return self.expected_loss

        assert False

    def predict(self, task_id, mini_batch, return_proba=False):
        pass

def test_mtask_prepare_reconstruct_edge_feat():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnData(part_config=part_config)

    setup_device(0)
    device = get_device()

    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}

    task_id = "test_edge_feat_reconstruct"

    # label is same as edge label
    dataloader = GSgnnEdgeDataLoader(ep_data,
                                     target_idx, [10], 10,
                                     label_field='label',
                                     node_feats='feat',
                                     train_task=False,
                                     remove_target_edge_type=False)
    task_config = GSConfig.__new__(GSConfig)
    setattr(task_config, "task_weight", 0.75)
    task_info = TaskInfo(task_type=BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT,
                         task_id=task_id,
                         task_config=task_config,
                         dataloader=dataloader)
    input_node_idx = {
        "n0": th.arange(10),
        "n1": th.arange(20),
    }
    node_feats = ep_data.get_node_feats(input_node_idx, 'feat')
    labels = ep_data.get_edge_feats(target_idx, 'label')
    batch_graph = dgl.heterograph(
        {('n0', 'r1', 'n1'): (th.randint(g.number_of_nodes("n0"), (g.number_of_edges('r1'),)),
                              th.randint(g.number_of_nodes("n1"), (g.number_of_edges('r1'),)))}
    )
    batch_graph.edges[('n0', 'r1', 'n1')].data[dgl.EID] = th.arange(ep_data.g.number_of_edges('r1'))
    mini_batch = (input_node_idx, batch_graph, None)

    blocks, edge_graph, input_feats, _, \
        edge_decoder_feats, lbl, input_nodes = \
            prepare_reconstruct_edge_feat(ep_data, task_info, mini_batch, device)
    assert blocks is None
    assert edge_decoder_feats is None
    assert edge_graph.number_of_edges('r1') == batch_graph.number_of_edges('r1')
    assert_equal(input_nodes["n0"].numpy(), input_node_idx["n0"].numpy())
    assert_equal(input_nodes["n1"].numpy(), input_node_idx["n1"].numpy())
    assert_equal(input_feats["n0"].cpu().numpy(), node_feats["n0"].numpy())
    assert_equal(input_feats["n1"].cpu().numpy(), node_feats["n1"].numpy())
    assert_equal(lbl[('n0', 'r1', 'n1')].cpu().numpy(), labels[('n0', 'r1', 'n1')].numpy())

    # there is no node feat
    dataloader = GSgnnEdgeDataLoader(ep_data,
                                     target_idx, [10], 10,
                                     label_field='label',
                                     train_task=False,
                                     remove_target_edge_type=False)
    task_info = TaskInfo(task_type=BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT,
                         task_id=task_id,
                         task_config=task_config,
                         dataloader=dataloader)
    _, _, input_feats, _, \
        _, lbl, input_nodes = \
            prepare_reconstruct_edge_feat(ep_data, task_info, mini_batch, device)
    assert_equal(input_nodes["n0"].numpy(), input_node_idx["n0"].numpy())
    assert_equal(input_nodes["n1"].numpy(), input_node_idx["n1"].numpy())
    assert len(input_feats) == 0
    assert_equal(lbl[('n0', 'r1', 'n1')].cpu().numpy(), labels[('n0', 'r1', 'n1')].numpy())

def test_mtask_prepare_reconstruct_node_feat():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        np_data = GSgnnData(part_config=part_config)

    setup_device(0)
    device = get_device()
    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {'n1': th.arange(np_data.g.number_of_nodes('n1'))}
    task_id = "test_node_feat_reconstruct"

    # label is same as node feat
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10,
                                     label_field='feat',
                                     node_feats='feat',
                                     train_task=False)
    task_config = GSConfig.__new__(GSConfig)
    setattr(task_config, "task_weight", 0.75)
    task_info = TaskInfo(task_type=BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                         task_id=task_id,
                         task_config=task_config,
                         dataloader=dataloader)
    node_feats = np_data.get_node_feats(target_idx, 'feat')
    labels = np_data.get_node_feats(target_idx, 'feat')
    mini_batch = (target_idx, target_idx, None)

    blocks, input_feats, _, lbl, input_nodes = \
        prepare_reconstruct_node_feat(np_data, task_info, mini_batch, device)
    assert blocks is None
    assert_equal(input_nodes["n1"].numpy(), target_idx["n1"].numpy())
    assert_equal(input_feats["n1"].cpu().numpy(), node_feats["n1"].numpy())
    assert_equal(lbl["n1"].cpu().numpy(), labels["n1"].numpy())
    assert_equal(node_feats["n1"].cpu().numpy(), lbl["n1"].cpu().numpy())

    # there is no node feat
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10,
                                     label_field='feat',
                                     train_task=False)
    task_info = TaskInfo(task_type=BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                         task_id=task_id,
                         task_config=task_config,
                         dataloader=dataloader)
    _, input_feats, _, lbl, input_nodes = \
        prepare_reconstruct_node_feat(np_data, task_info, mini_batch, device)
    assert_equal(input_nodes["n1"].numpy(), target_idx["n1"].numpy())
    assert len(input_feats) == 0
    assert_equal(lbl["n1"].cpu().numpy(), labels["n1"].numpy())
    assert_equal(node_feats["n1"].cpu().numpy(), lbl["n1"].cpu().numpy())

def test_mtask_prepare_node_mini_batch():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        np_data = GSgnnData(part_config=part_config)

    setup_device(0)
    device = get_device()
    # Without shuffling, the seed nodes should have the same order as the target nodes.
    target_idx = {'n1': th.arange(np_data.g.number_of_nodes('n1'))}
    task_id = "test_node_prediction"
    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10,
                                     label_field='label',
                                     node_feats='feat',
                                     train_task=False)
    task_config = GSConfig.__new__(GSConfig)
    setattr(task_config, "task_weight", 0.75)
    task_info = TaskInfo(task_type=BUILTIN_TASK_NODE_CLASSIFICATION,
                         task_id=task_id,
                         task_config=task_config,
                         dataloader=dataloader)
    node_feats = np_data.get_node_feats(target_idx, 'feat')
    labels = np_data.get_node_feats(target_idx, 'label')
    mini_batch = (target_idx, target_idx, None)

    blocks, input_feats, _, lbl, input_nodes = \
        prepare_node_mini_batch(np_data, task_info, mini_batch, device)
    assert blocks is None
    assert_equal(input_nodes["n1"].numpy(), target_idx["n1"].numpy())
    assert_equal(input_feats["n1"].cpu().numpy(), node_feats["n1"].numpy())
    assert_equal(lbl["n1"].cpu().numpy(), labels["n1"].numpy())

    dataloader = GSgnnNodeDataLoader(np_data, target_idx, [10], 10,
                                     label_field='label',
                                     train_task=False)
    task_info = TaskInfo(task_type=BUILTIN_TASK_NODE_CLASSIFICATION,
                         task_id=task_id,
                         task_config=task_config,
                         dataloader=dataloader)
    _, input_feats, _, lbl, input_nodes = \
        prepare_node_mini_batch(np_data, task_info, mini_batch, device)
    assert_equal(input_nodes["n1"].numpy(), target_idx["n1"].numpy())
    assert len(input_feats) == 0
    assert_equal(lbl["n1"].cpu().numpy(), labels["n1"].numpy())

def test_mtask_prepare_edge_mini_batch():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnData(part_config=part_config)

    setup_device(0)
    device = get_device()

    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    task_id = "test_edge_prediction"
    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10,
                                     node_feats='feat',
                                     label_field='label',
                                     train_task=True, remove_target_edge_type=False)

    task_config = GSConfig.__new__(GSConfig)
    setattr(task_config, "task_weight", 0.71)
    task_info = TaskInfo(task_type=BUILTIN_TASK_EDGE_REGRESSION,
                         task_id=task_id,
                         task_config=task_config,
                         dataloader=dataloader)
    input_node_idx = {
        "n0": th.arange(10),
        "n1": th.arange(20),
    }
    node_feats = ep_data.get_node_feats(input_node_idx, 'feat')
    labels = ep_data.get_edge_feats(target_idx, 'label')
    batch_graph = dgl.heterograph(
        {('n0', 'r1', 'n1'): (th.randint(g.number_of_nodes("n0"), (g.number_of_edges('r1'),)),
                              th.randint(g.number_of_nodes("n1"), (g.number_of_edges('r1'),)))}
    )
    batch_graph.edges[('n0', 'r1', 'n1')].data[dgl.EID] = th.arange(ep_data.g.number_of_edges('r1'))
    mini_batch = (input_node_idx, batch_graph, None)
    blocks, edge_graph, input_feats, _, \
        edge_decoder_feats, lbl, input_nodes = prepare_edge_mini_batch(ep_data, task_info, mini_batch, device)

    assert blocks is None
    assert edge_decoder_feats is None
    assert edge_graph.number_of_edges('r1') == batch_graph.number_of_edges('r1')
    assert_equal(input_nodes["n0"].numpy(), input_node_idx["n0"].numpy())
    assert_equal(input_nodes["n1"].numpy(), input_node_idx["n1"].numpy())
    assert_equal(input_feats["n0"].cpu().numpy(), node_feats["n0"].numpy())
    assert_equal(input_feats["n1"].cpu().numpy(), node_feats["n1"].numpy())
    assert_equal(lbl[('n0', 'r1', 'n1')].cpu().numpy(), labels[('n0', 'r1', 'n1')].numpy())

    dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, [10], 10,
                                     label_field='label',
                                     train_task=True, remove_target_edge_type=False)
    task_info = TaskInfo(task_type=BUILTIN_TASK_EDGE_REGRESSION,
                         task_id=task_id,
                         task_config=task_config,
                         dataloader=dataloader)
    _, _, input_feats, _, \
        _, lbl, input_nodes = prepare_edge_mini_batch(ep_data, task_info, mini_batch, device)
    assert_equal(input_nodes["n0"].numpy(), input_node_idx["n0"].numpy())
    assert_equal(input_nodes["n1"].numpy(), input_node_idx["n1"].numpy())
    assert len(input_feats) == 0
    assert_equal(lbl[('n0', 'r1', 'n1')].cpu().numpy(), labels[('n0', 'r1', 'n1')].numpy())

def test_mtask_prepare_lp_mini_batch():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy', dirname=tmpdirname)
        ep_data = GSgnnData(part_config=part_config)

    setup_device(0)
    device = get_device()

    target_idx = {('n0', 'r1', 'n1'): th.arange(ep_data.g.number_of_edges('r1'))}
    task_id = "test_link_prediction"
    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx,
                                               [10], 10,
                                               num_negative_edges=2,
                                               node_feats='feat',
                                               train_task=False)
    task_config = GSConfig.__new__(GSConfig)
    setattr(task_config, "task_weight", 0.72)
    task_info = TaskInfo(task_type=BUILTIN_TASK_LINK_PREDICTION,
                         task_id=task_id,
                         task_config=task_config,
                         dataloader=dataloader)
    input_node_idx = {
        "n0": th.arange(10),
        "n1": th.arange(20),
    }
    node_feats = ep_data.get_node_feats(input_node_idx, 'feat')

    input_pos_graph = dgl.heterograph(
        {('n0', 'r1', 'n1'): (th.tensor([0,1]),
                              th.tensor([1,2]))})
    input_neg_graph = dgl.heterograph(
        {('n0', 'r1', 'n1'): (th.tensor([0,1]),
                              th.tensor([1,2]))})

    mini_batch = (input_node_idx, input_pos_graph, input_neg_graph, None)

    blocks, pos_graph, neg_graph, input_feats, _, \
        pos_graph_feats, _, input_nodes = \
            prepare_link_predict_mini_batch(ep_data, task_info, mini_batch, device)

    assert blocks is None
    assert_equal(input_nodes["n0"].numpy(), input_node_idx["n0"].numpy())
    assert_equal(input_nodes["n1"].numpy(), input_node_idx["n1"].numpy())
    assert_equal(input_feats["n0"].cpu().numpy(), node_feats["n0"].numpy())
    assert_equal(input_feats["n1"].cpu().numpy(), node_feats["n1"].numpy())
    assert pos_graph_feats is None
    assert input_pos_graph.number_of_edges('r1') == pos_graph.number_of_edges('r1')
    assert input_neg_graph.number_of_edges('r1') == neg_graph.number_of_edges('r1')

    dataloader = GSgnnLinkPredictionDataLoader(ep_data, target_idx,
                                               [10], 10,
                                               num_negative_edges=2,
                                               train_task=False)
    task_info = TaskInfo(task_type=BUILTIN_TASK_LINK_PREDICTION,
                         task_id=task_id,
                         task_config=task_config,
                         dataloader=dataloader)
    _, _, _, input_feats, _, \
        _, _, input_nodes = \
            prepare_link_predict_mini_batch(ep_data, task_info, mini_batch, device)
    assert len(input_feats) == 0
    assert_equal(input_nodes["n0"].numpy(), input_node_idx["n0"].numpy())
    assert_equal(input_nodes["n1"].numpy(), input_node_idx["n1"].numpy())

class MTaskCheckerEvaluator(GSgnnMultiTaskEvaluator):
    def __init__(self, val_rankings, test_rankings, total_iters):
        self._val_results = val_rankings
        self._test_results = test_rankings
        self._steps = total_iters

    def evaluate(self, val_results, test_results, total_iters, **kwargs):
        assert self._steps == total_iters
        def compare_results(target_res, check_res):
            assert len(target_res) == len(check_res)
            for task_id, target_r in target_res.items():
                assert task_id in check_res
                check_r = check_res[task_id]
                if isinstance(target_r, tuple):
                    # prediction tasks
                    tr_1, tr_2 = target_r
                    cr_1, cr_2 = check_r
                    assert_equal(tr_1, cr_1)
                    assert_equal(tr_2, cr_2)
                else:
                    # In case LP results also returned candidate list
                    # lengths, we check the lengths and values
                    if isinstance(check_r, tuple):
                        check_r, candidate_sizes = check_r
                        if candidate_sizes.shape[0] > 1:
                            assert check_r.shape[0] == candidate_sizes.shape[0], \
                                ("ranking and candidate_sizes must have the same length, "
                                f"got {check_r.shape=} {candidate_sizes.shape=}" )
                            assert th.all(check_r <= candidate_sizes).item(), \
                                "all ranks must be <= candidate_sizes"
                    assert_equal(target_r, check_r)

        if self._val_results is not None:
            compare_results(self._val_results, val_results)
        if self._test_results is not None:
            compare_results(self._test_results, test_results)
        return None, None

    @property
    def task_tracker(self):
        return "dummy tracker"

def test_mtask_eval():
    task_info_nc = TaskInfo(task_type=BUILTIN_TASK_NODE_CLASSIFICATION,
                            task_id='nc_task',
                            task_config=None)
    nc_dataloader = DummyGSgnnNodeDataLoader()
    task_info_nr = TaskInfo(task_type=BUILTIN_TASK_NODE_REGRESSION,
                            task_id='nr_task',
                            task_config=None)
    nr_dataloader = DummyGSgnnNodeDataLoader()
    task_info_ec = TaskInfo(task_type=BUILTIN_TASK_EDGE_CLASSIFICATION,
                            task_id='ec_task',
                            task_config=None)
    ec_dataloader = DummyGSgnnEdgeDataLoader()
    task_info_er = TaskInfo(task_type=BUILTIN_TASK_EDGE_REGRESSION,
                            task_id='er_task',
                            task_config=None)
    er_dataloader = DummyGSgnnEdgeDataLoader()
    task_config = GSConfig.__new__(GSConfig)
    setattr(task_config, "train_mask", "train_mask")
    task_info_lp = TaskInfo(task_type=BUILTIN_TASK_LINK_PREDICTION,
                            task_id='lp_task',
                            task_config=task_config)

    encoder_model = DummyGSgnnEncoderModel()
    model = DummyGSgnnMTModel(encoder_model, decoders={task_info_lp.task_id: "dummy"}, has_sparse=True)

    mt_trainer = GSgnnMultiTaskLearningTrainer(model)
    mt_trainer._device = 'cpu'

    lp_dataloader = DummyGSgnnLinkPredictionDataLoader()
    task_info_nfr = TaskInfo(task_type=BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                            task_id='nfr_task',
                            task_config=None)
    nfr_dataloader = DummyGSgnnNodeDataLoader()
    tast_info_efr = TaskInfo(task_type=BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT,
                            task_id='efr_task',
                            task_config=None)
    efr_dataloader = DummyGSgnnEdgeDataLoader()

    task_infos = [task_info_nc, task_info_nr, task_info_ec,
                  task_info_er, task_info_lp,
                  task_info_nfr, tast_info_efr]

    data = None
    res = mt_trainer.eval(model, data, None, None, 100)
    assert res is None

    def mock_func_do_mini_batch_inference(*args, **kwargs):
        return None

    def mock_func_do_full_graph_inference(*args, **kwargs):
        return None

    def mock_func_run_lp_mini_batch_predict(*args, **kwargs):
        return lp_res

    ntask_res = (np.arange(10), np.arange(10))
    etask_res = (np.arange(20), np.arange(20))
    lp_res = np.arange(5)
    def mock_func_multi_task_mini_batch_predict(model, emb, dataloaders, task_infos, device, return_proba, return_label):
        res = {}
        for dataloader, task_info in zip(dataloaders, task_infos):
            if task_info.task_type in \
            [BUILTIN_TASK_NODE_CLASSIFICATION,
             BUILTIN_TASK_NODE_REGRESSION,
             BUILTIN_TASK_RECONSTRUCT_NODE_FEAT]:
                if dataloader is None:
                    res[task_info.task_id] = (None, None)
                else:
                    res[task_info.task_id] = ntask_res
            elif task_info.task_type in \
            [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION,
             BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT]:
                if dataloader is None:
                    res[task_info.task_id] = (None, None)
                else:
                    res[task_info.task_id] = etask_res
            elif task_info.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
                if dataloader is None:
                    res[task_info.task_id] = None
                else:
                    res[task_info.task_id] = lp_res

        return res

    # avoid calling log_print_metrics
    def mock_func_get_rank():
        return 1

    @patch("graphstorm.trainer.mt_trainer.get_rank", side_effect = mock_func_get_rank)
    @patch("graphstorm.trainer.mt_trainer.multi_task_mini_batch_predict", side_effect = mock_func_multi_task_mini_batch_predict)
    @patch("graphstorm.trainer.mt_trainer.run_lp_mini_batch_predict", side_effect = mock_func_run_lp_mini_batch_predict)
    @patch("graphstorm.trainer.mt_trainer.do_full_graph_inference", side_effect = mock_func_do_full_graph_inference)
    @patch("graphstorm.trainer.mt_trainer.do_mini_batch_inference", side_effect = mock_func_do_mini_batch_inference)
    def check_eval(mock_do_mini_batch_inference,
                   mock_do_full_graph_inference,
                   mock_run_lp_mini_batch_predict,
                   mock_multi_task_mini_batch_predict,
                   mock_get_rank):

        val_dataloaders = [nc_dataloader, nr_dataloader, ec_dataloader,
                       er_dataloader, lp_dataloader, nfr_dataloader,
                       efr_dataloader]
        test_dataloaders = [nc_dataloader, nr_dataloader, ec_dataloader,
                            er_dataloader, lp_dataloader, nfr_dataloader,
                            efr_dataloader]
        val_loader = GSgnnMultiTaskDataLoader(None, task_infos, val_dataloaders)
        test_loader = GSgnnMultiTaskDataLoader(None, task_infos, test_dataloaders)

        target_res = {
            "nc_task":ntask_res,
            "nr_task":ntask_res,
            "ec_task":etask_res,
            "er_task":etask_res,
            "lp_task":lp_res,
            "nfr_task":ntask_res,
            "efr_task":etask_res
        }
        evaluator = MTaskCheckerEvaluator(target_res, None, 100)
        mt_trainer.setup_evaluator(evaluator)

        # test when val_loader is None
        mt_trainer.eval(model, data, val_loader, None, 100)

        evaluator = MTaskCheckerEvaluator(None, target_res, 100)
        mt_trainer.setup_evaluator(evaluator)
        mt_trainer.eval(model, data, None, test_loader, 100)

        evaluator = MTaskCheckerEvaluator(target_res, target_res, 100)
        mt_trainer.setup_evaluator(evaluator)
        mt_trainer.eval(model, data, val_loader, test_loader, 100)

        # predict tasks are empty
        val_dataloaders = [None, None, None,
                       None, lp_dataloader, nfr_dataloader,
                       efr_dataloader]
        test_dataloaders = [None, None, None,
                            None, lp_dataloader, nfr_dataloader,
                            efr_dataloader]
        val_loader = GSgnnMultiTaskDataLoader(None, task_infos, val_dataloaders)
        test_loader = GSgnnMultiTaskDataLoader(None, task_infos, test_dataloaders)

        target_res = {
            "lp_task":lp_res,
            "nfr_task":ntask_res,
            "efr_task":etask_res
        }
        evaluator = MTaskCheckerEvaluator(target_res, target_res, 100)
        mt_trainer.setup_evaluator(evaluator)
        mt_trainer.eval(model, data, val_loader, test_loader, 100)

        # lp tasks are empty
        val_dataloaders = [nc_dataloader, nr_dataloader, ec_dataloader,
                       er_dataloader, None, nfr_dataloader, efr_dataloader]
        test_dataloaders = [nc_dataloader, nr_dataloader, ec_dataloader,
                            er_dataloader, None, nfr_dataloader, efr_dataloader]
        val_loader = GSgnnMultiTaskDataLoader(None, task_infos, val_dataloaders)
        test_loader = GSgnnMultiTaskDataLoader(None, task_infos, test_dataloaders)
        target_res = {
            "nc_task":ntask_res,
            "nr_task":ntask_res,
            "ec_task":etask_res,
            "er_task":etask_res,
            "nfr_task":ntask_res,
            "efr_task":etask_res
        }
        evaluator = MTaskCheckerEvaluator(target_res, target_res, 200)
        mt_trainer.setup_evaluator(evaluator)
        mt_trainer.eval(model, data, val_loader, test_loader, 200)

        # node feature reconstruct tasks are empty
        # edge feature reconstruct tasks are empty
        val_dataloaders = [nc_dataloader, nr_dataloader, ec_dataloader,
                       er_dataloader, lp_dataloader, None, None]
        test_dataloaders = [nc_dataloader, nr_dataloader, ec_dataloader,
                            er_dataloader, lp_dataloader, None, None]
        val_loader = GSgnnMultiTaskDataLoader(None, task_infos, val_dataloaders)
        test_loader = GSgnnMultiTaskDataLoader(None, task_infos, test_dataloaders)
        target_res = {
            "nc_task":ntask_res,
            "nr_task":ntask_res,
            "ec_task":etask_res,
            "er_task":etask_res,
            "lp_task":lp_res,
        }
        evaluator = MTaskCheckerEvaluator(target_res, target_res, 200)
        mt_trainer.setup_evaluator(evaluator)
        mt_trainer.eval(model, data, val_loader, test_loader, 200)

    check_eval()

def create_config4ef(tmp_path, file_name, encoder='rgcn', task='nc', use_ef=True):
    """ Create a specific config object for yaml configuration.

    encoder can be "rgcn" and "hgt".
    task can be "nc", "ec", and "lp".

    If later on you want to add new encoders or tasks, please add corresponding config objects.
    """
    conf_object = {
        "version": 1.0
        }

    gsf_object = {}

    # config basic object
    basic_obj = {"node_feat_name": ["feat"]}
    if use_ef:
        basic_obj["edge_feat_name"] = ["n0,r1,n1:feat",
                                       "n0,r0,n1:feat"]

    gsf_object["basic"] = basic_obj

    # config gnn object
    gnn_obj = {
                "num_layers": 2,
                "hidden_size": 16,
                "lr": 0.001
        }

    gnn_obj["model_encoder_type"] = encoder

    gsf_object["gnn"] = gnn_obj
    
    # config input and output
    gsf_object["input"] = {}
    gsf_object["output"] = {}

    # config hyper parameters
    hp_ob = {
        "fanout": "10,10",
        "batch_size": 2
    }
    gsf_object["hyperparam"] = hp_ob

    # config encoder model specific configurations
    if encoder == "hgt":
        hgt_obj = {"num_heads": 4}
        gsf_object["hgt"] = hgt_obj

    # config task specific configurations
    if task == "nc":
        nc_obj = {
            "num_classes": 10,
            "target_ntype": "n1",
            "label_field": "label"
        }
        gsf_object["node_classification"] = nc_obj
    elif task == "ec":
        ec_obj = {
            "num_classes": 10,
            "target_etype": ["n0,r1,n1"],
            "label_field": "label",
            "remove_target_edge_type": False
        }
        gsf_object["edge_classification"] = ec_obj
    elif task == "lp":
        lp_obj = {
            "train_etype": ["n0,r0,n1", "n0,r1,n1"],
            "eval_etype": ["n0,r0,n1"],
            "exclude_training_targets": False,
            "num_negative_edges": 10,
            "lp_decoder_type": "dot_product"
        }
        gsf_object["link_prediction"] = lp_obj
    else:
        raise NotImplementedError(f'This test config does not support the {task} task. Options include' + \
            '\"nc\", \"ec\", and \"lp\".')

    conf_object["gsf"] = gsf_object

    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

def test_rgcn_nc4ef():
    """ Test RGCN model Node Classification traning pipeline with/without edge features.
    """
    print(f'=================== Test RGCN Training NC with EF ===================')
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    setup_device(0)
    device = get_device()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # Test case 0: normal case, set RGCN model with edge features for NC, and provide
        #              edge features.
        #              Should complete 2 epochs and output training loss and evaluation
        #              metrics.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model1 = create_builtin_node_gnn_model(gdata.g, config, True)
        trainer1 = GSgnnNodePredictionTrainer(model1)
        trainer1.setup_device(device)

        train_dataloader1 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_train_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            train_task=True)
        val_dataloader1 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_val_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            train_task=False)

        evaluator1 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)
        trainer1.setup_evaluator(evaluator1)

        trainer1.fit(
            train_loader=train_dataloader1,
            val_loader=val_dataloader1,
            num_epochs=2
            )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # Test case 1: normal case, set RGCN model without edge features for NC, and not
        #              provide edge features.
        #              Should complete 2 epochs and output training loss and evaluation
        #              metrics.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model2 = create_builtin_node_gnn_model(gdata.g, config, True)
        trainer2 = GSgnnNodePredictionTrainer(model2)
        trainer2.setup_device(device)

        train_dataloader2 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_train_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            train_task=True)
        val_dataloader2 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_val_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            train_task=False)

        evaluator2 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)

        trainer2.setup_evaluator(evaluator2)

        trainer2.fit(
            train_loader=train_dataloader2,
            val_loader=val_dataloader2,
            num_epochs=2
            )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # Test case 2: abnormal case, set RGCN model with edge features for NC, but not
        #              provide edge features.
        #              This will trigger an assertion error, asking for giving edge feature
        #              for message passing computation.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model3 = create_builtin_node_gnn_model(gdata.g, config, True)
        trainer3 = GSgnnNodePredictionTrainer(model3)
        trainer3.setup_device(device)

        train_dataloader3 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_train_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            train_task=True)
        val_dataloader3 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_val_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            train_task=False)

        evaluator3 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)

        trainer3.setup_evaluator(evaluator3)

        with assert_raises(AssertionError):
            trainer3.fit(
                train_loader=train_dataloader3,
                val_loader=val_dataloader3,
                num_epochs=2
                )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # Test case 3: abnormal case, set RGCN model without edge features for NC, but 
        #              provide edge features.
        #              This will trigger an assertion error, asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model4 = create_builtin_node_gnn_model(gdata.g, config, True)
        trainer4 = GSgnnNodePredictionTrainer(model4)
        trainer4.setup_device(device)

        train_dataloader4 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_train_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            train_task=True)
        val_dataloader4 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_val_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            train_task=False)
        evaluator4 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)

        trainer4.setup_evaluator(evaluator4)

        with assert_raises(AssertionError):
            trainer4.fit(
                train_loader=train_dataloader4,
                val_loader=val_dataloader4,
                num_epochs=2
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_hgt_nc4ef():
    """ Test HGT model Node Classification traning pipeline with/without edge features.
    
    Because HGT encoder dose not support edge feature so far, if initialized with edge_feat_name,
    it will trigger a Not-support assertion error.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    setup_device(0)
    device = get_device()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # Test case 0: normal case, set HGT model without edge features for NC, and not provide
        #              edge features.
        #              Should complete 2 epochs and output training loss and evaluation
        #              metrics.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model1 = create_builtin_node_gnn_model(gdata.g, config, True)
        trainer1 = GSgnnNodePredictionTrainer(model1)
        trainer1.setup_device(device)

        train_dataloader1 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_train_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            train_task=True)
        val_dataloader1 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_val_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            train_task=False)

        evaluator1 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)

        trainer1.setup_evaluator(evaluator1)

        trainer1.fit(
            train_loader=train_dataloader1,
            val_loader=val_dataloader1,
            num_epochs=2
            )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # Test case 1: abnormal case, set HGT model with edge features for NC, and not provide
        #              edge features.
        #              Should trigger an assertion error of not support edge feature in hgt
        #              encoder.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        with assert_raises(AssertionError):
            model2 = create_builtin_node_gnn_model(gdata.g, config, True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # Test case 2: abnormal case, set HGT model without edge features for NC, but provide
        #              edge features.
        #              Should trigger an assertion error， asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model3 = create_builtin_node_gnn_model(gdata.g, config, True)
        trainer3 = GSgnnNodePredictionTrainer(model3)
        trainer3.setup_device(device)

        train_dataloader3 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_train_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            train_task=True)
        val_dataloader3 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_val_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            train_task=False)

        evaluator3 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)

        trainer3.setup_evaluator(evaluator3)

        with assert_raises(AssertionError):
            trainer3.fit(
                train_loader=train_dataloader3,
                val_loader=val_dataloader3,
                num_epochs=2
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_rgcn_ec4ef():
    """ Test RGCN model Edge Classification traning pipeline with/without edge features.
    """
    print(f'=================== Test RGCN Training EC with EF ===================')
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    setup_device(0)
    device = get_device()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname, add_reverse=True)
        gdata = GSgnnData(part_config=part_config)

        # Test case 0: normal case, set RGCN model with edge features for EC, and provide
        #              edge features.
        #              Should complete 2 epochs and output training loss and evaluation
        #              metrics.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='ec', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        
        model1 = create_builtin_edge_gnn_model(gdata.g, config, True)
        trainer1 = GSgnnEdgePredictionTrainer(model1)
        trainer1.setup_device(device)

        train_dataloader1 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_train_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=True)
        val_dataloader1 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)

        evaluator1 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)
        trainer1.setup_evaluator(evaluator1)

        trainer1.fit(
            train_loader=train_dataloader1,
            val_loader=val_dataloader1,
            num_epochs=2
            )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname, add_reverse=True)
        gdata = GSgnnData(part_config=part_config)

        # Test case 1: normal case, set RGCN model without edge features for EC, and not
        #              provide edge features.
        #              Should complete 2 epochs and output training loss and evaluation
        #              metrics.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='ec', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model2 = create_builtin_edge_gnn_model(gdata.g, config, True)
        trainer2 = GSgnnEdgePredictionTrainer(model2)
        trainer2.setup_device(device)

        train_dataloader2 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_train_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=True)
        val_dataloader2 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)

        evaluator2 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)
        trainer2.setup_evaluator(evaluator2)

        trainer2.fit(
            train_loader=train_dataloader2,
            val_loader=val_dataloader2,
            num_epochs=2
            )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname, add_reverse=True)
        gdata = GSgnnData(part_config=part_config)

        # Test case 2: abnormal case, set RGCN model with edge features for EC, but not
        #              provide edge features.
        #              This will trigger an assertion error, asking for giving edge feature
        #              for message passing computation.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='ec', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        
        model3 = create_builtin_edge_gnn_model(gdata.g, config, True)
        trainer3 = GSgnnEdgePredictionTrainer(model3)
        trainer3.setup_device(device)

        train_dataloader3 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_train_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=True)
        val_dataloader3 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)

        evaluator3 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)
        trainer3.setup_evaluator(evaluator3)

        with assert_raises(AssertionError):
            trainer3.fit(
                train_loader=train_dataloader3,
                val_loader=val_dataloader3,
                num_epochs=2
                )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname, add_reverse=True)
        gdata = GSgnnData(part_config=part_config)

        # Test case 3: abnormal case, set RGCN model without edge features for NC, but 
        #              provide edge features.
        #              This will trigger an assertion error, asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='ec', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        
        model4 = create_builtin_edge_gnn_model(gdata.g, config, True)
        trainer4 = GSgnnEdgePredictionTrainer(model4)
        trainer4.setup_device(device)

        train_dataloader4 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_train_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=True)
        val_dataloader4 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)

        evaluator4 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)
        trainer4.setup_evaluator(evaluator4)

        with assert_raises(AssertionError):
            trainer4.fit(
                train_loader=train_dataloader4,
                val_loader=val_dataloader4,
                num_epochs=2
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_hgt_ec4ef():
    """ Test HGT model Edge Classification traning pipeline with/without edge features.
    
    Because HGT encoder dose not support edge feature so far, if initialized with edge_feat_name,
    it will trigger a Not-support assertion error.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    setup_device(0)
    device = get_device()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # Test case 0: normal case, set HGT model without edge features for EC, and not provide
        #              edge features.
        #              Should complete 2 epochs and output training loss and evaluation
        #              metrics.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt',
                         task='ec', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model1 = create_builtin_edge_gnn_model(gdata.g, config, True)
        trainer1 = GSgnnEdgePredictionTrainer(model1)
        trainer1.setup_device(device)

        train_dataloader1 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_train_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=True)
        val_dataloader1 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)

        evaluator1 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)
        trainer1.setup_evaluator(evaluator1)

        trainer1.fit(
            train_loader=train_dataloader1,
            val_loader=val_dataloader1,
            num_epochs=2
            )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # Test case 1: abnormal case, set HGT model with edge features for EC, and not provide
        #              edge features.
        #              Should trigger an assertion error of not support edge feature in hgt
        #              encoder.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt', task='ec', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        with assert_raises(AssertionError):
            model2 = create_builtin_edge_gnn_model(gdata.g, config, True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        gdata = GSgnnData(part_config=part_config)

        # Test case 2: abnormal case, set HGT model without edge features for EC, but provide
        #              edge features.
        #              Should trigger an assertion error， asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt',
                         task='ec', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model3 = create_builtin_edge_gnn_model(gdata.g, config, True)
        trainer3 = GSgnnEdgePredictionTrainer(model3)
        trainer3.setup_device(device)

        train_dataloader3 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_train_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=True)
        val_dataloader3 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)

        evaluator3 = GSgnnClassificationEvaluator(config.eval_frequency,
                                                    config.eval_metric,
                                                    config.multilabel,
                                                    config.use_early_stop)
        trainer3.setup_evaluator(evaluator3)

        with assert_raises(AssertionError):
            trainer3.fit(
                train_loader=train_dataloader3,
                val_loader=val_dataloader3,
                num_epochs=2
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_rgcn_lp4ef():
    """ Test RGCN model Link Prediction traning pipeline with/without edge features.
    """
    print(f'=================== Test RGCN Training LP with EF ===================')
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    setup_device(0)
    device = get_device()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)

        # Test case 0: normal case, set RGCN model with edge features for LP, and provide
        #              edge features.
        #              Should complete 2 epochs and output training loss and evaluation
        #              metrics.
        create_config4ef(Path(tmpdirname), 'gnn_lp.yaml', task='lp', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_lp.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata_lp = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name,  # Need to set these features in
                          edge_feat_field=config.edge_feat_name,  # dataset, as lp_trainer uses
                          )                                       # a different mini-batch method.

        model1 = create_builtin_lp_gnn_model(gdata_lp.g, config, True)
        trainer1 = GSgnnLinkPredictionTrainer(model1)
        trainer1.setup_device(device)
        
        train_dataloader1 = GSgnnLinkPredictionDataLoader(
            gdata_lp,
            target_idx=gdata_lp.get_edge_train_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            num_negative_edges=config.num_negative_edges,
            exclude_training_targets=config.exclude_training_targets,
            train_task=True)
        val_dataloader1 = GSgnnLinkPredictionTestDataLoader(
            gdata_lp,
            target_idx=gdata_lp.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            num_negative_edges=config.num_negative_edges)

        evaluator1 = GSgnnLPEvaluator(config.eval_frequency,
                                      config.eval_metric,
                                      config.multilabel,
                                      config.use_early_stop)
        trainer1.setup_evaluator(evaluator1)

        trainer1.fit(
            train_loader=train_dataloader1,
            val_loader=val_dataloader1,
            num_epochs=2
            )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)

        # Test case 1: normal case, set RGCN model without edge features for LP, and not
        #              provide edge features.
        #              Should complete 2 epochs and output training loss and evaluation
        #              metrics.
        create_config4ef(Path(tmpdirname), 'gnn_lp.yaml', task='lp', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_lp.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata_lp = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name)

        model2 = create_builtin_lp_gnn_model(gdata_lp.g, config, True)
        trainer2 = GSgnnLinkPredictionTrainer(model2)
        trainer2.setup_device(device)
        
        train_dataloader2 = GSgnnLinkPredictionDataLoader(
            gdata_lp,
            target_idx=gdata_lp.get_edge_train_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,  # Because LP use gdata to extract feature, this
                                               # setting does not change results
            num_negative_edges=config.num_negative_edges,
            exclude_training_targets=config.exclude_training_targets,
            train_task=True)
        val_dataloader2 = GSgnnLinkPredictionTestDataLoader(
            gdata_lp,
            target_idx=gdata_lp.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,  # Because LP use gdata to extract feature, this
                                               # setting does not change results
            num_negative_edges=config.num_negative_edges)

        evaluator2 = GSgnnLPEvaluator(config.eval_frequency,
                                      config.eval_metric,
                                      config.multilabel,
                                      config.use_early_stop)
        trainer2.setup_evaluator(evaluator2)

        trainer2.fit(
            train_loader=train_dataloader2,
            val_loader=val_dataloader2,
            num_epochs=2
            )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)

        # Test case 2: abnormal case, set RGCN model with edge features for LP, but not
        #              provide edge features.
        #              This will trigger an assertion error, asking for giving edge feature
        #              for message passing computation.
        create_config4ef(Path(tmpdirname), 'gnn_lp.yaml', task='lp', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_lp.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata_lp = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name)  # only provide node features.

        model3 = create_builtin_lp_gnn_model(gdata_lp.g, config, True)
        trainer3 = GSgnnLinkPredictionTrainer(model3)
        trainer3.setup_device(device)
        
        train_dataloader3 = GSgnnLinkPredictionDataLoader(
            gdata_lp,
            target_idx=gdata_lp.get_edge_train_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,  # Because LP use gdata to extract feature, this
                                               # setting does not change results
            num_negative_edges=config.num_negative_edges,
            exclude_training_targets=config.exclude_training_targets,
            train_task=True)
        val_dataloader3 = GSgnnLinkPredictionTestDataLoader(
            gdata_lp,
            target_idx=gdata_lp.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,  # Because LP use gdata to extract feature, this
                                               # setting does not change results
            num_negative_edges=config.num_negative_edges)

        evaluator3 = GSgnnLPEvaluator(config.eval_frequency,
                                      config.eval_metric,
                                      config.multilabel,
                                      config.use_early_stop)
        trainer3.setup_evaluator(evaluator3)

        with assert_raises(AssertionError):
            trainer3.fit(
                train_loader=train_dataloader3,
                val_loader=val_dataloader3,
                num_epochs=2
                )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)

        # Test case 3: abnormal case, set RGCN model without edge features for LP, but 
        #              provide edge features.
        #              This will trigger an assertion error, asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_lp.yaml', task='lp', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_lp.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        gdata_lp = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name,
                          edge_feat_field={('n0', 'r0', 'n1'): ['feat'],
                                            ('n0', 'r1', 'n1'): ['feat']},# Manually set, as 
                                                                          # config does not have it
                          )
        model4 = create_builtin_lp_gnn_model(gdata_lp.g, config, True)
        trainer4 = GSgnnLinkPredictionTrainer(model4)
        trainer4.setup_device(device)
        
        train_dataloader4 = GSgnnLinkPredictionDataLoader(
            gdata_lp,
            target_idx=gdata_lp.get_edge_train_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,
            num_negative_edges=config.num_negative_edges,
            exclude_training_targets=config.exclude_training_targets,
            train_task=True)
        val_dataloader4 = GSgnnLinkPredictionTestDataLoader(
            gdata_lp,
            target_idx=gdata_lp.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,
            num_negative_edges=config.num_negative_edges)

        evaluator4 = GSgnnLPEvaluator(config.eval_frequency,
                                      config.eval_metric,
                                      config.multilabel,
                                      config.use_early_stop)
        trainer4.setup_evaluator(evaluator4)

        with assert_raises(AssertionError):
            trainer4.fit(
                train_loader=train_dataloader4,
                val_loader=val_dataloader4,
                num_epochs=2
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_hgt_lp4ef():
    """ Test HGT model Link Prediction traning pipeline with/without edge features.
    
    Because HGT encoder dose not support edge feature so far, if initialized with edge_feat_name,
    it will trigger a Not-support assertion error.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    setup_device(0)
    device = get_device()

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)

        # Test case 0: normal case, set HGT model without edge features for LP, and not provide
        #              edge features.
        #              Should complete 2 epochs and output training loss and evaluation
        #              metrics.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt', task='lp', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name)

        model1 = create_builtin_lp_gnn_model(gdata.g, config, True)
        trainer1 = GSgnnLinkPredictionTrainer(model1)
        trainer1.setup_device(device)
        
        train_dataloader1 = GSgnnLinkPredictionDataLoader(
            gdata,
            target_idx=gdata.get_edge_train_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,  # Because LP use gdata to extract feature, this
                              # setting does not change results
            num_negative_edges=config.num_negative_edges,
            exclude_training_targets=config.exclude_training_targets,
            train_task=True)
        val_dataloader1 = GSgnnLinkPredictionTestDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,  # Because LP use gdata to extract feature, this
                              # setting does not change results
            num_negative_edges=config.num_negative_edges)

        evaluator1 = GSgnnLPEvaluator(config.eval_frequency,
                                      config.eval_metric,
                                      config.multilabel,
                                      config.use_early_stop)
        trainer1.setup_evaluator(evaluator1)

        trainer1.fit(
            train_loader=train_dataloader1,
            val_loader=val_dataloader1,
            num_epochs=2
            )

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)

        # Test case 1: abnormal case, set HGT model with edge features for LP, and not provide
        #              edge features.
        #              Should trigger an assertion error of not support edge feature in hgt
        #              encoder.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt', task='lp', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name)

        with assert_raises(AssertionError):
            model2 = create_builtin_lp_gnn_model(gdata.g, config, True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)

        # Test case 2: abnormal case, set HGT model without edge features for LP, but provide
        #              edge features.
        #              Should trigger an assertion error， asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt',
                         task='lp', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name,
                          edge_feat_field={('n0', 'r0', 'n1'): ['feat'],
                                            ('n0', 'r1', 'n1'): ['feat']},# Manually set, as 
                                                                          # config does not have it
                          )

        model3 = create_builtin_lp_gnn_model(gdata.g, config, True)
        trainer3 = GSgnnLinkPredictionTrainer(model3)
        trainer3.setup_device(device)
        
        train_dataloader3 = GSgnnLinkPredictionDataLoader(
            gdata,
            target_idx=gdata.get_edge_train_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,  # Because LP use gdata to extract feature, this
                              # setting does not change results
            num_negative_edges=config.num_negative_edges,
            exclude_training_targets=config.exclude_training_targets,
            train_task=True)
        val_dataloader3 = GSgnnLinkPredictionTestDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,  # Because LP use gdata to extract feature, this
                              # setting does not change results
            num_negative_edges=config.num_negative_edges)

        evaluator3 = GSgnnLPEvaluator(config.eval_frequency,
                                      config.eval_metric,
                                      config.multilabel,
                                      config.use_early_stop)
        trainer3.setup_evaluator(evaluator3)

        with assert_raises(AssertionError):
            trainer3.fit(
                train_loader=train_dataloader3,
                val_loader=val_dataloader3,
                num_epochs=2
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()


if __name__ == '__main__':
    test_mtask_eval()
    test_trainer_setup_evaluator()

    test_mtask_prepare_node_mini_batch()
    test_mtask_prepare_edge_mini_batch()
    test_mtask_prepare_lp_mini_batch()
    test_mtask_prepare_reconstruct_node_feat()
    test_mtask_prepare_reconstruct_edge_feat()

    test_rgcn_nc4ef()
    test_hgt_nc4ef()
    test_rgcn_ec4ef()
    test_hgt_ec4ef()
    test_rgcn_lp4ef()
    test_hgt_lp4ef()
