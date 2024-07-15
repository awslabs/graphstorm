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
import numpy as np
import torch as th
from unittest.mock import patch

from graphstorm.tracker import GSSageMakerTaskTracker
from graphstorm import create_builtin_node_gnn_model
from graphstorm.inference.graphstorm_infer import GSInferrer
from graphstorm.eval import GSgnnClassificationEvaluator
from graphstorm.dataloading import GSgnnMultiTaskDataLoader
from graphstorm.inference import GSgnnMultiTaskLearningInferrer
from graphstorm.model import LinkPredictDistMultDecoder

from graphstorm.config import (GSConfig, TaskInfo)
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                                BUILTIN_TASK_NODE_REGRESSION,
                                BUILTIN_TASK_EDGE_CLASSIFICATION,
                                BUILTIN_TASK_EDGE_REGRESSION,
                                BUILTIN_TASK_LINK_PREDICTION,
                                BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)

from data_utils import generate_dummy_dist_graph

from util import (DummyGSgnnData,
                  DummyGSgnnEncoderModel,
                  DummyGSgnnMTModel,
                  DummyGSgnnNodeDataLoader,
                  DummyGSgnnEdgeDataLoader,
                  DummyGSgnnLinkPredictionDataLoader)
from test_trainer import MTaskCheckerEvaluator

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


def test_inferrer_setup_evaluator():

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
    inferrer = GSInferrer(model)

    # case 1: by default trainer has no task_tracker
    assert inferrer.task_tracker is None

    evaluator = GSgnnClassificationEvaluator(config.eval_frequency,
                                             config.eval_metric,
                                             config.multilabel,
                                             config.use_early_stop)

    # case 2: evaluator has no task_tracker by default
    assert evaluator.task_tracker is None

    # case 3: when setup an evaluator that has no task_tracker and train has no task tracker
    #         eitehr, create a new task_tracker and set it to the evaluator.
    inferrer.setup_evaluator(evaluator)

    assert inferrer.task_tracker is not None
    assert evaluator.eval_frequency == inferrer.task_tracker.log_report_frequency
    assert evaluator.task_tracker == inferrer.task_tracker

    # case 4: when setup an evaluator that has no task_tracker, but train has a task tracker,
    #         use the trainer's task_tracker to setup the evaluator.
    inferrer.setup_task_tracker(GSSageMakerTaskTracker(10))
    evaluator.setup_task_tracker(None)
    inferrer.setup_evaluator(evaluator)

    assert evaluator.task_tracker == inferrer.task_tracker
    assert evaluator.task_tracker.log_report_frequency == 10

    # case 5: when setup an evaluator that has a task_tracker, no change of the evaluator.
    evaluator.setup_task_tracker(GSSageMakerTaskTracker(100))
    inferrer.setup_evaluator(evaluator)

    assert evaluator.task_tracker != inferrer.task_tracker
    assert evaluator.task_tracker.log_report_frequency == 100
    assert inferrer.task_tracker.log_report_frequency != 100

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_mtask_infer():
    tast_info_nc = TaskInfo(task_type=BUILTIN_TASK_NODE_CLASSIFICATION,
                            task_id='nc_task',
                            task_config=None)
    nc_dataloader = DummyGSgnnNodeDataLoader()
    tast_info_nr = TaskInfo(task_type=BUILTIN_TASK_NODE_REGRESSION,
                            task_id='nr_task',
                            task_config=None)
    nr_dataloader = DummyGSgnnNodeDataLoader()
    tast_info_ec = TaskInfo(task_type=BUILTIN_TASK_EDGE_CLASSIFICATION,
                            task_id='ec_task',
                            task_config=None)
    ec_dataloader = DummyGSgnnEdgeDataLoader()
    tast_info_er = TaskInfo(task_type=BUILTIN_TASK_EDGE_REGRESSION,
                            task_id='er_task',
                            task_config=None)
    er_dataloader = DummyGSgnnEdgeDataLoader()

    task_config = GSConfig.__new__(GSConfig)
    setattr(task_config, "train_mask", "train_mask")
    tast_info_lp = TaskInfo(task_type=BUILTIN_TASK_LINK_PREDICTION,
                            task_id='lp_task',
                            task_config=task_config)
    lp_dataloader = DummyGSgnnLinkPredictionDataLoader()

    tast_info_nfr = TaskInfo(task_type=BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                            task_id='nfr_task',
                            task_config=None)
    nfr_dataloader = DummyGSgnnNodeDataLoader()

    encoder_model = DummyGSgnnEncoderModel()
    model = DummyGSgnnMTModel(encoder_model, decoders={tast_info_lp.task_id: LinkPredictDistMultDecoder([("n1","r1","n2")], 128)}, has_sparse=True)
    mt_infer = GSgnnMultiTaskLearningInferrer(model)
    mt_infer._device = 'cpu'

    predict_dataloaders = [nc_dataloader, nr_dataloader,
                           ec_dataloader, er_dataloader]
    predict_tasks = [tast_info_nc, tast_info_nr,
                     tast_info_ec, tast_info_er]
    lp_dataloaders = [lp_dataloader]
    lp_tasks = [tast_info_lp]

    nfr_dataloaders = [nfr_dataloader]
    nfr_tasks = [tast_info_nfr]

    data = DummyGSgnnData()

    output_path = "/tmp/output/"
    emb_path = os.path.join(output_path, "emb")
    # We do not test save_prediction_path is not None,
    # which is too complex to mock in unit tests.
    # Will test it in end2end test.
    pred_path = None

    def mock_func_log_print_metrics(*args, **kwargs):
        pass

    def mock_func_save_relation_embeddings(rel_emb_path, decoder):
        assert rel_emb_path == os.path.join(emb_path, tast_info_lp.task_id)
        assert decoder.num_rels == 1

    def mock_func_save_gsgnn_embeddings(g, save_embed_path, embs, node_id_mapping_file, save_embed_format):
        assert save_embed_path == emb_path
        # return by mock_func_do_mini_batch_inference or mock_func_do_full_graph_inference
        assert len(embs) == 2

    # avoid calling log_print_metrics
    def mock_func_get_rank():
        return 0

    def mock_func_do_mini_batch_inference(*args, **kwargs):
        return {
            "n1": None,
            "n2": None,
        }

    def mock_func_do_full_graph_inference(*args, **kwargs):
        return {
            "n1": None,
            "n2": None,
        }

    def mock_func_run_lp_mini_batch_predict(*args, **kwargs):
        return lp_res

    ntask_res = (np.arange(10), np.arange(10))
    etask_res = (np.arange(20), np.arange(20))
    lp_res = np.arange(5)
    def mock_func_multi_task_mini_batch_predict(model, emb, dataloaders, task_infos, device, return_proba, return_label):
        assert len(emb) == 2
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
            [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
                if dataloader is None:
                    res[task_info.task_id] = (None, None)
                else:
                    res[task_info.task_id] = etask_res
            elif task_info.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
                # Link prediction predict should call
                # run_lp_mini_batch_predict
                assert False
        return res

    @patch("graphstorm.inference.mt_infer.GSgnnMultiTaskLearningInferrer.log_print_metrics", side_effect = mock_func_log_print_metrics)
    @patch("graphstorm.inference.mt_infer.save_relation_embeddings", side_effect = mock_func_save_relation_embeddings)
    @patch("graphstorm.inference.mt_infer.save_gsgnn_embeddings", side_effect = mock_func_save_gsgnn_embeddings)
    @patch("graphstorm.inference.mt_infer.get_rank", side_effect = mock_func_get_rank)
    @patch("graphstorm.inference.mt_infer.multi_task_mini_batch_predict", side_effect = mock_func_multi_task_mini_batch_predict)
    @patch("graphstorm.inference.mt_infer.run_lp_mini_batch_predict", side_effect = mock_func_run_lp_mini_batch_predict)
    @patch("graphstorm.inference.mt_infer.do_full_graph_inference", side_effect = mock_func_do_full_graph_inference)
    @patch("graphstorm.inference.mt_infer.do_mini_batch_inference", side_effect = mock_func_do_mini_batch_inference)
    def check_eval(mock_do_mini_batch_inference,
                mock_do_full_graph_inference,
                mock_run_lp_mini_batch_predict,
                mock_multi_task_mini_batch_predict,
                mock_get_rank,
                mock_save_gsgnn_embeddings,
                mock_save_relation_embeddings,
                mock_log_print_metrics):
        test_dataloader = GSgnnMultiTaskDataLoader(None, predict_tasks, predict_dataloaders)

        lp_test_dataloader = GSgnnMultiTaskDataLoader(None, lp_tasks, lp_dataloaders)

        nfr_test_dataloader = GSgnnMultiTaskDataLoader(None, nfr_tasks, nfr_dataloaders)

        target_res = {
            "nc_task":ntask_res,
            "nr_task":ntask_res,
            "ec_task":etask_res,
            "er_task":etask_res,
            "lp_task":lp_res,
            "nfr_task":ntask_res
        }
        evaluator = MTaskCheckerEvaluator(target_res, target_res, 0)
        mt_infer.setup_evaluator(evaluator)

        mt_infer.infer(data,
                test_dataloader,
                lp_test_dataloader,
                nfr_test_dataloader,
                emb_path,
                pred_path,
                use_mini_batch_infer=True,
                node_id_mapping_file=None,
                edge_id_mapping_file=None,
                return_proba=True)

        mt_infer.infer(data,
                test_dataloader,
                lp_test_dataloader,
                nfr_test_dataloader,
                emb_path,
                pred_path,
                use_mini_batch_infer=False,
                node_id_mapping_file=None,
                edge_id_mapping_file=None,
                return_proba=False)

        target_res = {
            "nc_task":ntask_res,
            "nr_task":ntask_res,
            "ec_task":etask_res,
            "er_task":etask_res,
        }
        evaluator = MTaskCheckerEvaluator(target_res, target_res, 0)
        mt_infer.setup_evaluator(evaluator)
        mt_infer.infer(data,
                test_dataloader,
                None,
                None,
                emb_path,
                pred_path,
                use_mini_batch_infer=True,
                node_id_mapping_file=None,
                edge_id_mapping_file=None,
                return_proba=True)

        target_res = {
            "lp_task":lp_res
        }
        evaluator = MTaskCheckerEvaluator(target_res, target_res, 0)
        mt_infer.setup_evaluator(evaluator)
        mt_infer.infer(data,
                None,
                lp_test_dataloader,
                None,
                emb_path,
                pred_path,
                use_mini_batch_infer=True,
                node_id_mapping_file=None,
                edge_id_mapping_file=None,
                return_proba=True)

        target_res = {
            "nfr_task":ntask_res
        }
        evaluator = MTaskCheckerEvaluator(target_res, target_res, 0)
        mt_infer.setup_evaluator(evaluator)

        mt_infer.infer(data,
                None,
                None,
                nfr_test_dataloader,
                emb_path,
                pred_path,
                use_mini_batch_infer=True,
                node_id_mapping_file=None,
                edge_id_mapping_file=None,
                return_proba=True)

    check_eval()

if __name__ == '__main__':
    test_mtask_infer()

    test_inferrer_setup_evaluator()
