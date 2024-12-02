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
import shutil
from argparse import Namespace
import numpy as np
import torch as th
from unittest.mock import patch

from graphstorm.tracker import GSSageMakerTaskTracker
from graphstorm import create_builtin_node_gnn_model
from graphstorm.utils import setup_device, get_device
from graphstorm.inference.graphstorm_infer import GSInferrer
from graphstorm.eval import GSgnnClassificationEvaluator
from graphstorm.dataloading import GSgnnData, GSgnnMultiTaskDataLoader
from graphstorm.dataloading import (GSgnnNodeDataLoader,
                                    GSgnnEdgeDataLoader,
                                    GSgnnLinkPredictionTestDataLoader)
from graphstorm.inference import (GSgnnMultiTaskLearningInferrer,
                                  GSgnnNodePredictionInferrer,
                                  GSgnnEdgePredictionInferrer,
                                  GSgnnLinkPredictionInferrer)
from graphstorm.model import LinkPredictDistMultDecoder
from graphstorm import (create_builtin_node_gnn_model,
                        create_builtin_edge_gnn_model,
                        create_builtin_lp_gnn_model)
from graphstorm.config import (GSConfig, TaskInfo)
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                                BUILTIN_TASK_NODE_REGRESSION,
                                BUILTIN_TASK_EDGE_CLASSIFICATION,
                                BUILTIN_TASK_EDGE_REGRESSION,
                                BUILTIN_TASK_LINK_PREDICTION,
                                BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                                BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT)

from numpy.testing import assert_raises, assert_equal
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
    tast_info_efr = TaskInfo(task_type=BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT,
                            task_id='efr_task',
                            task_config=None)
    efr_dataloader = DummyGSgnnEdgeDataLoader()

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

    efr_dataloaders = [efr_dataloader]
    efr_tasks = [tast_info_efr]

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

    lp_res = np.arange(5)
    lp_length = np.array([5])
    def mock_func_run_lp_mini_batch_predict(*args, **kwargs):
        return lp_res, lp_length

    ntask_res = (np.arange(10), np.arange(10))
    etask_res = (np.arange(20), np.arange(20))
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
            [BUILTIN_TASK_EDGE_CLASSIFICATION,
             BUILTIN_TASK_EDGE_REGRESSION,
             BUILTIN_TASK_RECONSTRUCT_EDGE_FEAT]:
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

        efr_test_dataloader = GSgnnMultiTaskDataLoader(None, efr_tasks, efr_dataloaders)

        target_res = {
            "nc_task":ntask_res,
            "nr_task":ntask_res,
            "ec_task":etask_res,
            "er_task":etask_res,
            "lp_task":lp_res,
            "nfr_task":ntask_res,
            "efr_task":etask_res
        }
        evaluator = MTaskCheckerEvaluator(target_res, target_res, 0)
        mt_infer.setup_evaluator(evaluator)

        mt_infer.infer(data,
                test_dataloader,
                lp_test_dataloader,
                nfr_test_dataloader,
                efr_test_dataloader,
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
                efr_test_dataloader,
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
                None,
                emb_path,
                pred_path,
                use_mini_batch_infer=True,
                node_id_mapping_file=None,
                edge_id_mapping_file=None,
                return_proba=True)

        target_res = {
            "efr_task":etask_res
        }
        evaluator = MTaskCheckerEvaluator(target_res, target_res, 0)
        mt_infer.setup_evaluator(evaluator)

        mt_infer.infer(data,
                None,
                None,
                None,
                efr_test_dataloader,
                emb_path,
                pred_path,
                use_mini_batch_infer=True,
                node_id_mapping_file=None,
                edge_id_mapping_file=None,
                return_proba=True)

    check_eval()

def test_rgcn_infer_nc4ef():
    """ Test RGCN model Node Classification inference pipeline with/without edge features.
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

        # Test case 0: normal case, set RGCN model with edge features for NC, and provide
        #              edge features.
        #              Should complete inference process and save embeddings at /tmp/embs
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model1 = create_builtin_node_gnn_model(gdata.g, config, True)
        inferrer1 = GSgnnNodePredictionInferrer(model1)
        inferrer1.setup_device(device)

        infer_dataloader1 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_test_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            train_task=False)

        inferrer1.infer(
            loader=infer_dataloader1,
            save_embed_path='/tmp/embs',
            save_prediction_path='/tmp/prediction',
            use_mini_batch_infer=True
            )

        target_idx = gdata.get_node_test_set(config.target_ntype)

        embs_path = os.path.join('/tmp/embs', config.target_ntype, 'embed-00000.pt')
        embs_nid_path = os.path.join('/tmp/embs', config.target_ntype, 'embed_nids-00000.pt')
        embs = th.load(embs_path)
        embs_nid = th.load(embs_nid_path)
        
        assert embs.shape[0] == target_idx[config.target_ntype].shape[0]
        assert embs.shape[1] == config.hidden_size
        assert_equal(embs_nid.numpy(), target_idx[config.target_ntype].numpy())

        preds_path = os.path.join('/tmp/prediction', config.target_ntype, 'predict-00000.pt')
        preds_nid_path = os.path.join('/tmp/prediction', config.target_ntype, 'predict_nids-00000.pt')
        predicts = th.load(preds_path)
        predict_nid = th.load(preds_nid_path)

        assert predicts.shape[0] == target_idx[config.target_ntype].shape[0]
        assert predicts.shape[1] == config.num_classes
        assert_equal(predict_nid.numpy(), target_idx[config.target_ntype].numpy())

        # delete temporary results
        if os.path.exists('/tmp/embs'):
            shutil.rmtree('/tmp/embs')
        if os.path.exists('/tmp/prediction'):
            shutil.rmtree('/tmp/prediction')

        # Test case 1: normal case, set RGCN model without edge features for NC, and not
        #              provide edge features.
        #              Should complete inference process and save embeddings at /tmp/embs
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model2 = create_builtin_node_gnn_model(gdata.g, config, True)
        inferrer2 = GSgnnNodePredictionInferrer(model2)
        inferrer2.setup_device(device)

        infer_dataloader2 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_test_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            train_task=False)

        inferrer2.infer(
            loader=infer_dataloader2,
            save_embed_path='/tmp/embs',
            save_prediction_path='/tmp/prediction',
            use_mini_batch_infer=True
            )

        target_idx = gdata.get_node_test_set(config.target_ntype)

        embs_path = os.path.join('/tmp/embs', config.target_ntype, 'embed-00000.pt')
        embs_nid_path = os.path.join('/tmp/embs', config.target_ntype, 'embed_nids-00000.pt')
        embs = th.load(embs_path)
        embs_nid = th.load(embs_nid_path)
        
        assert embs.shape[0] == target_idx[config.target_ntype].shape[0]
        assert embs.shape[1] == config.hidden_size
        assert_equal(embs_nid.numpy(), target_idx[config.target_ntype].numpy())

        preds_path = os.path.join('/tmp/prediction', config.target_ntype, 'predict-00000.pt')
        preds_nid_path = os.path.join('/tmp/prediction', config.target_ntype, 'predict_nids-00000.pt')
        predicts = th.load(preds_path)
        predict_nid = th.load(preds_nid_path)

        assert predicts.shape[0] == target_idx[config.target_ntype].shape[0]
        assert predicts.shape[1] == config.num_classes
        assert_equal(predict_nid.numpy(), target_idx[config.target_ntype].numpy())

        # delete temporary results
        if os.path.exists('/tmp/embs'):
            shutil.rmtree('/tmp/embs')
        if os.path.exists('/tmp/prediction'):
            shutil.rmtree('/tmp/prediction')

        # Test case 2: abnormal case, set RGCN model with edge features for NC, and provide edge
        #              features. But use full graph inference method.
        #              This will trigger an assertion error, saying currently full graph
        #              inference does not support using edge features, should
        #              use mini-batch inferenc.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model3 = create_builtin_node_gnn_model(gdata.g, config, True)
        inferrer3 = GSgnnNodePredictionInferrer(model3)
        inferrer3.setup_device(device)

        infer_dataloader3 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_test_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            train_task=False)

        with assert_raises(AssertionError):
            inferrer3.infer(
                loader=infer_dataloader3,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=False
                )

        # Test case 3: abnormal case, set RGCN model without edge features for NC, but 
        #              provide edge features.
        #              This will trigger an assertion error, asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model4 = create_builtin_node_gnn_model(gdata.g, config, True)
        inferrer4 = GSgnnNodePredictionInferrer(model4)
        inferrer4.setup_device(device)

        infer_dataloader4 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_test_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            train_task=False)

        with assert_raises(AssertionError):
            inferrer4.infer(
                loader=infer_dataloader4,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=True
                )

        # Test case 4: abnormal case, set RGCN model with edge features for NC, but 
        #              not provide edge features.
        #              This will trigger an assertion error, asking for giving edge feature
        #              for message passing computation.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model5 = create_builtin_node_gnn_model(gdata.g, config, True)
        inferrer5 = GSgnnNodePredictionInferrer(model5)
        inferrer5.setup_device(device)

        infer_dataloader5 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_test_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            train_task=False)

        with assert_raises(AssertionError):
            inferrer5.infer(
                loader=infer_dataloader5,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=True
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_rgcn_infer_ec4ef():
    """ Test RGCN model Edge Classification inference pipeline with/without edge features.
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

        # Test case 0: normal case, set RGCN model with edge features for EC, and provide
        #              edge features.
        #              Should complete inference process
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='ec', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model1 = create_builtin_edge_gnn_model(gdata.g, config, True)
        inferrer1 = GSgnnEdgePredictionInferrer(model1)
        inferrer1.setup_device(device)

        infer_dataloader1 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_test_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)
 
        inferrer1.infer(
            loader=infer_dataloader1,
            save_embed_path=None,
            save_prediction_path='/tmp/prediction',
            use_mini_batch_infer=True
            )

        target_idx = gdata.get_edge_test_set(config.target_etype)
        src, dst = gdata.g.find_edges(target_idx[config.target_etype[0]],
                                      etype=config.target_etype[0])

        preds_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                  'predict-00000.pt')
        src_nid_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                    'src_nids-00000.pt')
        dst_nid_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                    'dst_nids-00000.pt')
        predicts = th.load(preds_path)
        src_nid = th.load(src_nid_path)
        dst_nid = th.load(dst_nid_path)

        assert predicts.shape[0] == target_idx[config.target_etype[0]].shape[0]
        assert predicts.shape[1] == config.num_classes
        assert_equal(src.numpy(), src_nid.numpy())
        assert_equal(dst.numpy(), dst_nid.numpy())

        # delete temporary results
        if os.path.exists('/tmp/prediction'):
            shutil.rmtree('/tmp/prediction')

        # Test case 1: normal case, set RGCN model without edge features for EC, and not
        #              provide edge features.
        #              Should complete inference process
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='ec', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model2 = create_builtin_edge_gnn_model(gdata.g, config, True)
        inferrer2 = GSgnnEdgePredictionInferrer(model2)
        inferrer2.setup_device(device)

        infer_dataloader2 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_test_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)
 
        inferrer2.infer(
            loader=infer_dataloader2,
            save_embed_path=None,
            save_prediction_path='/tmp/prediction',
            use_mini_batch_infer=True
            )

        target_idx = gdata.get_edge_test_set(config.target_etype)
        src, dst = gdata.g.find_edges(target_idx[config.target_etype[0]],
                                      etype=config.target_etype[0])

        preds_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                  'predict-00000.pt')
        src_nid_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                    'src_nids-00000.pt')
        dst_nid_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                    'dst_nids-00000.pt')
        predicts = th.load(preds_path)
        src_nid = th.load(src_nid_path)
        dst_nid = th.load(dst_nid_path)

        assert predicts.shape[0] == target_idx[config.target_etype[0]].shape[0]
        assert predicts.shape[1] == config.num_classes
        assert_equal(src.numpy(), src_nid.numpy())
        assert_equal(dst.numpy(), dst_nid.numpy())

        # delete temporary results
        if os.path.exists('/tmp/prediction'):
            shutil.rmtree('/tmp/prediction')

        # Test case 2: abnormal case, set RGCN model with edge features for EC, and provide edge
        #              features. But use full graph inference method.
        #              This will trigger an assertion error, saying currently full graph
        #              inference does not support using edge features, should
        #              use mini-batch inferenc.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='ec', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model3 = create_builtin_edge_gnn_model(gdata.g, config, True)
        inferrer3 = GSgnnEdgePredictionInferrer(model3)
        inferrer3.setup_device(device)

        infer_dataloader3 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_test_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)
 
        with assert_raises(AssertionError):
            inferrer3.infer(
                loader=infer_dataloader3,
                save_embed_path=None,
                use_mini_batch_infer=False
                )

        # Test case 3: abnormal case, set RGCN model without edge features for EC, but 
        #              provide edge features.
        #              This will trigger an assertion error, asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='ec', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model4 = create_builtin_edge_gnn_model(gdata.g, config, True)
        inferrer4 = GSgnnEdgePredictionInferrer(model4)
        inferrer4.setup_device(device)

        infer_dataloader4 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_test_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)
 
        with assert_raises(AssertionError):
            inferrer4.infer(
                loader=infer_dataloader4,
                save_embed_path=None,
                use_mini_batch_infer=True
                )

        # Test case 4: abnormal case, set RGCN model with edge features for EC, but 
        #              not provide edge features.
        #              This will trigger an assertion error, asking for giving edge feature
        #              for message passing computation.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='ec', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model5 = create_builtin_edge_gnn_model(gdata.g, config, True)
        inferrer5 = GSgnnEdgePredictionInferrer(model5)
        inferrer5.setup_device(device)

        infer_dataloader5 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_test_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)
 
        with assert_raises(AssertionError):
            inferrer5.infer(
                loader=infer_dataloader5,
                save_embed_path=None,
                use_mini_batch_infer=True
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_rgcn_infer_lp4ef():
    """ Test RGCN model Link Prediction inference pipeline with/without edge features.
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

        # Test case 0: normal case, set RGCN model with edge features for LP, and provide
        #              edge features.
        #              Should complete inference process
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='lp', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name,  # Need to set these features in
                          edge_feat_field=config.edge_feat_name,  # dataset, as lp_trainer uses
                          )                                        # a different mini-batch method.

        model1 = create_builtin_lp_gnn_model(gdata.g, config, True)
        inferrer1 = GSgnnLinkPredictionInferrer(model1)
        inferrer1.setup_device(device)

        infer_dataloader1 = GSgnnLinkPredictionTestDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.eval_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            num_negative_edges=config.num_negative_edges)

        inferrer1.infer(
                data=gdata,
                loader=infer_dataloader1,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=True
                )

        src_embs_path = os.path.join('/tmp/embs', config.eval_etype[0][0],
                                  'embed-00000.pt')
        src_nid_path = os.path.join('/tmp/embs', config.eval_etype[0][0],
                                    'embed_nids-00000.pt')
        dst_embs_path = os.path.join('/tmp/embs', config.eval_etype[0][2],
                                  'embed-00000.pt')
        dst_nid_path = os.path.join('/tmp/embs', config.eval_etype[0][2],
                                    'embed_nids-00000.pt')        
        src_embs = th.load(src_embs_path)
        dst_embs = th.load(dst_embs_path)
        src_nid = th.load(src_nid_path)
        dst_nid = th.load(dst_nid_path)

        assert src_embs.shape[0] == gdata.g.num_nodes(config.eval_etype[0][0])
        assert dst_embs.shape[0] == gdata.g.num_nodes(config.eval_etype[0][2])
        assert src_embs.shape[1] == config.hidden_size
        assert dst_embs.shape[1] == config.hidden_size
        assert src_nid.shape[0] == gdata.g.num_nodes(config.eval_etype[0][0])
        assert dst_nid.shape[0] == gdata.g.num_nodes(config.eval_etype[0][2])

        # delete temporary results
        if os.path.exists('/tmp/embs'):
            shutil.rmtree('/tmp/embs')

        # Test case 1: normal case, set RGCN model without edge features for LP, and not
        #              provide edge features.
        #              Should complete inference process
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='lp', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name)

        model2 = create_builtin_lp_gnn_model(gdata.g, config, True)
        inferrer2 = GSgnnLinkPredictionInferrer(model2)
        inferrer2.setup_device(device)

        infer_dataloader2 = GSgnnLinkPredictionTestDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,
            num_negative_edges=config.num_negative_edges)

        inferrer2.infer(
                data=gdata,
                loader=infer_dataloader2,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=True
                )

        src_embs_path = os.path.join('/tmp/embs', config.eval_etype[0][0],
                                  'embed-00000.pt')
        src_nid_path = os.path.join('/tmp/embs', config.eval_etype[0][0],
                                    'embed_nids-00000.pt')
        dst_embs_path = os.path.join('/tmp/embs', config.eval_etype[0][2],
                                  'embed-00000.pt')
        dst_nid_path = os.path.join('/tmp/embs', config.eval_etype[0][2],
                                    'embed_nids-00000.pt')        
        src_embs = th.load(src_embs_path)
        dst_embs = th.load(dst_embs_path)
        src_nid = th.load(src_nid_path)
        dst_nid = th.load(dst_nid_path)

        assert src_embs.shape[0] == gdata.g.num_nodes(config.eval_etype[0][0])
        assert dst_embs.shape[0] == gdata.g.num_nodes(config.eval_etype[0][2])
        assert src_embs.shape[1] == config.hidden_size
        assert dst_embs.shape[1] == config.hidden_size
        assert src_nid.shape[0] == gdata.g.num_nodes(config.eval_etype[0][0])
        assert dst_nid.shape[0] == gdata.g.num_nodes(config.eval_etype[0][2])

        # delete temporary results
        if os.path.exists('/tmp/embs'):
            shutil.rmtree('/tmp/embs')

        # Test case 2: abnormal case, set RGCN model with edge features for LP, and provide edge
        #              features. But use full graph inference method.
        #              This will trigger an assertion error, saying currently full graph
        #              inference does not support using edge features, should
        #              use mini-batch inferenc.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='lp', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name,  # Need to set these features in
                          edge_feat_field=config.edge_feat_name,  # dataset, as lp_trainer uses
                          )                                        # a different mini-batch method.

        model3 = create_builtin_lp_gnn_model(gdata.g, config, True)
        inferrer3 = GSgnnLinkPredictionInferrer(model3)
        inferrer3.setup_device(device)

        infer_dataloader3 = GSgnnLinkPredictionTestDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=config.edge_feat_name,
            num_negative_edges=config.num_negative_edges)

        with assert_raises(AssertionError):
            inferrer3.infer(
                data=gdata,
                loader=infer_dataloader3,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=False
                )

        # Test case 3: abnormal case, set RGCN model without edge features for LP, but 
        #              provide edge features.
        #              This will trigger an assertion error, asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='lp', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name,  # Need to set these features in
                          edge_feat_field={('n0', 'r0', 'n1'): ['feat'],
                                           ('n0', 'r1', 'n1'): ['feat']},# Manually set, as 
                                                                         # config does not have it
                          )

        model4 = create_builtin_lp_gnn_model(gdata.g, config, True)
        inferrer4 = GSgnnLinkPredictionInferrer(model4)
        inferrer4.setup_device(device)

        infer_dataloader4 = GSgnnLinkPredictionTestDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,
            num_negative_edges=config.num_negative_edges)

        with assert_raises(AssertionError):
            inferrer4.infer(
                data=gdata,
                loader=infer_dataloader4,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=True
                )

        # Test case 4: abnormal case, set RGCN model with edge features for LP, but 
        #              not provide edge features.
        #              This will trigger an assertion error, asking for giving edge feature
        #              for message passing computation.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', task='lp', use_ef=True)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name)

        model5 = create_builtin_lp_gnn_model(gdata.g, config, True)
        inferrer5 = GSgnnLinkPredictionInferrer(model5)
        inferrer5.setup_device(device)

        infer_dataloader5 = GSgnnLinkPredictionTestDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,
            num_negative_edges=config.num_negative_edges)

        with assert_raises(AssertionError):
            inferrer5.infer(
                data=gdata,
                loader=infer_dataloader5,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=True
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_hgt_infer_nc4ef():
    """ Test HGT model Node Classification inference pipeline with/without edge features.
    
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
        #              Should complete inference process
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model1 = create_builtin_node_gnn_model(gdata.g, config, True)
        inferrer1 = GSgnnNodePredictionInferrer(model1)
        inferrer1.setup_device(device)

        infer_dataloader1 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_test_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            train_task=False)

        inferrer1.infer(
            loader=infer_dataloader1,
            save_embed_path='/tmp/embs',
            use_mini_batch_infer=True
            )

        inferrer1.infer(
            loader=infer_dataloader1,
            save_embed_path='/tmp/embs',
            save_prediction_path='/tmp/prediction',
            use_mini_batch_infer=True
            )

        target_idx = gdata.get_node_test_set(config.target_ntype)

        embs_path = os.path.join('/tmp/embs', config.target_ntype, 'embed-00000.pt')
        embs_nid_path = os.path.join('/tmp/embs', config.target_ntype, 'embed_nids-00000.pt')
        embs = th.load(embs_path)
        embs_nid = th.load(embs_nid_path)
        
        assert embs.shape[0] == target_idx[config.target_ntype].shape[0]
        assert embs.shape[1] == config.hidden_size
        assert_equal(embs_nid.numpy(), target_idx[config.target_ntype].numpy())

        preds_path = os.path.join('/tmp/prediction', config.target_ntype, 'predict-00000.pt')
        preds_nid_path = os.path.join('/tmp/prediction', config.target_ntype, 'predict_nids-00000.pt')
        predicts = th.load(preds_path)
        predict_nid = th.load(preds_nid_path)

        assert predicts.shape[0] == target_idx[config.target_ntype].shape[0]
        assert predicts.shape[1] == config.num_classes
        assert_equal(predict_nid.numpy(), target_idx[config.target_ntype].numpy())

        # delete temporary results
        if os.path.exists('/tmp/embs'):
            shutil.rmtree('/tmp/embs')
        if os.path.exists('/tmp/prediction'):
            shutil.rmtree('/tmp/prediction')

        # Test case 1: normal case, set HGT model without edge features for NC, and not provide
        #              edge features. Use full graph inference method.
        #              Should complete inference process
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        # Full graph inference get node/edfe_feat_field from GSgnnData, so need to set these
        # two fields here
        gdata2 = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name
                          )
        model2 = create_builtin_node_gnn_model(gdata2.g, config, True)
        inferrer2 = GSgnnNodePredictionInferrer(model2)
        inferrer2.setup_device(device)

        infer_dataloader2 = GSgnnNodeDataLoader(
            gdata2,
            target_idx=gdata2.get_node_test_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            train_task=False)

        inferrer2.infer(
            loader=infer_dataloader2,
            save_embed_path='/tmp/embs',
            save_prediction_path='/tmp/prediction',
            use_mini_batch_infer=True
            )

        target_idx = gdata.get_node_test_set(config.target_ntype)

        embs_path = os.path.join('/tmp/embs', config.target_ntype, 'embed-00000.pt')
        embs_nid_path = os.path.join('/tmp/embs', config.target_ntype, 'embed_nids-00000.pt')
        embs = th.load(embs_path)
        embs_nid = th.load(embs_nid_path)
        
        assert embs.shape[0] == target_idx[config.target_ntype].shape[0]
        assert embs.shape[1] == config.hidden_size
        assert_equal(embs_nid.numpy(), target_idx[config.target_ntype].numpy())

        preds_path = os.path.join('/tmp/prediction', config.target_ntype, 'predict-00000.pt')
        preds_nid_path = os.path.join('/tmp/prediction', config.target_ntype, 'predict_nids-00000.pt')
        predicts = th.load(preds_path)
        predict_nid = th.load(preds_nid_path)

        assert predicts.shape[0] == target_idx[config.target_ntype].shape[0]
        assert predicts.shape[1] == config.num_classes
        assert_equal(predict_nid.numpy(), target_idx[config.target_ntype].numpy())

        # delete temporary results
        if os.path.exists('/tmp/embs'):
            shutil.rmtree('/tmp/embs')
        if os.path.exists('/tmp/prediction'):
            shutil.rmtree('/tmp/prediction')

        # Test case 2: abnormal case, set HGT model without edge features for NC, but provide
        #              edge features. Use mini-batch inference
        #              Should trigger an assertion error， asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model3 = create_builtin_node_gnn_model(gdata.g, config, True)
        inferrer3 = GSgnnNodePredictionInferrer(model3)
        inferrer3.setup_device(device)

        infer_dataloader3 = GSgnnNodeDataLoader(
            gdata,
            target_idx=gdata.get_node_test_set(config.target_ntype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            train_task=False)

        with assert_raises(AssertionError):
            inferrer3.infer(
                loader=infer_dataloader3,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=True
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_hgt_infer_ec4ef():
    """ Test HGT model Edge Classification inference pipeline with/without edge features.
    
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
        #              Should complete inference process
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt',
                         task='ec', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model1 = create_builtin_edge_gnn_model(gdata.g, config, True)
        inferrer1 = GSgnnEdgePredictionInferrer(model1)
        inferrer1.setup_device(device)

        infer_dataloader1 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_test_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)
 
        inferrer1.infer(
            loader=infer_dataloader1,
            save_embed_path=None,
            save_prediction_path='/tmp/prediction',
            use_mini_batch_infer=True
            )

        target_idx = gdata.get_edge_test_set(config.target_etype)
        src, dst = gdata.g.find_edges(target_idx[config.target_etype[0]],
                                      etype=config.target_etype[0])

        preds_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                  'predict-00000.pt')
        src_nid_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                    'src_nids-00000.pt')
        dst_nid_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                    'dst_nids-00000.pt')
        predicts = th.load(preds_path)
        src_nid = th.load(src_nid_path)
        dst_nid = th.load(dst_nid_path)

        assert predicts.shape[0] == target_idx[config.target_etype[0]].shape[0]
        assert predicts.shape[1] == config.num_classes
        assert_equal(src.numpy(), src_nid.numpy())
        assert_equal(dst.numpy(), dst_nid.numpy())

        # delete temporary results
        if os.path.exists('/tmp/prediction'):
            shutil.rmtree('/tmp/prediction')

        # Test case 1: normal case, set HGT model without edge features for EC, and not provide
        #              edge features. Use full graph inference method.
        #              Should complete inference process
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt',
                         task='ec', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        # Full graph inference get node/edfe_feat_field from GSgnnData, so need to set these
        # two fields here
        gdata2 = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name
                          )
        model2 = create_builtin_edge_gnn_model(gdata2.g, config, True)
        inferrer2 = GSgnnEdgePredictionInferrer(model2)
        inferrer2.setup_device(device)

        infer_dataloader2 = GSgnnEdgeDataLoader(
            gdata2,
            target_idx=gdata2.get_edge_test_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats=None,
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)
 
        inferrer2.infer(
            loader=infer_dataloader2,
            save_embed_path=None,
            save_prediction_path='/tmp/prediction',
            use_mini_batch_infer=True
            )

        target_idx = gdata.get_edge_test_set(config.target_etype)
        src, dst = gdata.g.find_edges(target_idx[config.target_etype[0]],
                                      etype=config.target_etype[0])

        preds_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                  'predict-00000.pt')
        src_nid_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                    'src_nids-00000.pt')
        dst_nid_path = os.path.join('/tmp/prediction', '_'.join(config.target_etype[0]),
                                    'dst_nids-00000.pt')
        predicts = th.load(preds_path)
        src_nid = th.load(src_nid_path)
        dst_nid = th.load(dst_nid_path)

        assert predicts.shape[0] == target_idx[config.target_etype[0]].shape[0]
        assert predicts.shape[1] == config.num_classes
        assert_equal(src.numpy(), src_nid.numpy())
        assert_equal(dst.numpy(), dst_nid.numpy())

        # delete temporary results
        if os.path.exists('/tmp/prediction'):
            shutil.rmtree('/tmp/prediction')

        # Test case 2: abnormal case, set HGT model without edge features for EC, but provide
        #              edge features. Use mini-batch inference
        #              Should trigger an assertion error， asking for projection weights
        #              in the GSEdgeEncoderInputLayer.
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt',
                         task='ec', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)

        model3 = create_builtin_edge_gnn_model(gdata.g, config, True)
        inferrer3 = GSgnnEdgePredictionInferrer(model3)
        inferrer3.setup_device(device)

        infer_dataloader3 = GSgnnEdgeDataLoader(
            gdata,
            target_idx=gdata.get_edge_test_set(config.target_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            label_field=config.label_field,
            node_feats=config.node_feat_name,
            edge_feats={('n0', 'r0', 'n1'): ['feat'],
                        ('n0', 'r1', 'n1'): ['feat']},  # Manually set, as config does not have it.
            remove_target_edge_type=config.remove_target_edge_type,
            train_task=False)
 
        with assert_raises(AssertionError):
            inferrer3.infer(
                loader=infer_dataloader3,
                save_embed_path=None,
                use_mini_batch_infer=True
                )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_hgt_infer_lp4ef():
    """ Test HGT model Link Prediction inference pipeline with/without edge features.
    
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
        #              edge features. Use mini-batch inference
        #              Should complete inference process
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt',
                         task='lp', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name)

        model1 = create_builtin_lp_gnn_model(gdata.g, config, True)
        inferrer1 = GSgnnLinkPredictionInferrer(model1)
        inferrer1.setup_device(device)

        infer_dataloader1 = GSgnnLinkPredictionTestDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,  # Because LP use gdata to extract feature, this
                              # setting does not change results
            num_negative_edges=config.num_negative_edges)

        inferrer1.infer(
                data=gdata,
                loader=infer_dataloader1,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=True
                )

        src_embs_path = os.path.join('/tmp/embs', config.eval_etype[0][0],
                                  'embed-00000.pt')
        src_nid_path = os.path.join('/tmp/embs', config.eval_etype[0][0],
                                    'embed_nids-00000.pt')
        dst_embs_path = os.path.join('/tmp/embs', config.eval_etype[0][2],
                                  'embed-00000.pt')
        dst_nid_path = os.path.join('/tmp/embs', config.eval_etype[0][2],
                                    'embed_nids-00000.pt')        
        src_embs = th.load(src_embs_path)
        dst_embs = th.load(dst_embs_path)
        src_nid = th.load(src_nid_path)
        dst_nid = th.load(dst_nid_path)

        assert src_embs.shape[0] == gdata.g.num_nodes(config.eval_etype[0][0])
        assert dst_embs.shape[0] == gdata.g.num_nodes(config.eval_etype[0][2])
        assert src_embs.shape[1] == config.hidden_size
        assert dst_embs.shape[1] == config.hidden_size
        assert src_nid.shape[0] == gdata.g.num_nodes(config.eval_etype[0][0])
        assert dst_nid.shape[0] == gdata.g.num_nodes(config.eval_etype[0][2])

        # delete temporary results
        if os.path.exists('/tmp/embs'):
            shutil.rmtree('/tmp/embs')

        # Test case 1: normal case, set HGT model without edge features for LP, and not provide
        #              edge features. Use full graph inference method.
        #              Should complete inference process
        create_config4ef(Path(tmpdirname), 'gnn_nc.yaml', encoder='hgt',
                         task='lp', use_ef=False)
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                            local_rank=0)
        config = GSConfig(args)
        gdata = GSgnnData(part_config=part_config,
                          node_feat_field=config.node_feat_name)

        model2 = create_builtin_lp_gnn_model(gdata.g, config, True)
        inferrer2 = GSgnnLinkPredictionInferrer(model2)
        inferrer2.setup_device(device)

        infer_dataloader2 = GSgnnLinkPredictionTestDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,  # Because LP use gdata to extract feature, this
                              # setting does not change results
            num_negative_edges=config.num_negative_edges)

        inferrer2.infer(
                data=gdata,
                loader=infer_dataloader2,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=True
                )

        src_embs_path = os.path.join('/tmp/embs', config.eval_etype[0][0],
                                  'embed-00000.pt')
        src_nid_path = os.path.join('/tmp/embs', config.eval_etype[0][0],
                                    'embed_nids-00000.pt')
        dst_embs_path = os.path.join('/tmp/embs', config.eval_etype[0][2],
                                  'embed-00000.pt')
        dst_nid_path = os.path.join('/tmp/embs', config.eval_etype[0][2],
                                    'embed_nids-00000.pt')        
        src_embs = th.load(src_embs_path)
        dst_embs = th.load(dst_embs_path)
        src_nid = th.load(src_nid_path)
        dst_nid = th.load(dst_nid_path)

        assert src_embs.shape[0] == gdata.g.num_nodes(config.eval_etype[0][0])
        assert dst_embs.shape[0] == gdata.g.num_nodes(config.eval_etype[0][2])
        assert src_embs.shape[1] == config.hidden_size
        assert dst_embs.shape[1] == config.hidden_size
        assert src_nid.shape[0] == gdata.g.num_nodes(config.eval_etype[0][0])
        assert dst_nid.shape[0] == gdata.g.num_nodes(config.eval_etype[0][2])

        # delete temporary results
        if os.path.exists('/tmp/embs'):
            shutil.rmtree('/tmp/embs')

        # Test case 2: abnormal case, set HGT model without edge features for LP, but provide
        #              edge features. Use mini-batch inference
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
        inferrer3 = GSgnnLinkPredictionInferrer(model3)
        inferrer3.setup_device(device)

        infer_dataloader3 = GSgnnLinkPredictionTestDataLoader(
            gdata,
            target_idx=gdata.get_edge_val_set(config.train_etype),
            fanout=config.fanout,
            batch_size=config.batch_size,
            node_feats=config.node_feat_name,
            edge_feats=None,  # Because LP use gdata to extract feature, this
                              # setting does not change results
            num_negative_edges=config.num_negative_edges)

        with assert_raises(AssertionError):
            inferrer3.infer(
                data=gdata,
                loader=infer_dataloader3,
                save_embed_path='/tmp/embs',
                use_mini_batch_infer=True
            )

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()


if __name__ == '__main__':
    test_mtask_infer()

    test_inferrer_setup_evaluator()

    test_rgcn_infer_nc4ef()
    test_rgcn_infer_ec4ef()
    test_rgcn_infer_lp4ef()
    test_hgt_infer_nc4ef()
    test_hgt_infer_ec4ef()
    test_hgt_infer_lp4ef()
