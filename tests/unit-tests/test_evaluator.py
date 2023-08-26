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
from unittest.mock import patch, MagicMock
import operator

import torch as th
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import dgl

from graphstorm.eval import GSgnnMrrLPEvaluator, GSgnnPerEtypeMrrLPEvaluator
from graphstorm.eval import GSgnnAccEvaluator
from graphstorm.eval import GSgnnRegressionEvaluator
from graphstorm.eval.evaluator import early_stop_avg_increase_judge
from graphstorm.eval.evaluator import early_stop_cons_increase_judge
from graphstorm.config.config import EARLY_STOP_AVERAGE_INCREASE_STRATEGY
from graphstorm.config.config import EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY
from graphstorm.config import BUILTIN_LP_DOT_DECODER
from graphstorm.config.config import LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL


from util import Dummy

def gen_hg():
    num_nodes_dict = {
        "n0": 100,
        "n1": 100,
    }

    edges = {
        ("n0", "r0", "n1"): (th.randint(100, (100,)), th.randint(100, (100,))),
        ("n0", "r1", "n1"): (th.randint(100, (200,)), th.randint(100, (200,))),
    }

    hg = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)
    return hg

def gen_mrr_lp_eval_data():
    # common Dummy objects
    train_data = Dummy({
            "train_idxs": th.randint(10, (10,)),
            "val_idxs": th.randint(10, (10,)),
            "test_idxs": th.randint(10, (10,)),
            "do_validation": True
        })

    config = Dummy({
            "num_negative_edges_eval": 10,
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "eval_frequency": 100,
            "use_early_stop": False,
        })

    etypes = [("n0", "r0", "n1"), ("n0", "r1", "n1")]
    # test compute_score
    val_pos_scores = th.rand((10,1))
    val_neg_scores = th.rand((10,10))
    test_pos_scores = th.rand((10,1))
    test_neg_scores = th.rand((10,10))

    return train_data, config, etypes, (val_pos_scores, val_neg_scores), (test_pos_scores, test_neg_scores)

def test_mrr_per_etype_lp_evaluation():
    # system heavily depends on th distributed
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=1,
                                      rank=0)
    train_data, config, etypes, val_scores, test_scores = gen_mrr_lp_eval_data()

    score = {
        ("a", "r1", "b"): 0.9,
        ("a", "r2", "b"): 0.8,
    }

    # Test get_major_score
    lp = GSgnnPerEtypeMrrLPEvaluator(10,
        train_data,
        num_negative_edges_eval=4,
        lp_decoder_type=BUILTIN_LP_DOT_DECODER,
        use_early_stop=False)
    assert lp.major_etype == LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL

    m_score = lp._get_major_score(score)
    assert m_score == sum(score.values()) / 2

    # Test get_major_score
    lp = GSgnnPerEtypeMrrLPEvaluator(config.eval_frequency,
        train_data,
        major_etype=("a", "r2", "b"),
        num_negative_edges_eval=config.num_negative_edges_eval,
        lp_decoder_type=config.lp_decoder_type,
        use_early_stop=config.use_early_stop)
    assert lp.major_etype == ("a", "r2", "b")

    m_score = lp._get_major_score(score)
    assert m_score == score[("a", "r2", "b")]

    val_pos_scores, val_neg_scores = val_scores
    test_pos_scores, test_neg_scores = test_scores

    lp = GSgnnPerEtypeMrrLPEvaluator(config.eval_frequency,
        train_data,
        num_negative_edges_eval=config.num_negative_edges_eval,
        lp_decoder_type=config.lp_decoder_type,
        use_early_stop=config.use_early_stop)

    rank0 = []
    rank1 = []
    for i in range(len(val_pos_scores)):
        val_pos = val_pos_scores[i]
        val_neg0 = val_neg_scores[i] / 2
        val_neg1 = val_neg_scores[i] / 4
        scores = th.cat([val_pos, val_neg0])
        _, indices = th.sort(scores, descending=True)
        ranking = th.nonzero(indices == 0) + 1
        rank0.append(ranking.cpu().detach())
        rank0.append(ranking.cpu().detach())
        scores = th.cat([val_pos, val_neg1])
        _, indices = th.sort(scores, descending=True)
        ranking = th.nonzero(indices == 0) + 1
        rank1.append(ranking.cpu().detach())
    val_ranks = {etypes[0]: th.cat(rank0, dim=0), etypes[1]: th.cat(rank1, dim=0)}
    val_s = lp.compute_score(val_ranks)
    mrr = 1.0/val_ranks[etypes[0]]
    mrr = th.sum(mrr) / len(mrr)
    assert_almost_equal(val_s['mrr'][etypes[0]], mrr.numpy(), decimal=7)
    mrr = 1.0/val_ranks[etypes[1]]
    mrr = th.sum(mrr) / len(mrr)
    assert_almost_equal(val_s['mrr'][etypes[1]], mrr.numpy(), decimal=7)

    rank0 = []
    rank1 = []
    for i in range(len(test_pos_scores)):
        val_pos = test_pos_scores[i]
        val_neg0 = test_neg_scores[i] / 2
        val_neg1 = test_neg_scores[i] / 4
        scores = th.cat([val_pos, val_neg0])
        _, indices = th.sort(scores, descending=True)
        ranking = th.nonzero(indices == 0) + 1
        rank0.append(ranking.cpu().detach())
        rank0.append(ranking.cpu().detach())
        scores = th.cat([val_pos, val_neg1])
        _, indices = th.sort(scores, descending=True)
        ranking = th.nonzero(indices == 0) + 1
        rank1.append(ranking.cpu().detach())
    test_ranks =  {etypes[0]: th.cat(rank0, dim=0), etypes[1]: th.cat(rank1, dim=0)}
    test_s = lp.compute_score(test_ranks)
    mrr = 1.0/test_ranks[etypes[0]]
    mrr = th.sum(mrr) / len(mrr)
    assert_almost_equal(np.array([test_s['mrr'][etypes[0]]]), mrr.numpy(), decimal=7)
    mrr = 1.0/test_ranks[etypes[1]]
    mrr = th.sum(mrr) / len(mrr)
    assert_almost_equal(np.array([test_s['mrr'][etypes[1]]]), mrr.numpy(), decimal=7)

    val_sc, test_sc = lp.evaluate(val_ranks, test_ranks, 0)
    val_s_mrr = (val_s['mrr'][etypes[0]] + val_s['mrr'][etypes[1]]) / 2
    test_s_mrr = (test_s['mrr'][etypes[0]] + test_s['mrr'][etypes[1]]) / 2
    assert_equal(val_s['mrr'][etypes[0]], val_sc['mrr'][etypes[0]])
    assert_equal(val_s['mrr'][etypes[1]], val_sc['mrr'][etypes[1]])
    assert_equal(test_s['mrr'][etypes[0]], test_sc['mrr'][etypes[0]])
    assert_equal(test_s['mrr'][etypes[1]], test_sc['mrr'][etypes[1]])

    assert_almost_equal(np.array([val_s_mrr]), lp.best_val_score['mrr'])
    assert_almost_equal(np.array([test_s_mrr]), lp.best_test_score['mrr'])

    lp = GSgnnPerEtypeMrrLPEvaluator(config.eval_frequency,
        train_data,
        major_etype=etypes[1],
        num_negative_edges_eval=config.num_negative_edges_eval,
        lp_decoder_type=config.lp_decoder_type,
        use_early_stop=config.use_early_stop)

    val_sc, test_sc = lp.evaluate(val_ranks, test_ranks, 0)
    assert_equal(val_s['mrr'][etypes[0]], val_sc['mrr'][etypes[0]])
    assert_equal(val_s['mrr'][etypes[1]], val_sc['mrr'][etypes[1]])
    assert_equal(test_s['mrr'][etypes[0]], test_sc['mrr'][etypes[0]])
    assert_equal(test_s['mrr'][etypes[1]], test_sc['mrr'][etypes[1]])

    assert_almost_equal(val_s['mrr'][etypes[1]], lp.best_val_score['mrr'])
    assert_almost_equal(test_s['mrr'][etypes[1]], lp.best_test_score['mrr'])

    th.distributed.destroy_process_group()

def test_mrr_lp_evaluator():
    # system heavily depends on th distributed
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=1,
                                      rank=0)
    train_data, config, etypes, val_scores, test_scores = gen_mrr_lp_eval_data()
    val_pos_scores, val_neg_scores = val_scores
    test_pos_scores, test_neg_scores = test_scores

    lp = GSgnnMrrLPEvaluator(config.eval_frequency,
                             train_data,
                             num_negative_edges_eval=config.num_negative_edges_eval,
                             lp_decoder_type=config.lp_decoder_type,
                             use_early_stop=config.use_early_stop)

    rank = []
    for i in range(len(val_pos_scores)):
        val_pos = val_pos_scores[i]
        val_neg0 = val_neg_scores[i] / 2
        val_neg1 = val_neg_scores[i] / 4
        scores = th.cat([val_pos, val_neg0])
        _, indices = th.sort(scores, descending=True)
        ranking = th.nonzero(indices == 0) + 1
        rank.append(ranking.cpu().detach())
        rank.append(ranking.cpu().detach())
        scores = th.cat([val_pos, val_neg1])
        _, indices = th.sort(scores, descending=True)
        ranking = th.nonzero(indices == 0) + 1
        rank.append(ranking.cpu().detach())
    val_ranks = {etypes[0]: th.cat(rank, dim=0)}
    val_s = lp.compute_score(val_ranks)
    mrr = 1.0/val_ranks[etypes[0]]
    mrr = th.sum(mrr) / len(mrr)
    assert_almost_equal(val_s['mrr'], mrr.numpy(), decimal=7)

    rank = []
    for i in range(len(test_pos_scores)):
        val_pos = test_pos_scores[i]
        val_neg0 = test_neg_scores[i] / 2
        val_neg1 = test_neg_scores[i] / 4
        scores = th.cat([val_pos, val_neg0])
        _, indices = th.sort(scores, descending=True)
        ranking = th.nonzero(indices == 0) + 1
        rank.append(ranking.cpu().detach())
        rank.append(ranking.cpu().detach())
        scores = th.cat([val_pos, val_neg1])
        _, indices = th.sort(scores, descending=True)
        ranking = th.nonzero(indices == 0) + 1
        rank.append(ranking.cpu().detach())
    test_ranks =  {etypes[0]: th.cat(rank, dim=0)}
    test_s = lp.compute_score(test_ranks)
    mrr = 1.0/test_ranks[etypes[0]]
    mrr = th.sum(mrr) / len(mrr)
    assert_almost_equal(test_s['mrr'], mrr.numpy(), decimal=7)

    val_sc, test_sc = lp.evaluate(val_ranks, test_ranks, 0)
    assert_equal(val_s['mrr'], val_sc['mrr'])
    assert_equal(test_s['mrr'], test_sc['mrr'])

    # val_ranks is None
    val_sc, test_sc = lp.evaluate(None, test_ranks, 0)
    assert_equal(val_sc['mrr'], "N/A")
    assert_equal(test_s['mrr'], test_sc['mrr'])

    # test_ranks is None
    val_sc, test_sc = lp.evaluate(val_ranks, None, 0)
    assert_equal(val_s['mrr'], val_sc['mrr'])
    assert_equal(test_sc['mrr'], "N/A")

    # test evaluate
    @patch.object(GSgnnMrrLPEvaluator, 'compute_score')
    def check_evaluate(mock_compute_score):
        lp = GSgnnMrrLPEvaluator(config.eval_frequency,
                                 train_data,
                                 num_negative_edges_eval=config. num_negative_edges_eval,
                                 lp_decoder_type=config.lp_decoder_type,
                                 use_early_stop=config.use_early_stop)

        mock_compute_score.side_effect = [
            {"mrr": 0.6},
            {"mrr": 0.7},
            {"mrr": 0.65},
            {"mrr": 0.8},
            {"mrr": 0.8},
            {"mrr": 0.7}
        ]

        val_score, test_score = lp.evaluate(
            {("u", "b", "v") : ()}, {("u", "b", "v") : ()}, 100)
        assert val_score["mrr"] == 0.7
        assert test_score["mrr"] == 0.6
        val_score, test_score = lp.evaluate(
            {("u", "b", "v") : ()}, {("u", "b", "v") : ()}, 200)
        assert val_score["mrr"] == 0.8
        assert test_score["mrr"] == 0.65
        val_score, test_score = lp.evaluate(
            {("u", "b", "v") : ()}, {("u", "b", "v") : ()}, 300)
        assert val_score["mrr"] == 0.7
        assert test_score["mrr"] == 0.8

        assert lp.best_val_score["mrr"] == 0.8
        assert lp.best_test_score["mrr"] == 0.65
        assert lp.best_iter_num["mrr"] == 200

    # check GSgnnMrrLPEvaluator.evaluate()
    check_evaluate()

    # common Dummy objects
    train_data = Dummy({
            "train_idxs": None,
            "val_idxs": None,
            "test_idxs": th.randint(10, (10,)),
            "do_validation": True
        })
    # test evaluate
    @patch.object(GSgnnMrrLPEvaluator, 'compute_score')
    def check_evaluate_infer(mock_compute_score):
        lp = GSgnnMrrLPEvaluator(config.eval_frequency,
                                 train_data,
                                 num_negative_edges_eval=config.num_negative_edges_eval,
                                 lp_decoder_type=config.lp_decoder_type,
                                 use_early_stop=config.use_early_stop)

        mock_compute_score.side_effect = [
            {"mrr": 0.6},
            {"mrr": 0.7},
        ]

        val_score, test_score = lp.evaluate(None, [], 100)
        assert val_score["mrr"] == "N/A"
        assert test_score["mrr"] == 0.6
        val_score, test_score = lp.evaluate(None, [], 200)
        assert val_score["mrr"] == "N/A"
        assert test_score["mrr"] == 0.7

        assert lp.best_val_score["mrr"] == 0 # Still initial value 0
        assert lp.best_test_score["mrr"] == 0 # Still initial value 0
        assert lp.best_iter_num["mrr"] == 0 # Still initial value 0

    # check GSgnnMrrLPEvaluator.evaluate()
    check_evaluate_infer()

    # check GSgnnMrrLPEvaluator.do_eval()
    # train_data.do_validation True
    # config.no_validation False
    lp = GSgnnMrrLPEvaluator(config.eval_frequency,
                             train_data,
                             num_negative_edges_eval=config.num_negative_edges_eval,
                             lp_decoder_type=config.lp_decoder_type,
                             use_early_stop=config.use_early_stop)
    assert lp.do_eval(120, epoch_end=True) is True
    assert lp.do_eval(200) is True
    assert lp.do_eval(0) is True
    assert lp.do_eval(1) is False

    config3 = Dummy({
            "num_negative_edges_eval": 10,
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "eval_frequency": 0,
            "use_early_stop": False,
        })

    # train_data.do_validation True
    # config.no_validation False
    # eval_frequency is 0
    lp = GSgnnMrrLPEvaluator(config3.eval_frequency,
                             train_data,
                             num_negative_edges_eval=config3.num_negative_edges_eval,
                             lp_decoder_type=config3.lp_decoder_type,
                             use_early_stop=config3.use_early_stop)
    assert lp.do_eval(120, epoch_end=True) is True
    assert lp.do_eval(200) is False

    th.distributed.destroy_process_group()

def test_acc_evaluator():
    # system heavily depends on th distributed
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=1,
                                      rank=0)

    config = Dummy({
            "multilabel": False,
            "eval_frequency": 100,
            "eval_metric": ["accuracy"],
            "use_early_stop": False,
        })

    # Test compute_score
    nc = GSgnnAccEvaluator(config.eval_frequency,
                           config.eval_metric,
                           config.multilabel,
                           config.use_early_stop)
    pred = th.randint(10, (100,))
    labels = th.randint(10, (100,))
    result = nc.compute_score(pred, labels, True)
    assert_equal(result["accuracy"],
                 th.sum(pred == labels).item() / len(labels))

    result = nc.compute_score(None, None, True)
    assert result["accuracy"] == "N/A"

    # Test evaluate
    @patch.object(GSgnnAccEvaluator, 'compute_score')
    def check_evaluate(mock_compute_score):
        nc = GSgnnAccEvaluator(config.eval_frequency,
                               config.eval_metric,
                               config.multilabel,
                               config.use_early_stop)
        mock_compute_score.side_effect = [
            {"accuracy": 0.7},
            {"accuracy": 0.65},
            {"accuracy": 0.8},
            {"accuracy": 0.7},
            {"accuracy": 0.76},
            {"accuracy": 0.8},
        ]
        val_score, test_score = nc.evaluate(th.rand((10,)), th.rand((10,)), th.rand((10,)), th.rand((10,)), 100)
        mock_compute_score.assert_called()
        assert val_score["accuracy"] == 0.7
        assert test_score["accuracy"] == 0.65

        val_score, test_score = nc.evaluate(th.rand((10,)), th.rand((10,)), th.rand((10,)), th.rand((10,)), 200)
        mock_compute_score.assert_called()
        assert val_score["accuracy"] == 0.8
        assert test_score["accuracy"] == 0.7

        val_score, test_score = nc.evaluate(th.rand((10,)), th.rand((10,)), th.rand((10,)), th.rand((10,)), 300)
        mock_compute_score.assert_called()
        assert val_score["accuracy"] == 0.76
        assert test_score["accuracy"] == 0.8

        assert nc.best_val_score["accuracy"] == 0.8
        assert nc.best_test_score["accuracy"] == 0.7
        assert nc.best_iter_num["accuracy"] == 200

    check_evaluate()

    # Test evaluate with out test score
    @patch.object(GSgnnAccEvaluator, 'compute_score')
    def check_evaluate_no_test(mock_compute_score):
        nc = GSgnnAccEvaluator(config.eval_frequency,
                               config.eval_metric,
                               config.multilabel,
                               config.use_early_stop)
        mock_compute_score.side_effect = [
            {"accuracy": 0.7},
            {"accuracy": "N/A"},
            {"accuracy": 0.8},
            {"accuracy": "N/A"},
            {"accuracy": 0.76},
            {"accuracy": "N/A"},
        ]
        val_score, test_score = nc.evaluate(th.rand((10,)), None, th.rand((10,)), None, 100)
        mock_compute_score.assert_called()
        assert val_score["accuracy"] == 0.7
        assert test_score["accuracy"] == "N/A"

        val_score, test_score = nc.evaluate(th.rand((10,)), None, th.rand((10,)), None, 200)
        mock_compute_score.assert_called()
        assert val_score["accuracy"] == 0.8
        assert test_score["accuracy"] == "N/A"

        val_score, test_score = nc.evaluate(th.rand((10,)), None, th.rand((10,)), None, 300)
        mock_compute_score.assert_called()
        assert val_score["accuracy"] == 0.76
        assert test_score["accuracy"] == "N/A"

        assert nc.best_val_score["accuracy"] == 0.8
        assert nc.best_test_score["accuracy"] == "N/A"
        assert nc.best_iter_num["accuracy"] == 200

    check_evaluate_no_test()

    # check GSgnnAccEvaluator.do_eval()
    # train_data.do_validation True
    # config.no_validation False
    nc = GSgnnAccEvaluator(config.eval_frequency,
                           config.eval_metric,
                           config.multilabel,
                           config.use_early_stop)
    assert nc.do_eval(120, epoch_end=True) is True
    assert nc.do_eval(200) is True
    assert nc.do_eval(0) is True
    assert nc.do_eval(1) is False

    config3 = Dummy({
            "multilabel": False,
            "eval_frequency": 0,
            "eval_metric": ["accuracy"],
            "use_early_stop": False,
        })

    # train_data.do_validation True
    # config.no_validation False
    # eval_frequency is 0
    nc = GSgnnAccEvaluator(config3.eval_frequency,
                           config3.eval_metric,
                           config3.multilabel,
                           config3.use_early_stop)
    assert nc.do_eval(120, epoch_end=True) is True
    assert nc.do_eval(200) is False
    th.distributed.destroy_process_group()

def test_regression_evaluator():
    # system heavily depends on th distributed
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=1,
                                      rank=0)

    config = Dummy({
            "eval_frequency": 100,
            "eval_metric": ["rmse"],
            "use_early_stop": False,
        })

    # Test compute_score
    nr = GSgnnRegressionEvaluator(config.eval_frequency,
                                  config.eval_metric,
                                  config.use_early_stop)
    pred = th.rand(100)
    labels = th.rand(100)
    result = nr.compute_score(pred, labels)
    diff = pred - labels
    assert_equal(result["rmse"],
                 th.sqrt(th.mean(diff * diff)).cpu().item())

    result = nr.compute_score(None, None)
    assert result["rmse"] == "N/A"

    # Test evaluate
    @patch.object(GSgnnRegressionEvaluator, 'compute_score')
    def check_evaluate(mock_compute_score):
        nr = GSgnnRegressionEvaluator(config.eval_frequency,
                                      config.eval_metric,
                                      config.use_early_stop)
        mock_compute_score.side_effect = [
            {"rmse": 0.7},
            {"rmse": 0.8},
            {"rmse": 0.2},
            {"rmse": 0.23},
            {"rmse": 0.3},
            {"rmse": 0.31},
        ]

        val_score, test_score = nr.evaluate(th.rand((10,)), th.rand((10,)), th.rand((10,)), th.rand((10,)), 100)
        mock_compute_score.assert_called()
        assert val_score["rmse"] == 0.7
        assert test_score["rmse"] == 0.8

        val_score, test_score = nr.evaluate(th.rand((10,)), th.rand((10,)), th.rand((10,)), th.rand((10,)), 300)
        mock_compute_score.assert_called()
        assert val_score["rmse"] == 0.2
        assert test_score["rmse"] == 0.23

        val_score, test_score = nr.evaluate(th.rand((10,)), th.rand((10,)), th.rand((10,)), th.rand((10,)), 500)
        mock_compute_score.assert_called()
        assert val_score["rmse"] == 0.3
        assert test_score["rmse"] == 0.31

        assert nr.best_val_score["rmse"] == 0.2
        assert nr.best_test_score["rmse"] == 0.23
        assert nr.best_iter_num["rmse"] == 300

    check_evaluate()

    # Test evaluate without test set
    @patch.object(GSgnnRegressionEvaluator, 'compute_score')
    def check_evaluate_no_test(mock_compute_score):
        nr = GSgnnRegressionEvaluator(config.eval_frequency,
                                      config.eval_metric,
                                      config.use_early_stop)
        mock_compute_score.side_effect = [
            {"rmse": 0.7},
            {"rmse": "N/A"},
            {"rmse": 0.2},
            {"rmse": "N/A"},
            {"rmse": 0.3},
            {"rmse": "N/A"},
        ]

        val_score, test_score = nr.evaluate(th.rand((10,)), None, th.rand((10,)), None, 100)
        mock_compute_score.assert_called()
        assert val_score["rmse"] == 0.7
        assert test_score["rmse"] == "N/A"

        val_score, test_score = nr.evaluate(th.rand((10,)), None, th.rand((10,)), None, 300)
        mock_compute_score.assert_called()
        assert val_score["rmse"] == 0.2
        assert test_score["rmse"] == "N/A"

        val_score, test_score = nr.evaluate(th.rand((10,)), None, th.rand((10,)), None, 500)
        mock_compute_score.assert_called()
        assert val_score["rmse"] == 0.3
        assert test_score["rmse"] == "N/A"

        assert nr.best_val_score["rmse"] == 0.2
        assert nr.best_test_score["rmse"] == "N/A"
        assert nr.best_iter_num["rmse"] == 300

    check_evaluate_no_test()

    # check GSgnnRegressionEvaluator.do_eval()
    # train_data.do_validation True
    nr = GSgnnRegressionEvaluator(config.eval_frequency,
                                  config.eval_metric,
                                  config.use_early_stop)
    assert nr.do_eval(120, epoch_end=True) is True
    assert nr.do_eval(200) is True
    assert nr.do_eval(0) is True
    assert nr.do_eval(1) is False

    config3 = Dummy({
            "eval_frequency": 0,
            "no_validation": False,
            "eval_metric": ["rmse"],
            "use_early_stop": False,
        })

    # train_data.do_validation True
    # eval_frequency is 0
    nr = GSgnnRegressionEvaluator(config3.eval_frequency,
                                  config3.eval_metric,
                                  config3.use_early_stop)
    assert nr.do_eval(120, epoch_end=True) is True
    assert nr.do_eval(200) is False
    th.distributed.destroy_process_group()

def test_early_stop_avg_increase_judge():
    comparator = operator.le
    val_score = 0.5
    val_perf_list = [0.4, 0.45, 0.6]
    assert early_stop_avg_increase_judge(val_score, val_perf_list, comparator) is False
    val_score = 0.4
    assert early_stop_avg_increase_judge(val_score, val_perf_list, comparator)

    comparator = operator.ge
    val_score = 0.4
    val_perf_list = [0.4, 0.45, 0.6]
    assert early_stop_avg_increase_judge(val_score, val_perf_list, comparator) is False
    val_score = 0.5
    assert early_stop_avg_increase_judge(val_score, val_perf_list, comparator)

def test_early_stop_cons_increase_judge():
    comparator = operator.le
    val_score = 0.5
    val_perf_list = [0.6, 0.45, 0.6]
    assert early_stop_cons_increase_judge(val_score, val_perf_list, comparator) is False
    val_score = 0.4
    assert early_stop_cons_increase_judge(val_score, val_perf_list, comparator)

    comparator = operator.ge
    val_score = 0.5
    val_perf_list = [0.45, 0.45, 0.55]
    assert early_stop_cons_increase_judge(val_score, val_perf_list, comparator) is False
    val_score = 0.55
    assert early_stop_cons_increase_judge(val_score, val_perf_list, comparator)

def test_early_stop_evaluator():
    # common Dummy objects
    config = Dummy({
            "eval_frequency": 100,
            "eval_metric": ["rmse"],
            "use_early_stop": False,
            "early_stop_burnin_rounds": 5,
            "early_stop_rounds": 3,
            "early_stop_strategy": EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY,
        })

    evaluator = GSgnnRegressionEvaluator(config.eval_frequency,
                                         config.eval_metric,
                                         config.use_early_stop,
                                         config.early_stop_burnin_rounds,
                                         config.early_stop_rounds,
                                         config.early_stop_strategy)
    for _ in range(10):
        # always return false
        assert evaluator.do_early_stop({"rmse": 0.1}) is False

    config = Dummy({
            "eval_frequency": 100,
            "eval_metric": ["rmse"],
            "use_early_stop": True,
            "early_stop_burnin_rounds": 5,
            "early_stop_rounds": 3,
            "early_stop_strategy": EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY,
        })

    evaluator = GSgnnRegressionEvaluator(config.eval_frequency,
                                         config.eval_metric,
                                         config.use_early_stop,
                                         config.early_stop_burnin_rounds,
                                         config.early_stop_rounds,
                                         config.early_stop_strategy)
    for _ in range(5):
        # always return false
        assert evaluator.do_early_stop({"rmse": 0.5}) is False

    for _ in range(3):
        # no enough data point
        assert evaluator.do_early_stop({"rmse": 0.4}) is False

    assert evaluator.do_early_stop({"rmse": 0.3}) is False # better result
    assert evaluator.do_early_stop({"rmse": 0.32}) is False
    assert evaluator.do_early_stop({"rmse": 0.3}) is False
    assert evaluator.do_early_stop({"rmse": 0.32}) # early stop

    config2 = Dummy({
            "multilabel": False,
            "eval_frequency": 100,
            "eval_metric": ["accuracy"],
            "use_early_stop": True,
            "early_stop_burnin_rounds": 5,
            "early_stop_rounds": 3,
            "early_stop_strategy": EARLY_STOP_AVERAGE_INCREASE_STRATEGY,
        })

    evaluator = GSgnnAccEvaluator(config2.eval_frequency,
                                  config2.eval_metric,
                                  config2.multilabel,
                                  config2.use_early_stop,
                                  config2.early_stop_burnin_rounds,
                                  config2.early_stop_rounds,
                                  config2.early_stop_strategy)
    for _ in range(5):
        # always return false
        assert evaluator.do_early_stop({"accuracy": 0.5}) is False

    for _ in range(3):
        # no enough data point
        assert evaluator.do_early_stop({"accuracy": 0.6}) is False

    assert evaluator.do_early_stop({"accuracy": 0.7}) is False # better than average
    assert evaluator.do_early_stop({"accuracy": 0.68}) is False # still better
    assert evaluator.do_early_stop({"accuracy": 0.66}) # early stop

def test_early_stop_lp_evaluator():
    # common Dummy objects
    train_data = Dummy({
            "train_idxs": th.randint(10, (10,)),
            "val_idxs": th.randint(10, (10,)),
            "test_idxs": th.randint(10, (10,)),
            "do_validation": True
        })

    config = Dummy({
            "num_negative_edges_eval": 10,
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "eval_frequency": 100,
            "use_early_stop": False,
        })
    evaluator = GSgnnMrrLPEvaluator(config.eval_frequency,
                                    train_data,
                                    num_negative_edges_eval=config.num_negative_edges_eval,
                                    lp_decoder_type=config.lp_decoder_type,
                                    use_early_stop=config.use_early_stop)
    for _ in range(10):
        # always return false
        assert evaluator.do_early_stop({"mrr": 0.5}) is False

    config = Dummy({
            "num_negative_edges_eval": 10,
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "eval_frequency": 100,
            "use_early_stop": True,
            "early_stop_burnin_rounds": 5,
            "early_stop_rounds": 3,
            "early_stop_strategy": EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY,
        })
    evaluator = GSgnnMrrLPEvaluator(config.eval_frequency,
                                    train_data,
                                    num_negative_edges_eval=config.num_negative_edges_eval,
                                    lp_decoder_type=config.lp_decoder_type,
                                    use_early_stop=config.use_early_stop,
                                    early_stop_burnin_rounds=config.early_stop_burnin_rounds,
                                    early_stop_rounds=config.early_stop_rounds,
                                    early_stop_strategy=config.early_stop_strategy)
    for _ in range(5):
        # always return false
        assert evaluator.do_early_stop({"mrr": 0.5}) is False

    for _ in range(3):
        # no enough data point
        assert evaluator.do_early_stop({"mrr": 0.4}) is False

    assert evaluator.do_early_stop({"mrr": 0.5}) is False # better result
    assert evaluator.do_early_stop({"mrr": 0.45}) is False
    assert evaluator.do_early_stop({"mrr": 0.45}) is False
    assert evaluator.do_early_stop({"mrr": 0.45}) # early stop

    config = Dummy({
            "num_negative_edges_eval": 10,
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "eval_frequency": 100,
            "use_early_stop": True,
            "early_stop_burnin_rounds": 5,
            "early_stop_rounds": 3,
            "early_stop_strategy": EARLY_STOP_AVERAGE_INCREASE_STRATEGY,
        })
    evaluator = GSgnnMrrLPEvaluator(config.eval_frequency,
                                    train_data,
                                    num_negative_edges_eval=config.num_negative_edges_eval,
                                    lp_decoder_type=config.lp_decoder_type,
                                    use_early_stop=config.use_early_stop,
                                    early_stop_burnin_rounds=config.early_stop_burnin_rounds,
                                    early_stop_rounds=config.early_stop_rounds,
                                    early_stop_strategy=config.early_stop_strategy)
    for _ in range(5):
        # always return false
        assert evaluator.do_early_stop({"accuracy": 0.5}) is False

    for _ in range(3):
        # no enough data point
        assert evaluator.do_early_stop({"accuracy": 0.6}) is False

    assert evaluator.do_early_stop({"accuracy": 0.7}) is False # better than average
    assert evaluator.do_early_stop({"accuracy": 0.68}) is False # still better
    assert evaluator.do_early_stop({"accuracy": 0.66})

def test_get_val_score_rank():
    # ------------------- test InstanceEvaluator -------------------
    # common Dummy objects
    config = Dummy({
            "multilabel": False,
            "eval_frequency": 100,
            "eval_metric": ["accuracy"],
            "use_early_stop": False,
        })

    evaluator = GSgnnAccEvaluator(config.eval_frequency,
                                  config.eval_metric,
                                  config.multilabel,
                                  config.use_early_stop)
    # For accuracy, the bigger the better.
    val_score = {"accuracy": 0.47}
    assert evaluator.get_val_score_rank(val_score) == 1
    val_score = {"accuracy": 0.40}
    assert evaluator.get_val_score_rank(val_score) == 2
    val_score = {"accuracy": 0.7}
    assert evaluator.get_val_score_rank(val_score) == 1
    val_score = {"accuracy": 0.47}
    assert evaluator.get_val_score_rank(val_score) == 3

    config = Dummy({
            "multilabel": False,
            "eval_frequency": 100,
            "eval_metric": ["mse"],
            "use_early_stop": False,
        })

    evaluator = GSgnnRegressionEvaluator(config.eval_frequency,
                                         config.eval_metric,
                                         config.use_early_stop)
    # For mse, the smaller the better
    val_score = {"mse": 0.47}
    assert evaluator.get_val_score_rank(val_score) == 1
    val_score = {"mse": 0.40}
    assert evaluator.get_val_score_rank(val_score) == 1
    val_score = {"mse": 0.7}
    assert evaluator.get_val_score_rank(val_score) == 3
    val_score = {"mse": 0.47}
    assert evaluator.get_val_score_rank(val_score) == 3

    config = Dummy({
            "multilabel": False,
            "eval_frequency": 100,
            "eval_metric": ["rmse"],
            "use_early_stop": False,
        })

    evaluator = GSgnnRegressionEvaluator(config.eval_frequency,
                                         config.eval_metric,
                                         config.use_early_stop)
    # For rmse, the smaller the better
    val_score = {"rmse": 0.47}
    assert evaluator.get_val_score_rank(val_score) == 1
    val_score = {"rmse": 0.40}
    assert evaluator.get_val_score_rank(val_score) == 1
    val_score = {"rmse": 0.7}
    assert evaluator.get_val_score_rank(val_score) == 3
    val_score = {"rmse": 0.47}
    assert evaluator.get_val_score_rank(val_score) == 3

    # ------------------- test LPEvaluator -------------------
    # common Dummy objects
    train_data = Dummy({
            "train_idxs": th.randint(10, (10,)),
            "val_idxs": th.randint(10, (10,)),
            "test_idxs": th.randint(10, (10,)),
        })

    config = Dummy({
            "num_negative_edges_eval": 10,
            "lp_decoder_type": BUILTIN_LP_DOT_DECODER,
            "eval_frequency": 100,
            "use_early_stop": False,
            "eval_metric": ["mrr"]
        })

    evaluator = GSgnnMrrLPEvaluator(config.eval_frequency,
                                    train_data,
                                    num_negative_edges_eval=config.num_negative_edges_eval,
                                    lp_decoder_type=config.lp_decoder_type,
                                    use_early_stop=config.use_early_stop)

    # For MRR, the bigger the better
    val_score = {"mrr": 0.47}
    assert evaluator.get_val_score_rank(val_score) == 1

    val_score = {"mrr": 0.40}
    assert evaluator.get_val_score_rank(val_score) == 2

    val_score = {"mrr": 0.7}
    assert evaluator.get_val_score_rank(val_score) == 1

    val_score = {"mrr": 0.47}
    assert evaluator.get_val_score_rank(val_score) == 3


if __name__ == '__main__':
    # test evaluators
    test_mrr_per_etype_lp_evaluation()
    test_mrr_lp_evaluator()
    test_acc_evaluator()
    test_regression_evaluator()
    test_early_stop_avg_increase_judge()
    test_early_stop_cons_increase_judge()
    test_early_stop_evaluator()
    test_early_stop_lp_evaluator()
    test_get_val_score_rank()
