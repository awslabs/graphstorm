
import importlib
import unittest
from unittest.mock import patch, MagicMock
import operator

import torch as th
import dgl

from graphstorm.model.evaluator import M5gnnMrrLPEvaluator
from graphstorm.model.evaluator import M5gnnAccEvaluator
from graphstorm.model.evaluator import M5gnnRegressionEvaluator
from graphstorm.model.evaluator import early_stop_avg_increase_judge
from graphstorm.model.evaluator import early_stop_cons_increase_judge
from graphstorm.config.config import EARLY_STOP_AVERAGE_INCREASE_STRATEGY
from graphstorm.config.config import EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY

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

def test_mrr_lp_evaluator():
    # system heavily depends on th distributed
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=1,
                                      rank=0)
    # common Dummy objects
    train_data = Dummy({
            "train_idxs": th.randint(10, (10,)),
            "val_idxs": th.randint(10, (10,)),
            "test_idxs": th.randint(10, (10,)),
            "do_validation": True
        })

    config = Dummy({
            "num_negative_edges_eval": 10,
            "use_dot_product": True,
            "evaluation_frequency": 100,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": False,
            "enable_early_stop": False,
        })
    hg = gen_hg()

    # test evaluate_on_idx
    @patch('builtins.print')
    @patch.object(M5gnnMrrLPEvaluator, '_fullgraph_eval')
    def check_evaluate_on_idx(mock_fullgraph_eval, mock_print):
        lp = M5gnnMrrLPEvaluator.__new__(M5gnnMrrLPEvaluator)

        # test lp.evaluate_on_idx
        lp.g = hg
        lp.num_negative_edges_eval = 10
        lp.use_dot_product = True
        lp.tracker = None
        lp.mlflow_report_frequency = 0

        val_idxs = {
            ("src", "rel_0", "dst"): th.tensor([1,2,3]),
            ("src", "rel_1", "dst"): th.tensor([4,5,6]),
            ("src", "rel_2", "dst"): th.tensor([2,4,6]),
        }

        mock_fullgraph_eval.side_effect = [
            {"MRR": 0.7, "HIT@1": 0.5},
            {"MRR": 0.6, "HIT@1": 0.45},
            {"MRR": 0.8, "HIT@1": 0.9}]

        lp.g.rank = MagicMock(return_value=0)
        metric = lp.evaluate_on_idx(None, None, None, val_idxs, "Valid")
        mock_fullgraph_eval.assert_called()
        mock_print.assert_called()
        assert "mrr" in metric
        assert metric["mrr"] == sum([0.7,0.6,0.8])/3

        mock_fullgraph_eval.reset_mock()
        mock_fullgraph_eval.side_effect = [
            {"MRR": 0.7, "HIT@1": 0.5},
            {"MRR": 0.6, "HIT@1": 0.45},
            {"MRR": 0.65, "HIT@1": 0.9}]
        mock_print.reset_mock()
        lp.g.rank = MagicMock(return_value=1)
        metric = lp.evaluate_on_idx(None, None, None, val_idxs, "Valid")
        mock_fullgraph_eval.assert_called()
        mock_print.assert_not_called()
        assert "mrr" in metric
        assert metric["mrr"] == sum([0.7,0.6,0.65])/3

        # check empty input
        val_idxs = {}
        metric = lp.evaluate_on_idx(None, None, None, val_idxs, "Valid")
        assert "mrr" in metric
        assert metric["mrr"] == -1

        val_idxs = None
        metric = lp.evaluate_on_idx(None, None, None, val_idxs, "Valid")
        assert "mrr" in metric
        assert metric["mrr"] == -1

    check_evaluate_on_idx()

    # test evaluate
    @patch.object(M5gnnMrrLPEvaluator, 'evaluate_on_idx')
    def check_evaluate(mock_evaluate_on_idx):
        lp = M5gnnMrrLPEvaluator(hg, config, train_data)
        lp.g.rank = MagicMock(return_value=0)

        mock_evaluate_on_idx.side_effect = [
            {"mrr": 0.6},
            {"mrr": 0.7},
            {"mrr": 0.65},
            {"mrr": 0.8},
            {"mrr": 0.8},
            {"mrr": 0.7}
        ]

        val_score, test_score = lp.evaluate(None, None, 100, None)
        assert val_score["mrr"] == 0.7
        assert test_score["mrr"] == 0.6
        val_score, test_score = lp.evaluate(None, None, 200, None)
        assert val_score["mrr"] == 0.8
        assert test_score["mrr"] == 0.65
        val_score, test_score = lp.evaluate(None, None, 300, None)
        assert val_score["mrr"] == 0.7
        assert test_score["mrr"] == 0.8

        assert lp.best_val_score["mrr"] == 0.8
        assert lp.best_test_score["mrr"] == 0.65
        assert lp.best_iter_num["mrr"] == 200

    # check M5gnnMrrLPEvaluator.evaluate()
    check_evaluate()

    # common Dummy objects
    train_data = Dummy({
            "train_idxs": None,
            "val_idxs": None,
            "test_idxs": th.randint(10, (10,)),
            "do_validation": True
        })
    # test evaluate
    @patch.object(M5gnnMrrLPEvaluator, 'evaluate_on_idx')
    def check_evaluate_infer(mock_evaluate_on_idx):
        lp = M5gnnMrrLPEvaluator(hg, config, train_data)
        lp.g.rank = MagicMock(return_value=0)

        mock_evaluate_on_idx.side_effect = [
            {"mrr": 0.6},
            {"mrr": 0.7},
        ]

        val_score, test_score = lp.evaluate(None, None, 100, None)
        assert val_score["mrr"] == -1
        assert test_score["mrr"] == 0.6
        val_score, test_score = lp.evaluate(None, None, 200, None)
        assert val_score["mrr"] == -1
        assert test_score["mrr"] == 0.7

        assert lp.best_val_score["mrr"] == 0 # Still initial value 0
        assert lp.best_test_score["mrr"] == 0 # Still initial value 0
        assert lp.best_iter_num["mrr"] == 0 # Still initial value 0

    # check M5gnnMrrLPEvaluator.evaluate()
    check_evaluate_infer()

    # check M5gnnMrrLPEvaluator.do_eval()
    # train_data.do_validation True
    # config.no_validation False
    lp = M5gnnMrrLPEvaluator(hg, config, train_data)
    assert lp.do_eval(120, epoch_end=True) is True
    assert lp.do_eval(200) is True
    assert lp.do_eval(0) is True
    assert lp.do_eval(1) is False

    # common Dummy objects
    train_data2 = Dummy({
            "train_idxs": th.randint(10, (10,)),
            "val_idxs": th.randint(10, (10,)),
            "test_idxs": th.randint(10, (10,)),
            "do_validation": False
        })
    # train_data.do_validation False
    # config.no_validation False
    lp = M5gnnMrrLPEvaluator(hg, config, train_data2)
    assert lp.do_eval(120, epoch_end=True) is False
    assert lp.do_eval(200) is False

    config2 = Dummy({
            "num_negative_edges_eval": 10,
            "use_dot_product": True,
            "evaluation_frequency": 100,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": True,
            "enable_early_stop": False,
        })

    # train_data.do_validation True
    # config.no_validation True
    lp = M5gnnMrrLPEvaluator(hg, config2, train_data)
    assert lp.do_eval(120, epoch_end=True) is False
    assert lp.do_eval(200) is False

    config3 = Dummy({
            "num_negative_edges_eval": 10,
            "use_dot_product": True,
            "evaluation_frequency": 0,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": False,
            "enable_early_stop": False,
        })

    # train_data.do_validation True
    # config.no_validation False
    # evaluation_frequency is 0
    lp = M5gnnMrrLPEvaluator(hg, config3, train_data)
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
    # common Dummy objects
    train_data = Dummy({
            "do_validation": True
        })

    config = Dummy({
            "multilabel": False,
            "evaluation_frequency": 100,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": False,
            "eval_metric": ["accuracy"],
            "enable_early_stop": False,
        })
    hg = gen_hg()

    # Test evaluate
    @patch.object(M5gnnAccEvaluator, 'compute_score')
    def check_evaluate(mock_compute_score):
        nc = M5gnnAccEvaluator(hg, config, train_data)
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

    # check M5gnnAccEvaluator.do_eval()
    # train_data.do_validation True
    # config.no_validation False
    nc = M5gnnAccEvaluator(hg, config, train_data)
    assert nc.do_eval(120, epoch_end=True) is True
    assert nc.do_eval(200) is True
    assert nc.do_eval(0) is True
    assert nc.do_eval(1) is False

    # common Dummy objects
    train_data2 = Dummy({
            "do_validation": False
        })
    # train_data.do_validation False
    # config.no_validation False
    nc = M5gnnAccEvaluator(hg, config, train_data2)
    assert nc.do_eval(120, epoch_end=True) is False
    assert nc.do_eval(200) is False

    config2 = Dummy({
            "multilabel": False,
            "evaluation_frequency": 100,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": True,
            "eval_metric": ["accuracy"],
            "enable_early_stop": False,
        })

    # train_data.do_validation True
    # config.no_validation True
    nc = M5gnnAccEvaluator(hg, config2, train_data)
    assert nc.do_eval(120, epoch_end=True) is False
    assert nc.do_eval(200) is False

    config3 = Dummy({
            "multilabel": False,
            "evaluation_frequency": 0,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": False,
            "eval_metric": ["accuracy"],
            "enable_early_stop": False,
        })

    # train_data.do_validation True
    # config.no_validation False
    # evaluation_frequency is 0
    nc = M5gnnAccEvaluator(hg, config3, train_data)
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
    # common Dummy objects
    train_data = Dummy({
            "do_validation": True
        })

    config = Dummy({
            "evaluation_frequency": 100,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": False,
            "eval_metric": ["rmse"],
            "enable_early_stop": False,
        })
    hg = gen_hg()

    # Test evaluate
    @patch.object(M5gnnRegressionEvaluator, 'compute_score')
    def check_evaluate(mock_compute_score):
        nr = M5gnnRegressionEvaluator(hg, config, train_data)
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

    # check M5gnnRegressionEvaluator.do_eval()
    # train_data.do_validation True
    # config.no_validation False
    nr = M5gnnRegressionEvaluator(hg, config, train_data)
    assert nr.do_eval(120, epoch_end=True) is True
    assert nr.do_eval(200) is True
    assert nr.do_eval(0) is True
    assert nr.do_eval(1) is False

    # common Dummy objects
    train_data2 = Dummy({
            "do_validation": False
        })
    # train_data.do_validation False
    # config.no_validation False
    nr = M5gnnRegressionEvaluator(hg, config, train_data2)
    assert nr.do_eval(120, epoch_end=True) is False
    assert nr.do_eval(200) is False

    config2 = Dummy({
            "evaluation_frequency": 100,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": True,
            "eval_metric": ["rmse"],
            "enable_early_stop": False,
        })
    # train_data.do_validation True
    # config.no_validation True
    nr = M5gnnRegressionEvaluator(hg, config2, train_data)
    assert nr.do_eval(120, epoch_end=True) is False
    assert nr.do_eval(200) is False

    config3 = Dummy({
            "evaluation_frequency": 0,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": False,
            "eval_metric": ["rmse"],
            "enable_early_stop": False,
        })

    # train_data.do_validation True
    # config.no_validation False
    # evaluation_frequency is 0
    nr = M5gnnRegressionEvaluator(hg, config3, train_data)
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
    train_data = Dummy({
            "do_validation": True
        })

    config = Dummy({
            "evaluation_frequency": 100,
            "no_validation": False,
            "eval_metric": ["rmse"],
            "enable_early_stop": False,
            "call_to_consider_early_stop": 5,
            "window_for_early_stop": 3,
            "early_stop_strategy": EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY,
        })

    evaluator = M5gnnRegressionEvaluator(None, config, train_data)
    for _ in range(10):
        # always return false
        assert evaluator.do_early_stop({"rmse": 0.1}) is False

    config = Dummy({
            "evaluation_frequency": 100,
            "no_validation": False,
            "eval_metric": ["rmse"],
            "enable_early_stop": True,
            "call_to_consider_early_stop": 5,
            "window_for_early_stop": 3,
            "early_stop_strategy": EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY,
        })

    evaluator = M5gnnRegressionEvaluator(None, config, train_data)
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
            "evaluation_frequency": 100,
            "no_validation": False,
            "eval_metric": ["accuracy"],
            "enable_early_stop": True,
            "call_to_consider_early_stop": 5,
            "window_for_early_stop": 3,
            "early_stop_strategy": EARLY_STOP_AVERAGE_INCREASE_STRATEGY,
        })

    evaluator = M5gnnAccEvaluator(None, config2, train_data)
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
            "use_dot_product": True,
            "evaluation_frequency": 100,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": False,
            "enable_early_stop": False,
        })
    hg = gen_hg()
    evaluator = M5gnnMrrLPEvaluator(hg, config, train_data)
    for _ in range(10):
        # always return false
        assert evaluator.do_early_stop({"mrr": 0.5}) is False

    config = Dummy({
            "num_negative_edges_eval": 10,
            "use_dot_product": True,
            "evaluation_frequency": 100,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": False,
            "enable_early_stop": True,
            "call_to_consider_early_stop": 5,
            "window_for_early_stop": 3,
            "early_stop_strategy": EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY,
        })
    evaluator = M5gnnMrrLPEvaluator(hg, config, train_data)
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
            "use_dot_product": True,
            "evaluation_frequency": 100,
            "mlflow_tracker": None,
            "mlflow_report_frequency": None,
            "no_validation": False,
            "enable_early_stop": True,
            "call_to_consider_early_stop": 5,
            "window_for_early_stop": 3,
            "early_stop_strategy": EARLY_STOP_AVERAGE_INCREASE_STRATEGY,
        })
    evaluator = M5gnnMrrLPEvaluator(hg, config, train_data)
    for _ in range(5):
        # always return false
        assert evaluator.do_early_stop({"accuracy": 0.5}) is False

    for _ in range(3):
        # no enough data point
        assert evaluator.do_early_stop({"accuracy": 0.6}) is False

    assert evaluator.do_early_stop({"accuracy": 0.7}) is False # better than average
    assert evaluator.do_early_stop({"accuracy": 0.68}) is False # still better
    assert evaluator.do_early_stop({"accuracy": 0.66})

if __name__ == '__main__':
    # test evaluators
    test_mrr_lp_evaluator()
    test_acc_evaluator()
    test_regression_evaluator()
    test_early_stop_avg_increase_judge()
    test_early_stop_cons_increase_judge()
    test_early_stop_evaluator()
    test_early_stop_lp_evaluator()
