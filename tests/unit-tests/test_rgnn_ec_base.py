""" test_rgnn_nc_base test functionalities of M5GNNEdgeClassificationModel

    The tested functions includes:
        M5GNNEdgeClassificationModel.eval(): We use mock to verify whether eval()
        function act as expected under single process evaluation and
        distributed evaluation.
        It verifies that the validation and test scores are correct and
        the logging system works as expected.

    TODO: Add more tests
"""

import multiprocessing as mp
import pytest

import torch as th
from unittest.mock import patch, MagicMock

from graphstorm.model.rgnn_edge_base import M5GNNEdgeModel
from graphstorm.model.evaluator import M5gnnAccEvaluator
from graphstorm.model.rgnn_ec_base import M5GNNEdgeClassificationModel

from util import Dummy
from test_rgnn_nc_base import gen_rand_labels


def run_dist_ec_eval_worker(eval_config, worker_rank, val_pred, test_pred,
    train_data, backend, conn):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=2,
                                      rank=worker_rank)
    device = 'cuda:%d' % worker_rank
    th.cuda.set_device(worker_rank)
    accEvaluator = M5gnnAccEvaluator(None, eval_config, train_data)
    val_labels = train_data.labels[train_data.val_idxs["test"]].to(device)
    test_labels = train_data.labels[train_data.test_idxs["test"]].to(device)

    val_score, test_score = accEvaluator.evaluate(
            val_pred.to(device), test_pred.to(device),
            val_labels, test_labels,
            0)

    @patch('time.time', MagicMock(return_value=12345))
    @patch.object(M5GNNEdgeClassificationModel, 'inference', return_value=(val_pred.to(device), val_labels.to(device), test_pred.to(device), test_labels.to(device)))
    @patch.object(M5GNNEdgeClassificationModel, 'log_print_metrics', return_value=None)
    def call_eval(mock_log_print_metrics, mock_inference):
        ec = M5GNNEdgeClassificationModel.__new__(M5GNNEdgeClassificationModel)
        ec.register_evaluator(accEvaluator)
        ec.target_etype = ("t0", "test", "t1")
        ec.eval(worker_rank, train_data, None, 100)
        # Note: we can not use
        # mock_inference.assert_called_once_with(None, None, None, val_labels, test_labels, None)
        # as torch does not allow Tensor == Tensor
        mock_inference.assert_called_once()
        if worker_rank == 0:
            mock_log_print_metrics.assert_called_with(
                val_score=val_score,
                test_score=test_score,
                dur_eval=0,
                total_steps=100)
        else:
            mock_log_print_metrics.assert_not_called()

    call_eval()

    if worker_rank == 0:
        conn.send((val_score, test_score))
        th.distributed.destroy_process_group()

def run_local_ec_eval_worker(eval_config, val_pred, test_pred,
    train_data, conn):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=1,
                                      rank=0)

    accEvaluator = M5gnnAccEvaluator(None, eval_config, train_data)
    val_labels = train_data.labels[train_data.val_idxs["test"]]
    test_labels = train_data.labels[train_data.test_idxs["test"]]

    val_score, test_score = accEvaluator.evaluate(
            val_pred, test_pred,
            val_labels, test_labels,
            0)

    @patch('time.time', MagicMock(return_value=12345))
    @patch.object(M5GNNEdgeClassificationModel, 'inference', return_value=(val_pred, val_labels, test_pred, test_labels))
    @patch.object(M5GNNEdgeClassificationModel, 'log_print_metrics', return_value=None)
    def call_eval(mock_log_print_metrics, mock_inference):
        ec = M5GNNEdgeClassificationModel.__new__(M5GNNEdgeClassificationModel)
        ec.register_evaluator(accEvaluator)
        ec.target_etype = ("t0", "test", "t1")
        ec.eval(0, train_data, None, 100)
        # Note: we can not use
        # mock_inference.assert_called_once_with(None, None, None, val_labels, test_labels, None)
        # as torch does not allow Tensor == Tensor
        mock_inference.assert_called_once()
        mock_log_print_metrics.assert_called_with(
            val_score=val_score,
            test_score=test_score,
            dur_eval=0,
            total_steps=100)

    call_eval()
    conn.send((val_score, test_score))

    th.distributed.destroy_process_group()

def run_local_ec_eval(eval_config, val_pred, test_pred, train_data):
    conn1, conn2 = mp.Pipe()
    ctx = mp.get_context('spawn')
    p = ctx.Process(target=run_local_ec_eval_worker,
                    args=(eval_config, val_pred, test_pred, train_data, conn2))
    p.start()
    p.join()
    assert p.exitcode == 0
    dist_result = conn1.recv()
    conn1.close()
    conn2.close()
    return dist_result

def run_dist_ec_eval(eval_config, val_pred, test_pred, train_data, backend):

    val_idxs = train_data.val_idxs["test"]
    test_idxs = train_data.test_idxs["test"]

    shift = 10
    val_pred_0 = val_pred[0:len(val_pred)//2 + shift].share_memory_()
    val_pred_1 = val_pred[len(val_pred)//2 + shift:].share_memory_()
    test_pred_0 = test_pred[0:len(test_pred)//2 + shift].share_memory_()
    test_pred_1 = test_pred[len(test_pred)//2 + shift:].share_memory_()
    val_idxs_0 = val_idxs[0:len(val_idxs)//2 + shift].share_memory_()
    val_idxs_1 = val_idxs[len(val_idxs)//2 + shift:].share_memory_()
    test_idxs_0 = test_idxs[0:len(test_idxs)//2 + shift].share_memory_()
    test_idxs_1 = test_idxs[len(test_idxs)//2 + shift:].share_memory_()

    train_data_0 = Dummy({
        "val_test_nodes": None,
        "labels": train_data.labels,
        "val_idxs": {"test": val_idxs_0},
        "test_idxs": {"test": test_idxs_0},
        "val_src_dst_pairs": None,
        "test_src_dst_pairs": None,
        "do_validation": True
    })

    train_data_1 = Dummy({
        "val_test_nodes": None,
        "labels": train_data.labels,
        "val_idxs": {"test": val_idxs_1},
        "test_idxs": {"test": test_idxs_1},
        "val_src_dst_pairs": None,
        "test_src_dst_pairs": None,
        "do_validation": True
    })

    ctx = mp.get_context('spawn')
    conn1, conn2 = mp.Pipe()
    p0 = ctx.Process(target=run_dist_ec_eval_worker,
                     args=(eval_config, 0,
                           val_pred_0, test_pred_0,
                           train_data_0, backend, conn2))
    p1 = ctx.Process(target=run_dist_ec_eval_worker,
                     args=(eval_config, 1,
                           val_pred_1, test_pred_1,
                           train_data_1, backend, None))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    assert p0.exitcode == 0
    assert p1.exitcode == 0

    dist_result = conn1.recv()
    conn1.close()
    conn2.close()
    return dist_result

@pytest.mark.parametrize("metric", [["accuracy"], ["f1_score", "accuracy"]])
@pytest.mark.parametrize("seed", [41, 42])
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_ec_dist_eval(metric, seed, backend):
    th.manual_seed(seed)

    labels, val_pred, test_pred, val_idxs, test_idxs = gen_rand_labels()
    val_pred = val_pred.share_memory_()
    test_pred = test_pred.share_memory_()

    config = Dummy({
        "eval_metric": metric,
        "no_validation": False,
        "multilabel": False,
        "evaluation_frequency": 100,
        "enable_early_stop": False,
    })

    train_data = Dummy({
        "val_test_nodes": None,
        "labels": labels.share_memory_(),
        "val_idxs": {"test": val_idxs.share_memory_()},
        "test_idxs": {"test": test_idxs.share_memory_()},
        "val_src_dst_pairs": None,
        "test_src_dst_pairs": None,
        "do_validation": True
    })

    local_result = run_local_ec_eval(config, val_pred, test_pred, train_data)
    dist_result = run_dist_ec_eval(config, val_pred, test_pred, train_data, backend)

    assert local_result[0] == dist_result[0]
    assert local_result[1] == dist_result[1]

def test_ec_init_m5gnn_model():
    @patch.object(M5GNNEdgeModel, 'init_m5gnn_model', return_value=None)
    def call_init_m5gnn_model(mock_linit_m5gnn_model):
        ec = M5GNNEdgeClassificationModel.__new__(M5GNNEdgeClassificationModel)
        setattr(ec, "multilabel", False)
        setattr(ec, "multilabel_weights", None)
        setattr(ec, "imbalance_class_weights", None)
        setattr(ec, "_dev_id", 0)
        ec.init_m5gnn_model()

        assert isinstance(ec.loss_func, th.nn.CrossEntropyLoss)
        mock_linit_m5gnn_model.assert_called_once()
        mock_linit_m5gnn_model.reset_mock()

        ec = M5GNNEdgeClassificationModel.__new__(M5GNNEdgeClassificationModel)
        setattr(ec, "multilabel", True)
        setattr(ec, "multilabel_weights", None)
        setattr(ec, "imbalance_class_weights", None)
        setattr(ec, "_dev_id", 0)
        ec.init_m5gnn_model()

        assert not isinstance(ec.loss_func, th.nn.CrossEntropyLoss)
        mock_linit_m5gnn_model.assert_called_once()
        mock_linit_m5gnn_model.reset_mock()

        ec = M5GNNEdgeClassificationModel.__new__(M5GNNEdgeClassificationModel)
        setattr(ec, "multilabel", True)
        setattr(ec, "multilabel_weights", th.tensor([1,2,3]))
        setattr(ec, "imbalance_class_weights", None)
        setattr(ec, "_dev_id", 0)
        ec.init_m5gnn_model()

        assert not isinstance(ec.loss_func, th.nn.CrossEntropyLoss)
        assert callable(ec.loss_func)
        mock_linit_m5gnn_model.assert_called_once()
        mock_linit_m5gnn_model.reset_mock()

        ec = M5GNNEdgeClassificationModel.__new__(M5GNNEdgeClassificationModel)
        setattr(ec, "multilabel", False)
        setattr(ec, "multilabel_weights", None)
        setattr(ec, "imbalance_class_weights", th.tensor([1,2,3]))
        setattr(ec, "_dev_id", 0)
        ec.init_m5gnn_model()

        assert isinstance(ec.loss_func, th.nn.CrossEntropyLoss)
        assert ec.loss_func.weight.tolist() == [1,2,3]
        mock_linit_m5gnn_model.assert_called_once()
        mock_linit_m5gnn_model.reset_mock()

    call_init_m5gnn_model()

if __name__ == '__main__':
    test_ec_dist_eval(["accuracy"], seed=41, backend="gloo")
    test_ec_dist_eval(["accuracy"], seed=41, backend="nccl")
    test_ec_init_m5gnn_model()