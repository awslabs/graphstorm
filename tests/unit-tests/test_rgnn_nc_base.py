""" test_rgnn_nc_base test functionalities of GSgnnNodeClassModel

    The tested functions includes:
        GSgnnNodeClassModel.eval(): We use mock to verify whether eval()
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

from graphstorm.eval import GSgnnAccEvaluator
from graphstorm.model.rgnn_nc_base import GSgnnNodeClassModel
from graphstorm.model.rgnn_node_base import GSgnnNodeModel

from util import Dummy

def gen_rand_labels():
    val_pred = th.randint(20, (200,))
    test_pred = th.randint(20, (200,))

    val_labels = th.randint(20, (200,))
    val_labels[:180] = val_pred[:180]
    test_labels = th.randint(20, (200,))
    test_labels[:80] = test_pred[:80]
    test_labels[100:170] = test_pred[100:170]

    labels = th.cat([val_labels, test_labels], dim=0)
    val_idxs = th.arange(200)
    test_idxs = th.arange(start=200, end=400)

    return labels, val_pred, test_pred, val_idxs, test_idxs

def run_dist_nc_eval_worker(eval_config, worker_rank, val_pred, test_pred,
    train_data, backend, conn):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=2,
                                      rank=worker_rank)
    device = 'cuda:%d' % worker_rank
    th.cuda.set_device(worker_rank)
    accEvaluator = GSgnnAccEvaluator(None, eval_config, train_data)
    val_labels = train_data.labels[train_data.val_idx].to(device)
    test_labels = train_data.labels[train_data.test_idx].to(device)

    val_score, test_score = accEvaluator.evaluate(
            val_pred.to(device), test_pred.to(device),
            val_labels, test_labels,
            0)

    @patch('time.time', MagicMock(return_value=12345))
    @patch.object(GSgnnNodeClassModel, 'inference', return_value=th.cat([val_pred, test_pred]).to(device))
    @patch.object(GSgnnNodeClassModel, 'log_print_metrics', return_value=None)
    def call_eval(mock_log_print_metrics, mock_inference):
        nc = GSgnnNodeClassModel.__new__(GSgnnNodeClassModel)
        nc.register_evaluator(accEvaluator)
        nc.eval(worker_rank, train_data, 100)
        # Note: we can not use
        # mock_inference.assert_called_once_with()
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

def run_local_nc_eval_worker(eval_config, val_pred, test_pred,
    train_data, conn):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=1,
                                      rank=0)

    accEvaluator = GSgnnAccEvaluator(None, eval_config, train_data)
    val_labels = train_data.labels[train_data.val_idx]
    test_labels = train_data.labels[train_data.test_idx]

    val_score, test_score = accEvaluator.evaluate(
            val_pred, test_pred,
            val_labels, test_labels,
            0)

    @patch('time.time', MagicMock(return_value=12345))
    @patch.object(GSgnnNodeClassModel, 'inference', return_value=th.cat([val_pred, test_pred]))
    @patch.object(GSgnnNodeClassModel, 'log_print_metrics', return_value=None)
    def call_eval(mock_log_print_metrics, mock_inference):
        nc = GSgnnNodeClassModel.__new__(GSgnnNodeClassModel)
        nc.register_evaluator(accEvaluator)
        nc.eval(0, train_data, 100)
        # Note: we can not use
        # mock_inference.assert_called_once_with()
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

def run_local_nc_eval(eval_config, val_pred, test_pred, train_data):
    conn1, conn2 = mp.Pipe()
    ctx = mp.get_context('spawn')
    p = ctx.Process(target=run_local_nc_eval_worker,
                    args=(eval_config, val_pred, test_pred, train_data, conn2))
    p.start()
    p.join()
    assert p.exitcode == 0
    dist_result = conn1.recv()
    conn1.close()
    conn2.close()
    return dist_result

def run_dist_nc_eval(eval_config, val_pred, test_pred, train_data, backend):
    val_idx = train_data.val_idx
    test_idx = train_data.test_idx

    shift = 10
    val_pred_0 = val_pred[0:len(val_pred)//2 + shift].share_memory_()
    val_pred_1 = val_pred[len(val_pred)//2 + shift:].share_memory_()
    test_pred_0 = test_pred[0:len(test_pred)//2 + shift].share_memory_()
    test_pred_1 = test_pred[len(test_pred)//2 + shift:].share_memory_()
    val_idxs_0 = val_idx[0:len(val_idx)//2 + shift].share_memory_()
    val_idxs_1 = val_idx[len(val_idx)//2 + shift:].share_memory_()
    test_idxs_0 = test_idx[0:len(test_idx)//2 + shift].share_memory_()
    test_idxs_1 = test_idx[len(test_idx)//2 + shift:].share_memory_()

    train_data_0 = Dummy({
        "labels": train_data.labels,
        "val_idx": val_idxs_0,
        "test_idx": test_idxs_0,
        "do_validation": True
    })

    train_data_1 = Dummy({
        "labels": train_data.labels,
        "val_idx": val_idxs_1,
        "test_idx":  test_idxs_1,
        "do_validation": True
    })

    ctx = mp.get_context('spawn')
    conn1, conn2 = mp.Pipe()
    p0 = ctx.Process(target=run_dist_nc_eval_worker,
                     args=(eval_config, 0,
                           val_pred_0, test_pred_0,
                           train_data_0, backend, conn2))
    p1 = ctx.Process(target=run_dist_nc_eval_worker,
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
def test_nc_dist_eval(metric, seed, backend):
    th.manual_seed(seed)

    labels, val_pred, test_pred, val_idx, test_idx = gen_rand_labels()
    val_pred = val_pred.share_memory_()
    test_pred = test_pred.share_memory_()
    val_idx = val_idx.share_memory_()
    test_idx = test_idx.share_memory_()

    config = Dummy({
        "eval_metric": metric,
        "no_validation": False,
        "multilabel": False,
        "evaluation_frequency": 100,
        "enable_early_stop": False,
    })

    train_data = Dummy({
        "labels": labels.share_memory_(),
        "val_idx": val_idx,
        "test_idx": test_idx,
        "do_validation": True
    })

    local_result = run_local_nc_eval(config, val_pred, test_pred, train_data)
    dist_result = run_dist_nc_eval(config, val_pred, test_pred, train_data, backend)

    assert local_result[0] == dist_result[0]
    assert local_result[1] == dist_result[1]

def test_nc_init_gsgnn_model():
    @patch.object(GSgnnNodeModel, 'init_gsgnn_model', return_value=None)
    def call_init_gsgnn_model(mock_linit_gsgnn_model):
        nc = GSgnnNodeClassModel.__new__(GSgnnNodeClassModel)
        setattr(nc, "multilabel", False)
        setattr(nc, "multilabel_weights", None)
        setattr(nc, "imbalance_class_weights", None)
        setattr(nc, "_dev_id", 0)
        nc.init_gsgnn_model()

        assert isinstance(nc.loss_func, th.nn.CrossEntropyLoss)
        mock_linit_gsgnn_model.assert_called_once()
        mock_linit_gsgnn_model.reset_mock()

        nc = GSgnnNodeClassModel.__new__(GSgnnNodeClassModel)
        setattr(nc, "multilabel", True)
        setattr(nc, "multilabel_weights", None)
        setattr(nc, "imbalance_class_weights", None)
        setattr(nc, "_dev_id", 0)
        nc.init_gsgnn_model()

        assert not isinstance(nc.loss_func, th.nn.CrossEntropyLoss)
        mock_linit_gsgnn_model.assert_called_once()
        mock_linit_gsgnn_model.reset_mock()

        nc = GSgnnNodeClassModel.__new__(GSgnnNodeClassModel)
        setattr(nc, "multilabel", True)
        setattr(nc, "multilabel_weights", th.tensor([1,2,3]))
        setattr(nc, "imbalance_class_weights", None)
        setattr(nc, "_dev_id", 0)
        nc.init_gsgnn_model()

        assert not isinstance(nc.loss_func, th.nn.CrossEntropyLoss)
        assert callable(nc.loss_func)
        mock_linit_gsgnn_model.assert_called_once()
        mock_linit_gsgnn_model.reset_mock()

        nc = GSgnnNodeClassModel.__new__(GSgnnNodeClassModel)
        setattr(nc, "multilabel", False)
        setattr(nc, "multilabel_weights", None)
        setattr(nc, "imbalance_class_weights", th.tensor([1,2,3]))
        setattr(nc, "_dev_id", 0)
        nc.init_gsgnn_model()

        assert isinstance(nc.loss_func, th.nn.CrossEntropyLoss)
        assert nc.loss_func.weight.tolist() == [1,2,3]
        mock_linit_gsgnn_model.assert_called_once()
        mock_linit_gsgnn_model.reset_mock()

    call_init_gsgnn_model()

if __name__ == '__main__':
    test_nc_dist_eval(["accuracy"], seed=41, backend="gloo")
    test_nc_dist_eval(["accuracy"], seed=41, backend="nccl")
    test_nc_init_gsgnn_model()
