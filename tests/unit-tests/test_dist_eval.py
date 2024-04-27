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

    test_dist_eval test distributed evaluation methods.

    For link prediction, it compares the outputs of `fullgraph_eval`
    when called by a single worker (single process evaluation)
    and when called by two workers (distributed evaluation).

    For classification tasks. it compares the output of
    `GSgnnClassificationEvaluator.evaluate' when called by a single worker
    (single process evaluation) and when called by two workers
    (distributed evaluation).
"""

import multiprocessing as mp
import pytest

import torch as th
from numpy.testing import assert_almost_equal
import numpy as np

from graphstorm.eval import GSgnnClassificationEvaluator
from graphstorm.eval import GSgnnRegressionEvaluator
from graphstorm.eval import GSgnnMrrLPEvaluator
from graphstorm.utils import setup_device

from graphstorm.config import BUILTIN_LP_DOT_DECODER

from util import Dummy

from test_evaluator import gen_hg

def run_dist_lp_eval_worker(worker_rank, config, val_scores, test_scores, conn):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=2,
                                      rank=worker_rank)

    lp_eval = GSgnnMrrLPEvaluator(config.eval_frequency,
                                  use_early_stop=config.use_early_stop)
    val_sc, test_sc = lp_eval.evaluate(val_scores, test_scores, 0)

    if worker_rank == 0:
        conn.send((val_sc, test_sc))
    th.distributed.destroy_process_group()

def run_dist_lp_eval(config,
        val_scores_0, val_scores_1,
        test_scores_0, test_scores_1):
    ctx = mp.get_context('spawn')
    conn1, conn2 = mp.Pipe()
    p0 = ctx.Process(target=run_dist_lp_eval_worker,
                     args=(0, config, val_scores_0, test_scores_0, conn2))
    p1 = ctx.Process(target=run_dist_lp_eval_worker,
                     args=(1, config, val_scores_1, test_scores_1, None))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    assert p0.exitcode == 0
    assert p1.exitcode == 0

    val_scores, test_scores = conn1.recv()
    conn1.close()
    conn2.close()
    return val_scores, test_scores

def run_local_lp_eval(config, val_scores, test_scores):
    ctx = mp.get_context('spawn')
    conn1, conn2 = mp.Pipe()
    p = ctx.Process(target=run_local_lp_eval_worker,
                    args=(config, val_scores, test_scores, conn2))
    p.start()
    p.join()
    assert p.exitcode == 0

    val_scores, test_scores = conn1.recv()
    conn1.close()
    conn2.close()
    return val_scores, test_scores

def run_local_lp_eval_worker(config, val_scores, test_scores, conn):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=1,
                                      rank=0)

    lp_eval = GSgnnMrrLPEvaluator(config.eval_frequency,
                                  use_early_stop=config.use_early_stop)
    val_sc, test_sc = lp_eval.evaluate(val_scores, test_scores, 0)
    conn.send((val_sc, test_sc))
    th.distributed.destroy_process_group()

@pytest.mark.parametrize("seed", [41, 42])
def test_lp_dist_eval(seed):
    """ distributed evaluation is implemented in graphstorm.model.utils.fullgraph_eval
    """
    th.manual_seed(seed)
    # Use full nodes as negative
    # Create a random heterogenous graph first
    etypes = [("n0", "r0", "n1"), ("n0", "r1", "n1")]

    val_ranking_etype1 = th.rand((10,2))
    val_ranking_etype2 = th.rand((10,2))
    val_scores_0 = {
        ("n0", "r0", "n1") : val_ranking_etype1,
        ("n0", "r1", "n1") : val_ranking_etype2
    }
    val_ranking_etype1 = th.rand((10,2))
    val_ranking_etype2 = th.rand((10,2))
    val_scores_1 = {
        ("n0", "r0", "n1") : val_ranking_etype1,
        ("n0", "r1", "n1") : val_ranking_etype2
    }

    test_ranking_etype1 = th.rand((10,2))
    test_ranking_etype2 = th.rand((10,2))
    test_scores_0 = {
        ("n0", "r0", "n1") : test_ranking_etype1,
        ("n0", "r1", "n1") : test_ranking_etype2
    }
    test_ranking_etype1 = th.rand((10,2))
    test_ranking_etype2 = th.rand((10,2))
    test_scores_1 = {
        ("n0", "r0", "n1") : test_ranking_etype1,
        ("n0", "r1", "n1") : test_ranking_etype2
    }

    # Dummy objects
    config = Dummy({
            "eval_frequency": 100,
            "use_early_stop": False,
            "eval_metric_list": ["mrr"]
        })

    # do evaluation with two workers
    val_dist, test_dist = run_dist_lp_eval(config,
        val_scores_0, val_scores_1,
        test_scores_0, test_scores_1)
    # do evaluation with single worker
    val_local, test_local = run_local_lp_eval(config,
        {etypes[0]: th.cat((val_scores_0[etypes[0]], val_scores_1[etypes[0]]), dim = 0),
         etypes[1]: th.cat((val_scores_0[etypes[1]], val_scores_1[etypes[1]]), dim = 0)},
        {etypes[0]: th.cat((test_scores_0[etypes[0]], test_scores_1[etypes[0]]), dim = 0),
         etypes[1]: th.cat((test_scores_0[etypes[1]], test_scores_1[etypes[1]]), dim =0)})

    print(f"dist {val_dist}")
    print(f"local {val_local}")
    assert_almost_equal(np.array(val_dist["mrr"]),
        np.array(val_local["mrr"]), decimal=5)
    assert_almost_equal(np.array(test_dist["mrr"]),
        np.array(test_local["mrr"]), decimal=5)

def run_dist_nc_eval_worker(eval_config, worker_rank, metric, val_pred, test_pred,
    val_labels0, val_labels1, val_labels2, test_labels, backend, conn):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=2,
                                      rank=worker_rank)

    th.cuda.set_device(worker_rank)
    device = setup_device(worker_rank)
    config = eval_config

    if config.eval_metric_list[0] in ["rmse", "mse"]:
        evaluator = GSgnnRegressionEvaluator(config.eval_frequency,
                                             config.eval_metric_list,
                                             config.use_early_stop)
    else:
        evaluator = GSgnnClassificationEvaluator(config.eval_frequency,
                                                 config.eval_metric_list,
                                                 config.multilabel,
                                                 config.use_early_stop)

    val_score0, test_score0 = evaluator.evaluate(
        val_pred.to(device),
        test_pred.to(device),
        val_labels0.to(device),
        test_labels.to(device), 100)
    val_score1, test_score1 = evaluator.evaluate(
        val_pred.to(device),
        test_pred.to(device),
        val_labels1.to(device),
        test_labels.to(device), 200)
    val_score2, _ = evaluator.evaluate(
        val_pred.to(device),
        test_pred.to(device),
        val_labels2.to(device),
        test_labels.to(device), 300)

    if worker_rank == 0:
        assert evaluator.metric_list == metric
        assert evaluator.best_iter_num[metric[0]] == 200
        assert evaluator.best_val_score == val_score1
        assert evaluator.best_test_score == test_score1

        metrics = {
            "val0": val_score0[metric[0]],
            "val1": val_score1[metric[0]],
            "val2": val_score2[metric[0]],
            "test0": test_score0[metric[0]],
        }
        conn.send(metrics)
        th.distributed.destroy_process_group()

def run_dist_nc_eval(eval_config, metric, val_pred, test_pred,
    val_labels0, val_labels1, val_labels2, test_labels, backend):
    # split pos_eids into two

    shift = 5
    val_pred_0 = val_pred[0:len(val_pred)//2 + shift]
    val_pred_1 = val_pred[len(val_pred)//2 + shift:]
    test_pred_0 = test_pred[0:len(test_pred)//2 + shift]
    test_pred_1 = test_pred[len(test_pred)//2 + shift:]
    val_labels0_0 = val_labels0[0:len(val_labels0)//2 + shift]
    val_labels0_1 = val_labels0[len(val_labels0)//2 + shift:]
    val_labels1_0 = val_labels1[0:len(val_labels1)//2 + shift]
    val_labels1_1 = val_labels1[len(val_labels1)//2 + shift:]
    val_labels2_0 = val_labels2[0:len(val_labels2)//2 + shift]
    val_labels2_1 = val_labels2[len(val_labels2)//2 + shift:]
    test_labels_0 = test_labels[0:len(test_labels)//2 + shift]
    test_labels_1 = test_labels[len(test_labels)//2 + shift:]

    ctx = mp.get_context('spawn')
    conn1, conn2 = mp.Pipe()
    p0 = ctx.Process(target=run_dist_nc_eval_worker,
                     args=(eval_config, 0, metric,
                           val_pred_0, test_pred_0,
                           val_labels0_0, val_labels1_0,
                           val_labels2_0, test_labels_0, backend, conn2))
    p1 = ctx.Process(target=run_dist_nc_eval_worker,
                     args=(eval_config, 1, metric,
                           val_pred_1, test_pred_1,
                           val_labels0_1, val_labels1_1,
                           val_labels2_1, test_labels_1, backend, None))
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

def run_local_nc_eval_worker(eval_config, metric, val_pred, test_pred,
    val_labels0, val_labels1, val_labels2, test_labels, conn):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12346')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=1,
                                      rank=0)
    config = eval_config

    if config.eval_metric_list[0] in ["rmse", "mse"]:
        evaluator = GSgnnRegressionEvaluator(config.eval_frequency,
                                             config.eval_metric_list,
                                             config.use_early_stop)
    else:
        evaluator = GSgnnClassificationEvaluator(config.eval_frequency,
                                                 config.eval_metric_list,
                                                 config.multilabel,
                                                 config.use_early_stop)

    val_score0, test_score0 = evaluator.evaluate(val_pred, test_pred, val_labels0, test_labels, 100)
    val_score1, test_score1 = evaluator.evaluate(val_pred, test_pred, val_labels1, test_labels, 200)
    val_score2, _ = evaluator.evaluate(val_pred, test_pred, val_labels2, test_labels, 300)
    assert val_score0 != val_score1
    assert val_score0 != val_score2
    assert test_score0 == test_score1

    assert evaluator.metric_list == metric
    assert evaluator.best_iter_num[metric[0]] == 200
    assert evaluator.best_val_score == val_score1
    assert evaluator.best_test_score == test_score1

    metrics = {
        "val0": val_score0[metric[0]],
        "val1": val_score1[metric[0]],
        "val2": val_score2[metric[0]],
        "test0": test_score0[metric[0]],
    }
    conn.send(metrics)
    th.distributed.destroy_process_group()

def run_local_nc_eval(eval_config, metric, val_pred, test_pred,
    val_labels0, val_labels1, val_labels2, test_labels):

    ctx = mp.get_context('spawn')
    conn1, conn2 = mp.Pipe()
    p = ctx.Process(target=run_local_nc_eval_worker,
                    args=(eval_config, metric, val_pred, test_pred,
                          val_labels0, val_labels1,
                          val_labels2, test_labels, conn2))
    p.start()
    p.join()
    assert p.exitcode == 0

    dist_result = conn1.recv()
    conn1.close()
    conn2.close()
    return dist_result

@pytest.mark.parametrize("metric", [["accuracy"], ["f1_score"], ["f1_score", "accuracy"]])
@pytest.mark.parametrize("seed", [41, 42])
@pytest.mark.parametrize("backend", ["gloo"])
def test_nc_dist_eval(metric, seed, backend):
    th.manual_seed(seed)

    val_pred = th.randint(20, (200,))
    test_pred = th.randint(20, (200,))
    val_labels0 = th.randint(20, (200,))
    val_labels0[:180] = val_pred[:180]
    test_labels = th.randint(20, (200,))
    test_labels[:80] = test_pred[:80]
    test_labels[100:170] = test_pred[100:170]

    val_labels1 = th.randint(20, (200,))
    val_labels1[:190] = val_pred[:190]

    val_labels2 = th.randint(20, (200,))
    val_labels2[:160] = val_pred[:160]

    config = Dummy({
        "eval_metric_list": metric,
        "no_validation": False,
        "multilabel": False,
        "eval_frequency": 100,
        "use_early_stop": False,
    })

    # do evaluation with single worker
    metrics_local = run_local_nc_eval(config, metric, val_pred, test_pred,
        val_labels0, val_labels1, val_labels2, test_labels)
    # do evaluation with two workers
    metrics_dist = run_dist_nc_eval(config, metric, val_pred, test_pred,
        val_labels0, val_labels1, val_labels2, test_labels, backend)


    metrics_keys = list(metrics_local.keys())
    for key in metrics_keys:
        assert_almost_equal(
            np.array(metrics_dist[key]),
            np.array(metrics_local[key]),
            decimal=8)

@pytest.mark.parametrize("seed", [41])
@pytest.mark.parametrize("backend", ["gloo"])
def test_nc_dist_eval_multilabel(seed, backend):
    """ Unitest for multi-label node classification distributed evaluation
    """
    th.manual_seed(seed)

    # Generate faked validation prediction results
    # test prediction results
    # validation labels and test labels
    # We generate three validation labels so that we can
    # also test the best_val_score and best_test_score functionality
    # of Evaluators.
    val_pred = th.randint(20, (200, 2))
    test_pred = th.randint(20, (200, 2))
    val_labels = th.nn.functional.one_hot(val_pred, num_classes=20)
    val_labels = val_labels[:,0] + val_labels[:,1]
    val_labels[val_labels > 0] = 1
    test_labels = th.nn.functional.one_hot(test_pred, num_classes=20)
    test_labels = test_labels[:,0] + test_labels[:,1]
    test_labels[test_labels > 0] = 1
    val_labels0 = th.randint(20, (200,2))
    val_labels0 = th.nn.functional.one_hot(val_labels0, num_classes=20)
    val_labels0 = val_labels0[:,0] + val_labels0[:,1]
    val_labels0[val_labels0 > 0] = 1
    val_labels0[:180] = val_labels[:180]
    test_labels0 = th.randint(20, (200,2))
    test_labels0 = th.nn.functional.one_hot(test_labels0, num_classes=20)
    test_labels0 = test_labels0[:,0] + test_labels0[:,1]
    test_labels0[test_labels0 > 0] = 1
    test_labels0[:80] = test_labels[:80]
    test_labels0[100:170] = test_labels[100:170]

    val_labels1 = th.randint(20, (200,2))
    val_labels1 = th.nn.functional.one_hot(val_labels1, num_classes=20)
    val_labels1 = val_labels1[:,0] + val_labels1[:,1]
    val_labels1[val_labels1 > 0] = 1
    val_labels1[:190] = val_labels[:190]

    val_labels2 = th.randint(20, (200,2))
    val_labels2 = th.nn.functional.one_hot(val_labels2, num_classes=20)
    val_labels2 = val_labels2[:,0] + val_labels2[:,1]
    val_labels2[val_labels2 > 0] = 1
    val_labels2[:160] = val_labels[:160]

    val_logits = th.rand((200, 20)) / 5
    val_logits += val_labels
    test_logits = th.rand((200, 20)) / 5
    test_logits += test_labels

    softmax = th.nn.Softmax(dim=1)
    val_logits = softmax(val_logits)
    test_logits = softmax(test_logits)

    config = Dummy({
        "eval_metric_list": ["accuracy"],
        "no_validation": False,
        "multilabel": True,
        "eval_frequency": 100,
        "use_early_stop": False,
    })

    # do evaluation with single worker
    metrics_local = run_local_nc_eval(config, ["accuracy"], val_logits, test_logits,
        val_labels0, val_labels1, val_labels2, test_labels0)
    # do evaluation with two workers
    metrics_dist = run_dist_nc_eval(config, ["accuracy"], val_logits, test_logits,
        val_labels0, val_labels1, val_labels2, test_labels0, backend)

    metrics_keys = list(metrics_local.keys())
    for key in metrics_keys:
        assert_almost_equal(
            np.array(metrics_dist[key]),
            np.array(metrics_local[key]),
            decimal=8)

@pytest.mark.parametrize("metric", [["rmse"], ["rmse", "mse"]])
@pytest.mark.parametrize("seed", [41, 42])
@pytest.mark.parametrize("backend", ["gloo"])
def test_nc_dist_regression_eval(metric, seed, backend):
    th.manual_seed(seed)

    val_pred = th.rand((200,1)) * 100
    test_pred = th.rand((200,1)) * 100
    val_labels0 = th.rand((200,1)) * 100
    val_labels1 = val_labels0.clone()
    val_labels2 = val_labels0.clone()
    val_labels0[:180] = val_pred[:180]
    val_labels1[:190] = val_pred[:190]
    val_labels2[:160] = val_pred[:160]
    test_labels = th.rand((200,1)) * 100
    test_labels[:80] = test_pred[:80]
    test_labels[100:170] = test_pred[100:170]

    config = Dummy({
        "eval_metric_list": metric,
        "no_validation": False,
        "eval_frequency": 100,
        "use_early_stop": False,
    })

    metrics_local = run_local_nc_eval(config, metric, val_pred, test_pred,
        val_labels0, val_labels1, val_labels2, test_labels)
    metrics_dist = run_dist_nc_eval(config, metric, val_pred, test_pred,
        val_labels0, val_labels1, val_labels2, test_labels, backend)

    metrics_keys = list(metrics_local.keys())
    for key in metrics_keys:
        assert_almost_equal(
            np.array(metrics_dist[key]),
            np.array(metrics_local[key]),
            decimal=8)

if __name__ == '__main__':
    test_lp_dist_eval(seed=41)
    test_lp_dist_eval(seed=42)

    #test_nc_dist_eval(["accuracy"], seed=41, backend="gloo")
    #test_nc_dist_eval(["f1_score"], seed=41, backend="nccl")
    #test_nc_dist_eval_multilabel(seed=41, backend="gloo")
    #test_nc_dist_eval_multilabel(seed=41, backend="nccl")

    test_nc_dist_regression_eval(["rmse"], seed=41, backend="gloo")

    test_nc_dist_eval(["accuracy"], seed=41, backend="gloo")
    ##test_nc_dist_eval(["f1_score"], seed=41, backend="nccl")
    test_nc_dist_eval_multilabel(seed=41, backend="gloo")
    ##test_nc_dist_eval_multilabel(seed=41, backend="nccl")
