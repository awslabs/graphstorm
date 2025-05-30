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
import inspect


import pytest
import torch as th
from numpy.testing import assert_almost_equal

from graphstorm.eval.eval_func import (eval_roc_auc,
                                       eval_acc)
from graphstorm.eval.eval_func import (
    compute_amri,
    compute_mse,
    compute_rmse,
    compute_roc_auc,
    compute_f1_score,
    compute_precision_recall_auc,
    compute_per_class_roc_auc,
    compute_hit_at_classification,
    compute_hit_at_link_prediction,
    compute_precision_recall_fscore,
    compute_precision,
    compute_recall,
    compute_fscore,
    compute_precision_at_recall,
    compute_recall_at_precision)
from graphstorm.eval.eval_func import ClassificationMetrics, LinkPredictionMetrics

def test_compute_mse():
    pred64 = th.rand((100,1), dtype=th.float64)
    label64 = pred64 + th.rand((100,1), dtype=th.float64) / 10

    pred32 = pred64.type(th.float32)
    label32 = label64.type(th.float32)
    mse32 = compute_mse(pred32, label32)

    mse_pred64 = compute_mse(pred64, label32)
    mse_label64 = compute_mse(pred32, label64)

    assert_almost_equal(mse32, mse_pred64)
    assert_almost_equal(mse32, mse_label64)

    # Test the case when label is a 1D tensor
    # and pred is a 2D tensor
    label32 = label32.squeeze()
    mse32_2 = compute_mse(pred32, label32)
    assert_almost_equal(mse32, mse32_2)

def test_compute_rmse():
    pred64 = th.rand((100,1), dtype=th.float64)
    label64 = pred64 + th.rand((100,1), dtype=th.float64) / 10

    pred32 = pred64.type(th.float32)
    label32 = label64.type(th.float32)
    rmse32 = compute_rmse(pred32, label32)

    rmse_pred64 = compute_rmse(pred64, label32)
    rmse_label64 = compute_rmse(pred32, label64)

    assert_almost_equal(rmse32, rmse_pred64)
    assert_almost_equal(rmse32, rmse_label64)

    # Test the case when label is a 1D tensor
    # and pred is a 2D tensor
    label32 = label32.squeeze()
    rmse32_2 = compute_rmse(pred32, label32)
    assert_almost_equal(rmse32, rmse32_2)

def test_eval_roc_auc():
    # GraphStorm inputs: preds are logits>= 2D, and labels are all 1D list.

    # Invalid case 1: different No. of rows, return -1 and throws an exception
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75])
    labels = th.concat([th.zeros(25), th.ones(25)]).long()
    try:
        error_score_1 = eval_roc_auc(preds, labels)
    except (AssertionError, ValueError) as e1:
        print(f'Test eval_roc_auc error1, {e1}')
        error_score_1 = -1

    # Invalid case 2: preds is 1D, return -1 and throws an exception
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75])
    labels = th.concat([th.zeros(50), th.ones(50)]).long()
    try:
        error_score_2 = eval_roc_auc(preds, labels)
    except (AssertionError, ValueError) as e2:
        print(f'Test eval_roc_auc error2, {e2}')
        error_score_2 = -1

    # Invalid case 3: preds in 3D, labels in 1D, but with 4 classes
    preds = th.concat([th.tensor([0.75, 0.15, 0.1]).repeat(25),
                       th.tensor([0.1, 0.75, 0.15]).repeat(25),
                       th.tensor([0.1, 0.1, 0.75]).repeat(25),
                       th.tensor([0.15, 0.1, 0.1]).repeat(25)], dim=0).reshape(100, 3)
    labels = th.concat([th.zeros(25),
                        th.ones(25),
                        th.ones(25) + 1,
                        th.ones(25) + 2]).long()
    try:
        error_score_3 = eval_roc_auc(preds, labels)
    except (AssertionError, ValueError, IndexError) as e3:
        print(f'Test eval_roc_auc error3, {e3}')
        error_score_3 = -1

    # Binary classification case 1: preds 2D, and label 1D
    preds = th.concat([th.ones(100,1)*0.25, th.ones(100,1)*0.75], dim=1)
    labels = th.concat([th.zeros(20), th.ones(80)]).long()
    bin_score_1 = eval_roc_auc(preds, labels)

    # Binary classification case 2: preds 2D, and label 2D but shape[1]==1
    preds = th.concat([th.ones(100,1)*0.25, th.ones(100,1)*0.75], dim=1)
    labels = th.concat([th.zeros(20), th.ones(80)]).long().reshape(-1, 1)
    bin_score_2 = eval_roc_auc(preds, labels)

    # Multiple classification case: preds 4D and label 2D.
    preds = th.concat([th.tensor([0.75, 0.15, 0.1, 0.1]).repeat(25),
                       th.tensor([0.1, 0.75, 0.15, 0.1]).repeat(25),
                       th.tensor([0.1, 0.1, 0.75, 0.15]).repeat(25),
                       th.tensor([0.15, 0.1, 0.1, 0.75]).repeat(25)], dim=0).reshape(100, 4)
    labels = th.concat([th.zeros(25),
                        th.ones(25),
                        th.ones(25) + 1,
                        th.ones(25) + 2]).long()
    multi_class_score = eval_roc_auc(preds, labels)

    # Multip label classification case: pred 2D, label 2D
    preds = th.concat([th.tensor([0.75, 0.15]).repeat(25),
                       th.tensor([0.1, 0.75]).repeat(25),
                       th.tensor([0.1, 0.1]).repeat(25),
                       th.tensor([0.15, 0.1]).repeat(25)], dim=0).reshape(100, 2)
    labels = th.concat([th.zeros(50),
                        th.ones(50),
                        th.ones(50),
                        th.zeros(50)]).long().reshape(100, 2)
    multi_label_score = eval_roc_auc(preds, labels)

    assert error_score_1 == -1
    assert error_score_2 == -1
    assert error_score_3 == -1
    assert bin_score_1 == 0.5
    assert bin_score_2 == 0.5
    assert multi_class_score == 1.0
    assert multi_label_score == 0.3125

def test_compute_roc_auc():
    # normal cases are similar to the eval_roc_auc() test.

    # Invalid case 1: different No. of rows, return -1 and throws an exception
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75])
    labels = th.concat([th.zeros(25), th.ones(25)]).long()
    try:
        error_score_1 = compute_roc_auc(preds, labels)
    except (AssertionError, ValueError) as e1:
        print(f'Test compute_roc_auc error 1, {e1}')
        error_score_1 = -1

    # Invalid case 2: sum of probablities in one row not equal to 1.0
    preds = th.concat([th.tensor([0.75, 0.15, 0.1, 0.1]).repeat(25),
                       th.tensor([0.1, 0.75, 0.1, 0.05]).repeat(25),
                       th.tensor([0.1, 0.1, 0.75, 0.15]).repeat(25),
                       th.tensor([0.15, 0.1, 0.1, 0.75]).repeat(25)], dim=0).reshape(100, 4)
    labels = th.concat([th.zeros(25),
                        th.ones(25),
                        th.ones(25) + 1,
                        th.ones(25) + 2]).long()
    try:
        error_score_2 = compute_roc_auc(preds, labels)
    except (AssertionError, ValueError) as e2:
        print(f'Test compute_roc_auc error 2, {e2}')
        error_score_2 = -1

    # Invalid case 3: preds in 3D, labels in 1D, but with 4 classes
    preds = th.concat([th.tensor([0.75, 0.15, 0.1]).repeat(25),
                       th.tensor([0.1, 0.75, 0.15]).repeat(25),
                       th.tensor([0.15, 0.1, 0.75]).repeat(25),
                       th.tensor([0.15, 0.1, 0.75]).repeat(25)], dim=0).reshape(100, 3)
    labels = th.concat([th.zeros(25),
                        th.ones(25),
                        th.ones(25) + 1,
                        th.ones(25) + 2]).long()
    try:
        error_score_3 = compute_roc_auc(preds, labels)
    except (AssertionError, ValueError) as e3:
        print(f'Test compute_roc_auc error 3, {e3}')
        error_score_3 = -1

    # Binary classification case 1: preds 2D and label 1D
    preds = th.concat([th.ones(100,1)*0.25, th.ones(100,1)*0.75], dim=1)
    labels = th.concat([th.zeros(20), th.ones(80)]).long()
    bin_score = compute_roc_auc(preds, labels)

    # Multiple classification case: preds 2D and label 2D.
    preds = th.concat([th.tensor([0.75, 0.15, 0.05, 0.05]).repeat(25),
                       th.tensor([0.05, 0.75, 0.15, 0.05]).repeat(25),
                       th.tensor([0.05, 0.05, 0.75, 0.15]).repeat(25),
                       th.tensor([0.15, 0.05, 0.05, 0.75]).repeat(25)], dim=0).reshape(100, 4)
    labels = th.concat([th.zeros(25),
                        th.ones(25),
                        th.ones(25) + 1,
                        th.ones(25) + 2]).long()
    multi_class_score = compute_roc_auc(preds, labels)

    # Multip label classification case: pred 2D, label 2D
    preds = th.concat([th.tensor([0.75, 0.15]).repeat(25),
                       th.tensor([0.1, 0.75]).repeat(25),
                       th.tensor([0.1, 0.1]).repeat(25),
                       th.tensor([0.15, 0.1]).repeat(25)], dim=0).reshape(100, 2)
    labels = th.concat([th.zeros(50),
                        th.ones(50),
                        th.ones(50),
                        th.zeros(50)]).long().reshape(100, 2)
    multi_label_score = compute_roc_auc(preds, labels)

    assert error_score_1 == -1
    assert error_score_2 == -1
    assert error_score_3 == -1
    assert bin_score ==  0.5
    assert multi_class_score == 1.0
    assert multi_label_score == 0.3125

def test_compute_f1_score():
    # GraphStorm inputs: preds 1D in 0s and 1s, target 1D

    # Invalid case 1: preds 4D (in logits format), return -1
    preds = th.concat([th.tensor([0.75, 0.15, 0.05, 0.05]).repeat(25),
                       th.tensor([0.05, 0.75, 0.15, 0.05]).repeat(25),
                       th.tensor([0.05, 0.05, 0.75, 0.15]).repeat(25),
                       th.tensor([0.15, 0.05, 0.05, 0.75]).repeat(25)], dim=0).reshape(100, 4)
    targets = th.concat([th.zeros(25),
                        th.ones(25),
                        th.ones(25) + 1,
                        th.ones(25) + 2]).long()
    try:
        error_score_1 = compute_f1_score(preds, targets)
    except (AssertionError, ValueError):
        error_score_1 = -1

    # Invalid case 2: preds 1D in probabilities, return -1
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75])
    targets = th.concat([th.zeros(25),
                        th.ones(25),
                        th.ones(25) + 1,
                        th.ones(25) + 2]).long()
    try:
        error_score_2 = compute_f1_score(preds, targets)
    except (AssertionError, ValueError):
        error_score_2 = -1

    # Normal case: preds 1D in 0s and 1s.
    preds = th.concat([th.zeros(50), th.ones(50)])
    targets = th.concat([th.zeros(50),
                        th.ones(50)]).long()
    f1_score = compute_f1_score(preds, targets)

    assert error_score_1 == -1
    assert error_score_2 == -1
    assert f1_score == 1.0

def test_eval_acc():
    # GraphStorm inputs: 1D in 0s and 1s, or nD in logits, Labels 1D

    # Invalid case 1: 2D input in 0s and 1s format.
    preds = th.concat([th.ones(50), th.ones(50)]).reshape(50, 2).long()
    labels = th.concat([th.zeros(25), th.ones(25)]).long()
    try:
        error_acc_1 = eval_acc(preds, labels)
    except (AssertionError, ValueError) as e1:
        print(f'Test eval_acc error 1, {e1}')
        error_acc_1 = -1

    # Invalid case 2: 1D input in logits format.
    preds = th.concat([th.ones(50) * 0.25, th.ones(50) * 0.75])
    labels = th.concat([th.zeros(50), th.ones(50)]).long()
    try:
        error_acc_2 = eval_acc(preds, labels)
    except (AssertionError, ValueError) as e2:
        print(f'Test eval_acc error 2, {e2}')
        error_acc_2 = -1

    # Normal case 1: preds 1D in 0s and 1s.
    preds = th.concat([th.zeros(25), th.ones(75)]).long()
    labels = th.concat([th.zeros(50),th.ones(50)]).long()
    acc_1 = eval_acc(preds, labels)

    # Normal case 2: preds 2D with the second dim as 1.
    preds = th.unsqueeze(th.concat([th.zeros(25), th.ones(75)]).long(), 1)
    labels = th.concat([th.zeros(50),th.ones(50)]).long()
    acc_2 = eval_acc(preds, labels)

    # Normal case 3: preds 4D in logits.
    preds = th.concat([th.tensor([0.75, 0.15, 0.2, 0.2]).repeat(25),
                       th.tensor([0.2, 0.75, 0.2, 0.05]).repeat(25),
                       th.tensor([0.2, 0.2, 0.75, 0.15]).repeat(25),
                       th.tensor([0.15, 0.2, 0.2, 0.75]).repeat(25)], dim=0).reshape(100, 4)
    labels = th.concat([th.zeros(25),
                        th.ones(25) + 1,
                        th.ones(25),
                        th.ones(25) + 2]).long()
    acc_3 = eval_acc(preds, labels)

    assert error_acc_1 == -1
    assert error_acc_2 == -1
    assert acc_1 == 0.75
    assert acc_2 == 0.75
    assert acc_3 == 0.5

def test_compute_precision_recall_auc():
    # GraphStorm inputs: preds are 1D or 2D, and labels are all 1D list.

    # Invalid case 1: preds 4D, and label 1D
    preds = th.concat([th.ones(100)*0.25, th.ones(100)*0.75]).reshape(50, 4)
    labels = th.concat([th.zeros(25), th.ones(25)]).long()
    try:
        error_score_1 = compute_precision_recall_auc(preds, labels)
    except (AssertionError, ValueError):
        error_score_1 = -1

    # Normal classification case 1: preds 1D logits
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75])
    labels = th.concat([th.zeros(20), th.ones(80)]).long()
    pr_auc_1 = compute_precision_recall_auc(preds, labels)

    # Normal classification case 2: preds 1D in 0s and 1s
    preds = th.concat([th.ones(50), th.ones(50)]).long()
    labels = th.concat([th.zeros(20), th.ones(80)]).long()
    pr_auc_2 = compute_precision_recall_auc(preds, labels)

    # Binary classification case: preds 2D in logits
    preds = th.concat([th.ones(100,1)*0.25, th.ones(100,1)*0.75], dim=1)
    labels = th.concat([th.zeros(20), th.ones(80)]).long()
    bin_pr_auc = compute_precision_recall_auc(preds, labels)

    assert error_score_1 == -1
    assert pr_auc_1 == 0.9625
    assert pr_auc_2 == 0.9
    assert bin_pr_auc == 0.9

def test_compute_per_class_roc_auc():
    # GraphStorm inputs: preds are 1D or 2D, and labels are all 1D list.

    # Invalid case 1: preds 1D
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75])
    targets = th.concat([th.zeros(25), th.ones(25)]).long()
    try:
        error_score_1 = compute_per_class_roc_auc(preds, targets)
    except (AssertionError, ValueError) as e1:
        print(f'Test compute_per_class_roc_auc error 1, {e1}')
        error_score_1 = -1

    # Invalid case 2: preds 2D, but shape[1] == 1
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75]).reshape(-1, 1)
    targets = th.concat([th.zeros(25), th.ones(25)]).long()
    try:
        error_score_2 = compute_per_class_roc_auc(preds, targets)
    except (AssertionError, ValueError) as e2:
        print(f'Test compute_per_class_roc_auc error 2, {e2}')
        error_score_2 = -1

    # Invalid case 3: targets 1D
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75]).reshape(-1, 2)
    targets = th.concat([th.zeros(25), th.ones(25)]).long()
    try:
        error_score_3 = compute_per_class_roc_auc(preds, targets)
    except (AssertionError, ValueError) as e3:
        print(f'Test compute_per_class_roc_auc error 3, {e3}')
        error_score_3 = -1

    # Invalid case 4: targets 2D, but shape[1] == 1
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75]).reshape(-1, 2)
    targets = th.concat([th.zeros(25), th.ones(25)]).long().reshape(-1, 1)
    try:
        error_score_4 = compute_per_class_roc_auc(preds, targets)
    except (AssertionError, ValueError) as e4:
        print(f'Test compute_per_class_roc_auc error 4, {e4}')
        error_score_4 = -1

    # Invalid case 5: preds and targets 2D, but have different shape[1]
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75]).reshape(-1, 4)
    targets = th.concat([th.zeros(25), th.ones(25)]).long().reshape(-1, 2)
    try:
        error_score_5 = compute_per_class_roc_auc(preds, targets)
    except (AssertionError, ValueError) as e5:
        print(f'Test compute_per_class_roc_auc error 5, {e5}')
        error_score_5 = -1

    # Normal case: preds and targets 2D, both shape[1] = 4
    preds = th.concat([th.tensor([0.75, 0.05, 0.1, 0.1]).repeat(25),
                       th.tensor([0.1, 0.75, 0.05, 0.1]).repeat(25),
                       th.tensor([0.1, 0.1, 0.75, 0.05]).repeat(25),
                       th.tensor([0.05, 0.1, 0.1, 0.75]).repeat(25)], dim=0).reshape(100, 4)
    targets = th.concat([th.zeros(200), th.ones(200)]).long().reshape(-1, 4)
    per_class_scores = compute_per_class_roc_auc(preds, targets)

    assert error_score_1 == -1
    assert error_score_2 == -1
    assert error_score_3 == -1
    assert error_score_4 == -1
    assert error_score_5 == -1
    assert per_class_scores['overall avg'] == 0.5
    assert per_class_scores[0] == 0.125
    assert per_class_scores[3] == 0.5

def test_ClassificationMetrics():
    eval_metric_list = ["accuracy", "hit_at_5", "hit_at_10"]
    metric = ClassificationMetrics(eval_metric_list, multilabel=False)

    assert "accuracy" in metric.metric_comparator
    assert "accuracy" in metric.metric_function
    assert "accuracy" in metric.metric_eval_function

    assert "precision_recall" in metric.metric_comparator
    assert "precision_recall" in metric.metric_function
    assert "precision_recall" in metric.metric_eval_function

    assert "roc_auc" in metric.metric_comparator
    assert "roc_auc" in metric.metric_function
    assert "roc_auc" in metric.metric_eval_function

    assert "f1_score" in metric.metric_comparator
    assert "f1_score" in metric.metric_function
    assert "f1_score" in metric.metric_eval_function

    assert "per_class_f1_score" in metric.metric_comparator
    assert "per_class_f1_score" in metric.metric_function
    assert "per_class_f1_score" in metric.metric_eval_function

    assert "per_class_roc_auc" in metric.metric_comparator
    assert "per_class_roc_auc" in metric.metric_function
    assert "per_class_roc_auc" in metric.metric_eval_function

    assert "hit_at_5" in metric.metric_comparator
    assert "hit_at_5" in metric.metric_function
    assert "hit_at_5" in metric.metric_eval_function
    assert "hit_at_10" in metric.metric_comparator
    assert "hit_at_10" in metric.metric_function
    assert "hit_at_10" in metric.metric_eval_function

    signature = inspect.signature(metric.metric_function["hit_at_5"])
    assert signature.parameters["k"].default == 5
    signature = inspect.signature(metric.metric_function["hit_at_10"])
    assert signature.parameters["k"].default == 10

    metric.assert_supported_metric("accuracy")
    metric.assert_supported_metric("precision_recall")
    metric.assert_supported_metric("roc_auc")
    metric.assert_supported_metric("f1_score")
    metric.assert_supported_metric("per_class_f1_score")
    metric.assert_supported_metric("per_class_roc_auc")
    metric.assert_supported_metric("hit_at_5")
    metric.assert_supported_metric("hit_at_10")

    pass_assert = False
    try:
        metric.assert_supported_metric("hit_at_ten")
        pass_assert = True
    except:
        pass_assert = False
    assert not pass_assert

def test_compute_hit_at_classification():
    preds = th.arange(100) / 102
    # preds is in a format as [probe_of_0, probe_of_1]
    preds = th.stack([preds, 1-preds]).T
    preds2 = preds[:,1].unsqueeze(1)
    labels = th.zeros((100,)) # 1D label tensor
    labels2 = th.zeros((100,1)) # 2D label tensor
    labels[0] = 1
    labels[2] = 1
    labels[4] = 1
    labels[11] = 1

    labels2[0][0] = 1
    labels2[2][0] = 1
    labels2[4][0] = 1
    labels2[11][0] = 1

    hit_at = compute_hit_at_classification(preds, labels, 5)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds, labels, 10)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds, labels, 20)
    assert hit_at == 4

    hit_at = compute_hit_at_classification(preds2, labels, 5)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds2, labels, 10)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds2, labels, 20)
    assert hit_at == 4

    shuff_idx = th.randperm(100)
    preds = preds[shuff_idx]
    preds2 = preds2[shuff_idx]
    labels = labels[shuff_idx]
    labels2 = labels2[shuff_idx]

    hit_at = compute_hit_at_classification(preds, labels, 5)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds, labels, 10)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds, labels, 20)
    assert hit_at == 4
    hit_at = compute_hit_at_classification(preds, labels2, 5)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds, labels2, 10)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds, labels2, 20)
    assert hit_at == 4

    hit_at = compute_hit_at_classification(preds2, labels, 5)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds2, labels, 10)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds2, labels, 20)
    assert hit_at == 4
    hit_at = compute_hit_at_classification(preds2, labels2, 5)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds2, labels2, 10)
    assert hit_at == 3
    hit_at = compute_hit_at_classification(preds2, labels2, 20)
    assert hit_at == 4

def test_LinkPredictionMetrics():
    eval_metric_list = ["mrr", "hit_at_5", "hit_at_10", "amri"]
    metric = LinkPredictionMetrics(eval_metric_list)

    assert "mrr" in metric.metric_comparator
    assert "mrr" in metric.metric_function
    assert "mrr" in metric.metric_eval_function
    assert "amri" in metric.metric_eval_function


    assert "hit_at_5" in metric.metric_comparator
    assert "hit_at_5" in metric.metric_function
    assert "hit_at_5" in metric.metric_eval_function
    assert "hit_at_10" in metric.metric_comparator
    assert "hit_at_10" in metric.metric_function
    assert "hit_at_10" in metric.metric_eval_function

    signature = inspect.signature(metric.metric_function["hit_at_5"])
    assert signature.parameters["k"].default == 5
    signature = inspect.signature(metric.metric_function["hit_at_10"])
    assert signature.parameters["k"].default == 10

    metric.assert_supported_metric("mrr")
    metric.assert_supported_metric("amri")
    metric.assert_supported_metric("hit_at_5")
    metric.assert_supported_metric("hit_at_10")

    with pytest.raises(AssertionError):
        metric.assert_supported_metric("hit_at_ten")

def test_compute_hit_at_link_prediction():
    preds = 1 - th.arange(100) / 120    # preds for all positive and negative samples
    # 1 indicates positive samples
    idx_positive = th.zeros(100)
    idx_positive[2] = 1
    idx_positive[4] = 1
    idx_positive[5] = 1
    idx_positive[7] = 1
    idx_positive[15] = 1
    idx_positive[21] = 1
    idx_positive[99] = 1
    ranking = th.argsort(preds, descending=True)[idx_positive.bool()]

    hit_at = compute_hit_at_link_prediction(ranking, 5)
    assert hit_at == 3 / 7
    hit_at = compute_hit_at_link_prediction(ranking, 10)
    assert hit_at == 4 / 7
    hit_at = compute_hit_at_link_prediction(ranking, 20)
    assert hit_at == 5 / 7
    hit_at = compute_hit_at_link_prediction(ranking, 100)
    assert hit_at == 7 / 7
    hit_at = compute_hit_at_link_prediction(ranking, 200)
    assert hit_at == 7 / 7

def test_compute_amri():
    # Compute amri when candidate lists vary in size
    ranks = th.tensor([4, 1, 4, 5, 1, 5, 5, 1, 2, 3])
    candidate_lists = th.tensor([10, 12, 7, 5, 4, 10, 12, 10, 20, 10])

    # Use the definition from the paper to verify the values
    def amri_definition(rankings, candidate_tensor) -> float:
        mr = th.sum(rankings) / rankings.shape[0]
        emr = (1 / (2 * candidate_tensor.shape[0])) * th.sum(candidate_tensor)
        expected_amri = 1 - ((mr - 1) / emr)
        return expected_amri

    actual_amri = compute_amri(ranks, candidate_lists)
    expected_amri = amri_definition(ranks, candidate_lists)
    assert_almost_equal(
        actual_amri,
        expected_amri,
        decimal=6
    )

    # Compute amri when all lists have the same size
    candidate_size = th.tensor([10])
    actual_amri = compute_amri(ranks, candidate_size)
    expected_amri = amri_definition(ranks, candidate_size)
    assert_almost_equal(
        actual_amri,
        expected_amri,
        decimal=4
    )

    # amri should be 0 when all ranks are the expected rank plus one
    ranks = th.tensor([10]*5) # MR is 10, MR-1 is 9
    candidate_size = th.tensor([18]*5) # E [MR] is (1/(2*5))*(18*5) = 9
    actual_amri = compute_amri(ranks, candidate_size)
    expected_amri = 0
    assert_almost_equal(
        actual_amri,
        expected_amri,
        decimal=2
    )

def test_compute_precision_recall_fscore():
    # GraphStorm inputs: preds 1D or 2D ints, target 1D or 2D ints.

    # Invalid case 1: preds 2D (in logits format) but labels 1D, assertion error, return -1
    preds = th.concat([th.tensor([0.75, 0.15, 0.05, 0.05]).repeat(25),
                       th.tensor([0.05, 0.75, 0.15, 0.05]).repeat(25),
                       th.tensor([0.05, 0.05, 0.75, 0.15]).repeat(25),
                       th.tensor([0.15, 0.05, 0.05, 0.75]).repeat(25)], dim=0).reshape(100, 4)
    targets = th.concat([th.zeros(25),
                        th.ones(25),
                        th.ones(25) + 1,
                        th.ones(25) + 2]).long()
    try:
        error_score_1 = compute_precision_recall_fscore(preds, targets)
    except (AssertionError, ValueError):
        error_score_1 = -1

    # Invalid case 2: preds 1D in probabilities, assertion error, return -1
    preds = th.concat([th.ones(50)*0.25, th.ones(50)*0.75])
    targets = th.concat([th.zeros(25),
                        th.ones(25),
                        th.ones(25) + 1,
                        th.ones(25) + 2]).long()
    try:
        error_score_2 = compute_precision_recall_fscore(preds, targets)
    except (AssertionError, ValueError):
        error_score_2 = -1

    # Invalid case 3: pred in 3D, assertion error, return -1
    preds = th.ones(100, 2, 2).int()
    targets = th.concat([th.zeros(25),
                        th.ones(25),
                        th.ones(25) + 1,
                        th.ones(25) + 2]).long()
    try:
        error_score_3 = compute_precision_recall_fscore(preds, targets)
    except (NotImplementedError):
        error_score_3 = -1

    assert error_score_1 == -1
    assert error_score_2 == -1
    assert error_score_3 == -1

    # Normal case 1: preds 1D and labels 1D in 0s and 1s, using binary
    preds = th.concat([th.zeros(40), th.ones(10), th.zeros(40), th.ones(10)]).int()
    targets = th.concat([th.zeros(50), th.ones(50)]).int()
    precision, recall, fscore = compute_precision_recall_fscore(preds, targets)

    assert precision == 0.2
    assert recall == 0.5
    expected_fscore = 0.3846
    assert_almost_equal(
        fscore,
        expected_fscore,
        decimal=4
    )

    # Normal case 2: preds 1D and labels 1D with multiple classes, using marco avg
    preds = th.concat([th.zeros(20), th.ones(5),
                       th.zeros(5), th.ones(20),
                       th.zeros(5), th.ones(20)+1,
                       th.zeros(5), th.ones(20)+2]).int()
    targets = th.concat([th.zeros(25), th.ones(25), th.ones(25)+1, th.ones(25)+2]).int()
    precision, recall, fscore = compute_precision_recall_fscore(preds, targets)

    assert precision == 0.8
    expected_recall = 0.8428
    assert_almost_equal(
        recall,
        expected_recall,
        decimal=4
    )
    expected_fscore = 0.8277
    assert_almost_equal(
        fscore,
        expected_fscore,
        decimal=4
    )
    
    # Normal case 3: preds 2D and labels 2D with multiple lables
    preds = th.concat([th.tensor([1, 0, 1, 0]).repeat(25),
                       th.tensor([0, 1, 0, 1]).repeat(25),
                       th.tensor([1, 0, 1, 0]).repeat(25),
                       th.tensor([0, 1, 0, 1]).repeat(25)], dim=0).reshape(100, 4)
    targets = th.concat([th.tensor([1, 0, 0, 0]).repeat(25),
                         th.tensor([0, 1, 0, 0]).repeat(25),
                         th.tensor([0, 0, 1, 0]).repeat(25),
                         th.tensor([0, 0, 0, 1]).repeat(25)], dim=0).reshape(100, 4)
    precision, recall, fscore = compute_precision_recall_fscore(preds, targets)
    
    expected_precisions = [1, 1, 1, 1]
    assert_almost_equal(
        precision,
        expected_precisions,
        decimal=4
    )
    expected_recalls = [0.5, 0.5, 0.5, 0.5]
    assert_almost_equal(
        recall,
        expected_recalls,
        decimal=4
    )
    expected_fscores = [0.5555, 0.5555, 0.5555, 0.5555]
    assert_almost_equal(
        fscore,
        expected_fscores,
        decimal=4
    )

def test_compute_precision():
    """ Test get precision results.
    
    Because the major computation occurs in the `compute_precision_recall_fscore` function, this
    test will only check if precision is return, not the other metrics.
    """
    preds = th.concat([th.zeros(40), th.ones(10), th.zeros(40), th.ones(10)]).int()
    targets = th.concat([th.zeros(50), th.ones(50)]).int()
    precision = compute_precision(preds, targets)

    assert precision == 0.2

def test_compute_recall():
    """ Test get recall results.
    
    Because the major computation occurs in the `compute_precision_recall_fscore` function, this
    test will only check if recall is return, not the other metrics.
    """
    preds = th.concat([th.zeros(40), th.ones(10), th.zeros(40), th.ones(10)]).int()
    targets = th.concat([th.zeros(50), th.ones(50)]).int()
    recall = compute_recall(preds, targets)

    assert recall == 0.5

@pytest.mark.parametrize("beta", [0.5, 1, 2, 10])
def test_compute_fscore(beta):
    """ Test get fscore results with different beta values.
    
    Because the major computation occurs in the `compute_precision_recall_fscore` function, this
    test will only check if precision is return, not the other metrics.
    """
    preds = th.concat([th.zeros(40), th.ones(10), th.zeros(40), th.ones(10)]).int()
    targets = th.concat([th.zeros(50), th.ones(50)]).int()
    fscore = compute_fscore(preds, targets, beta)

    precision = 0.2
    recall = 0.5
    expected_fscore = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    assert_almost_equal(fscore, expected_fscore, decimal=4)

def test_compute_precision_at_recall():
    """ Test get precision at certain recall with different beta values.
    """
    # GraphStorm inputs: preds 1D or 2D ints, target 1D or 2D ints.

    # Invalid case 1: preds 2D (in logits format) but labels 1D, assertion error, return -1
    preds = th.concat([th.tensor([0.75, 0.15, 0.05, 0.05]).repeat(25),
                       th.tensor([0.05, 0.75, 0.15, 0.05]).repeat(25),
                       th.tensor([0.05, 0.05, 0.75, 0.15]).repeat(25),
                       th.tensor([0.15, 0.05, 0.05, 0.75]).repeat(25)], dim=0).reshape(100, 4)
    targets = th.concat([th.zeros(50),
                         th.ones(50)]).long()
    try:
        error_score_1 = compute_precision_at_recall(preds, targets)
    except (AssertionError, ValueError):
        error_score_1 = -1

    # Invalid case 2: pred in 3D, assertion error, return -1
    preds = th.ones(100, 2, 2).int()
    targets = th.concat([th.zeros(50),
                         th.ones(50)]).long()
    try:
        error_score_2 = compute_precision_at_recall(preds, targets)
    except (AssertionError, ValueError):
        error_score_2 = -1

    # Invalid case 3: preds 1D with multiple classes, labels 1D
    targets = th.tensor([0, 0, 1, 2])
    preds = th.tensor([0.1, 0.4, 0.35, 0.8])
    try:
        error_score_3 = compute_precision_at_recall(preds, targets)
    except (AssertionError, ValueError):
        error_score_3 = -1

    assert error_score_1 == -1
    assert error_score_2 == -1
    assert error_score_3 == -1

    # Normal case 1: preds 1D and labels 1D in 0s and 1s, using binary; existing beta in recall
    targets = th.tensor([0, 0, 1, 1])
    preds = th.tensor([0.1, 0.4, 0.35, 0.8])
    beta = 0.5

    precision = compute_precision_at_recall(preds, targets, beta)
    assert precision == 1.

    # Normal case 2: preds 1D and labels 1D in 0s and 1s, using binary; existing beta in recall
    targets = th.tensor([0, 0, 1, 1])
    preds = th.tensor([0.1, 0.4, 0.35, 0.8])
    beta = 1

    precision = compute_precision_at_recall(preds, targets, beta)
    assert_almost_equal(
        precision,
        0.6667,
        decimal=4
    )

    # Normal case 3: preds 1D and labels 1D in 0s and 1s, using binary; non-existing beta in recall
    targets = th.tensor([0, 0, 1, 1])
    preds = th.tensor([0.1, 0.4, 0.35, 0.8])
    beta = 0.6

    precision = compute_precision_at_recall(preds, targets, beta)
    assert precision == 1.

def test_compute_recall_at_precision():
    """ Test get recall at certain precision with different beta values.
    """
    # GraphStorm inputs: preds 1D or 2D ints, target 1D or 2D ints.

    # Invalid case 1: preds 2D (in logits format) but labels 1D, assertion error, return -1
    preds = th.concat([th.tensor([0.75, 0.15, 0.05, 0.05]).repeat(25),
                       th.tensor([0.05, 0.75, 0.15, 0.05]).repeat(25),
                       th.tensor([0.05, 0.05, 0.75, 0.15]).repeat(25),
                       th.tensor([0.15, 0.05, 0.05, 0.75]).repeat(25)], dim=0).reshape(100, 4)
    targets = th.concat([th.zeros(50),
                         th.ones(50)]).long()
    try:
        error_score_1 = compute_recall_at_precision(preds, targets)
    except (AssertionError, ValueError):
        error_score_1 = -1

    # Invalid case 2: pred in 3D, assertion error, return -1
    preds = th.ones(100, 2, 2).int()
    targets = th.concat([th.zeros(50),
                         th.ones(50)]).long()
    try:
        error_score_2 = compute_recall_at_precision(preds, targets)
    except (AssertionError, ValueError):
        error_score_2 = -1

    # Invalid case 3: preds 1D with multiple classes, labels 1D
    targets = th.tensor([0, 0, 1, 2])
    preds = th.tensor([0.1, 0.4, 0.35, 0.8])
    try:
        error_score_3 = compute_recall_at_precision(preds, targets)
    except (AssertionError, ValueError):
        error_score_3 = -1

    assert error_score_1 == -1
    assert error_score_2 == -1
    assert error_score_3 == -1

    # Normal case 1: preds 1D and labels 1D in 0s and 1s, using binary; existing beta in precision
    targets = th.tensor([0, 0, 1, 1])
    preds = th.tensor([0.1, 0.4, 0.35, 0.8])
    beta = 0.5

    recall = compute_recall_at_precision(preds, targets, beta)
    assert recall == 1.

    # Normal case 2: preds 1D and labels 1D in 0s and 1s, using binary; existing beta in precision
    targets = th.tensor([0, 0, 1, 1])
    preds = th.tensor([0.1, 0.4, 0.35, 0.8])
    beta = 1

    recall = compute_recall_at_precision(preds, targets, beta)
    assert recall == 0.5

    # Normal case 3: preds 1D and labels 1D in 0s and 1s, using binary; beta not in precision
    targets = th.tensor([0, 0, 1, 1])
    preds = th.tensor([0.1, 0.4, 0.35, 0.8])
    beta = 0.6

    recall = compute_recall_at_precision(preds, targets, beta)
    assert recall == 1.

    # Normal case 4: preds 1D and labels 1D in 0s and 1s, using binary;
    # beta is too small to find the corresponding recall
    targets = th.tensor([0, 0, 1, 1])
    preds = th.tensor([0.1, 0.4, 0.35, 0.8])
    beta = 0.1

    recall = compute_recall_at_precision(preds, targets, beta)
    assert recall == 0.
