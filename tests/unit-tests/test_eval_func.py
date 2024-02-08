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
import numpy as np
import torch as th

from numpy.testing import assert_almost_equal
from graphstorm.eval.eval_func import compute_mse, compute_rmse, compute_roc_auc, eval_roc_auc
from graphstorm.eval.eval_func import compute_f1_score, eval_acc
from graphstorm.eval.eval_func import compute_precision_recall_auc, compute_per_class_roc_auc

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
    
    # Normal case 2: preds 4D in logits.
    preds = th.concat([th.tensor([0.75, 0.15, 0.2, 0.2]).repeat(25),
                       th.tensor([0.2, 0.75, 0.2, 0.05]).repeat(25),
                       th.tensor([0.2, 0.2, 0.75, 0.15]).repeat(25),
                       th.tensor([0.15, 0.2, 0.2, 0.75]).repeat(25)], dim=0).reshape(100, 4)
    labels = th.concat([th.zeros(25),
                        th.ones(25) + 1,
                        th.ones(25),
                        th.ones(25) + 2]).long()
    acc_2 = eval_acc(preds, labels)

    assert error_acc_1 == -1
    assert error_acc_2 == -1
    assert acc_1 == 0.75
    assert acc_2 == 0.5

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

if __name__ == '__main__':
    test_compute_mse()
    test_compute_rmse()

    test_eval_roc_auc()
    test_compute_roc_auc()
    test_compute_per_class_roc_auc()

    test_compute_f1_score()

    test_eval_acc()

    test_compute_precision_recall_auc()
