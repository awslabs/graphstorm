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

    Evaluation functions
"""
from enum import Enum
from functools import partial
import operator
import warnings
import numpy as np
import torch as th
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, classification_report

SUPPORTED_CLASSIFICATION_METRICS = {'accuracy', 'precision_recall', \
    'roc_auc', 'f1_score', 'per_class_f1_score'}
SUPPORTED_REGRESSION_METRICS = {'rmse', 'mse'}
SUPPORTED_LINK_PREDICTION_METRICS = {"mrr"}

class ClassificationMetrics:
    """ object that compute metrics for classification tasks.
    """
    def __init__(self, multilabel):
        self.supported_metrics = SUPPORTED_CLASSIFICATION_METRICS
        self.multilabel = multilabel

        # This is the operator used to compare whether current value is better than the current best
        self.metric_comparator = {}
        self.metric_comparator["accuracy"] = operator.le
        self.metric_comparator["precision_recall"] = operator.le
        self.metric_comparator["roc_auc"] = operator.le
        self.metric_comparator["f1_score"] = operator.le
        self.metric_comparator["per_class_f1_score"] = comparator_per_class_f1_score

        # This is the operator used to measure each metric performance in training
        self.metric_function = {}
        self.metric_function["accuracy"] = partial(compute_acc, multilabel=self.multilabel)
        self.metric_function["precision_recall"] = compute_precision_recall_auc
        self.metric_function["roc_auc"] = compute_roc_auc
        self.metric_function["f1_score"] = compute_f1_score
        self.metric_function["per_class_f1_score"] = compute_f1_score

        # This is the operator used to measure each metric performance in evaluation
        self.metric_eval_function = {}
        self.metric_eval_function["accuracy"] = partial(compute_acc, multilabel=self.multilabel)
        self.metric_eval_function["precision_recall"] = compute_precision_recall_auc
        self.metric_eval_function["roc_auc"] = compute_roc_auc
        self.metric_eval_function["f1_score"] = compute_f1_score
        self.metric_eval_function["per_class_f1_score"] = compute_per_class_f1_score

    def assert_supported_metric(self, metric):
        """ check if the given metric is supported.
        """
        assert metric in self.supported_metrics, \
            f"Metric {metric} not supported for classification"

    def init_best_metric(self, metric):
        """
        Return the initial value for the metric to keep track of the best metric.
        Parameters
        ----------
        metric: the metric to initialize

        Returns
        -------

        """
        # Need to check if the given metric is supported first
        self.assert_supported_metric(metric)
        return 0


class RegressionMetrics:
    """ object that compute metrics for regression tasks.
    """
    def __init__(self):
        self.supported_metrics = SUPPORTED_REGRESSION_METRICS

        # This is the operator used to compare whether current value is better than the current best
        self.metric_comparator = {}
        self.metric_comparator["rmse"] = operator.ge
        self.metric_comparator["mse"] = operator.ge

        # This is the operator used to measure each metric performance
        self.metric_function = {}
        self.metric_function["rmse"] = compute_rmse
        self.metric_function["mse"] = compute_mse

    def assert_supported_metric(self, metric):
        """ check if the given metric is supported.
        """
        assert metric in self.supported_metrics, \
            f"Metric {metric} not supported for regression"

    def init_best_metric(self, metric):
        """
        Return the initial value for the metric to keep track of the best metric.
        Parameters
        ----------
        metric: the metric to initialize

        Returns
        -------

        """
        # Need to check if the given metric is supported first
        self.assert_supported_metric(metric)
        return np.finfo(np.float32).max

class LinkPredictionMetrics:
    """ object that compute metrics for LP tasks.
    """
    def __init__(self):
        self.supported_metrics = SUPPORTED_LINK_PREDICTION_METRICS

        # This is the operator used to compare whether current value is better than the current best
        self.metric_comparator = {}
        self.metric_comparator["mrr"] = operator.le

    def assert_supported_metric(self, metric):
        """ check if the given metric is supported.
        """
        assert metric in self.supported_metrics, \
            f"Metric {metric} not supported for link prediction"

    def init_best_metric(self, metric):
        """
        Return the initial value for the metric to keep track of the best metric.
        Parameters
        ----------
        metric: the metric to initialize

        Returns
        -------

        """
        # Need to check if the given metric is supported first
        self.assert_supported_metric(metric)
        return 0

def labels_to_one_hot(labels, total_labels):
    '''
    This function converts the original labels to an one hot array
    Parameters
    ----------
    labels
    total_labels

    Returns
    -------

    '''
    if len(labels.shape)>1:
        return labels
    one_hot=np.zeros(shape=(len(labels),total_labels))
    for i, label in enumerate(labels):
        one_hot[i,label]=1
    return one_hot

def eval_roc_auc(logits,labels):
    '''
    Parameters
    ----------
    logits : Target scores.
    labels: Array-like of shape (n_samples,) or (n_samples, n_classes) True labels or
            binary label indicators. The binary and multiclass cases expect labels with
            shape (n_samples,) while the multilabel case expects binary label indicators
            with shape (n_samples, n_classes).

    Returns
    -------
    The roc_auc_score

    '''
    predicted_labels=logits
    predicted_labels=predicted_labels.detach().cpu().numpy()
    labels=labels.detach().cpu().numpy()
    # The roc_auc_score function computes the area under the receiver operating characteristic
    # (ROC) curve, which is also denoted by AUC or AUROC. The following returns the average AUC.
    rocauc_list = []
    labels=labels_to_one_hot(labels, predicted_labels.shape[1])
    for i in range(labels.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(labels[:, i] == 1) > 0 and np.sum(labels[:, i] == 0) > 0:
            is_labeled = labels[:, i] == labels[:, i]
            rocauc_list.append(roc_auc_score(labels[is_labeled, i],
                                             predicted_labels[is_labeled, i]))

    if len(rocauc_list) == 0:
        print('No positively labeled data available. Cannot compute ROC-AUC.')
        return 0

    return sum(rocauc_list) / len(rocauc_list)


def eval_acc(pred, labels):
    """compute evaluation accuracy.
    """
    if pred.dim() > 1:
        # if pred has dimension > 1, it has full logits instead of final prediction
        assert th.is_floating_point(pred), "Logits are expected to be float type"
        pred = pred.argmax(dim=1)
    # Check if pred is integer tensor
    assert(not th.is_floating_point(pred) and not th.is_complex(pred)), "Predictions are " \
        "expected to be integer type"
    return th.sum(pred.cpu() == labels.cpu()).item() / len(labels)


def compute_f1_score(y_preds, y_targets):
    """ compute macro_average f1 score
    """
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    report = classification_report(y_pred=y_pred, y_true=y_true, output_dict=True)
    return report['macro avg']['f1-score']


def compute_per_class_f1_score(y_preds, y_targets):
    """ compute f1 score per class
    """
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    report = classification_report(y_pred=y_pred, y_true=y_true, output_dict=True)
    return report


def comparator_per_class_f1_score(best_report, current_report):
    """ compare method for f1 score per class
    """
    return best_report['macro avg']['f1-score'] < current_report['macro avg']['f1-score']\
        if best_report != 0 else 0 < current_report['macro avg']['f1-score']


def compute_acc_lp(pos_score, neg_score):
    """
    This function calculates the LP accuracy. It is a cheap and fast way to evaluate the
    accuracy of the model. The scores are ranked from larger to smaller. If all the pos_scores
    are ranked before all the neg_scores then the value returned is 1 that is the maximum.

    Parameters
    ----------
    pos_score : the positive scores
    neg_score : the negative scores

    Returns
    -------
    lp_score : the lp accuracy.

    """
    num_pos=len(pos_score)
    # perturb object
    scores = th.cat([pos_score, neg_score], dim=0)
    scores = th.sigmoid(scores)
    _, rankings = th.sort(scores, dim=0, descending=True)
    rankings = rankings.cpu().detach().numpy()
    rankings = rankings <= num_pos
    lp_score = sum(rankings[:num_pos]) / num_pos

    return {"lp_fast_score": lp_score}


def compute_roc_auc(y_preds, y_targets, weights=None):
    """ compute ROC's auc score
    """
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    if weights is not None:
        weights = weights.cpu().numpy()
    auc_score = -1
    # adding checks since in certain cases the auc might not be defined we do not want to fail
    # the code
    try:
        auc_score = roc_auc_score(y_true, y_pred, sample_weight=weights, multi_class='ovr')
    except ValueError as e:
        print("Failure found during evaluation of the auc metric returning -1", e)
    return auc_score


class PRKeys(str, Enum):
    """ Enums support iteration in definition order--order matters here
    """
    PRECISION = "precision"
    RECALL = "recall"
    THRESHOLD = "threshold"


def compute_precision_recall_auc(y_preds, y_targets, weights=None):
    """ compute precision, recall, and auc values.
    """
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()
    keys = [key.value for key in PRKeys]
    auc_score = -1
    # adding checks since in certain cases the auc might not be defined we do not want to fail
    # the code
    try:
        pr_curve = dict(zip(keys, precision_recall_curve(y_true, y_pred, sample_weight=weights)))
        precision, recall = pr_curve[PRKeys.PRECISION], pr_curve[PRKeys.RECALL]
        auc_score = auc(recall, precision)
    except ValueError as e:
        print("Failure found during evaluation of the auc metric returning -1", e)
    return auc_score

def compute_acc(pred, labels, multilabel):
    '''Compute accuracy.

    Parameters
    ----------
    pred : tensor
        a 1-D tensor for single-label classification and 2-D tensor for multi-label classification.
        For 2-D tensor, the number of column is the number of labels.
    labels : tensor
        a 1-D tensor for single-label classification and 2-D tensor for multi-label classification.
        For 2-D tensor, the number of column is the number of labels.
    multilabel : bool
        Whether this is a multi-label classification task.

    Returns
    -------
        A 1-D tensor that stores the accuracy.
    '''
    if multilabel:
        return eval_roc_auc(pred, labels)
    else:
        return eval_acc(pred, labels)

def compute_rmse(pred, labels):
    """ compute RMSE for regression.
    """
    # TODO: check dtype of label before training or evaluation
    assert th.is_floating_point(pred) and th.is_floating_point(labels), \
        "prediction and labels must be floating points"

    assert pred.shape == labels.shape, \
        f"prediction and labels have different shapes. {pred.shape} vs. {labels.shape}"
    if pred.dtype != labels.dtype:
        warnings.warn("prediction and labels have different data types: "
                      f"{pred.dtype} vs. {labels.dtype}")
        warnings.warn("casting pred to the same dtype as labels")
        pred = pred.type(labels.dtype) # cast pred to the same dtype as labels.

    diff = pred.cpu() - labels.cpu()
    return th.sqrt(th.mean(diff * diff)).cpu().item()

def compute_mse(pred, labels):
    """ compute MSE for regression
    """
    # TODO: check dtype of label before training or evaluation
    assert th.is_floating_point(pred) and th.is_floating_point(labels), \
        "prediction and labels must be floating points"

    assert pred.shape == labels.shape, \
        f"prediction and labels have different shapes. {pred.shape} vs. {labels.shape}"
    if pred.dtype != labels.dtype:
        warnings.warn("prediction and labels have different data types: "
                      f"{pred.dtype} vs. {labels.dtype}")
        warnings.warn("casting pred to the same dtype as labels")
        pred = pred.type(labels.dtype) # cast pred to the same dtype as labels.

    diff = pred.cpu() - labels.cpu()
    return th.mean(diff * diff).cpu().item()
