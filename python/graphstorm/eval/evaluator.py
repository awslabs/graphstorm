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

    Evaluator for different tasks.
"""

import abc
from statistics import mean
import torch as th

from .eval_func import SUPPORTED_HIT_AT_METRICS
from .eval_func import ClassificationMetrics, RegressionMetrics, LinkPredictionMetrics
from .utils import broadcast_data
from ..config.config import (EARLY_STOP_AVERAGE_INCREASE_STRATEGY,
                             EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY,
                             LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL)
from ..utils import get_rank, get_world_size, barrier


def early_stop_avg_increase_judge(val_score, val_perf_list, comparator):
    """
    Stop the training early if the val_score `decreases` for the last early stop round.

    Note: val_score < Average[val scores in last K steps]

    Parameters
    ----------
    val_score: float
        Target validation score.
    val_loss_list: list
        A list holding the history validation scores.
    comparator: operator op
        Comparator

    Returns
    -------
    early_stop : A boolean indicating the early stop
    """
    avg_score = mean(val_perf_list)
    return comparator(val_score, avg_score)

def early_stop_cons_increase_judge(val_score, val_perf_list, comparator):
    """
    Stop the training early if for the last K consecutive steps the validation
    scores are `decreasing`. See the third approach in the Prechelt, L., 1998.
    Early stopping-but when?.
    In Neural Networks: Tricks of the trade (pp. 55-69). Springer, Berlin, Heidelberg.

    Parameters
    ----------
    val_score: float
        Target validation score.
    val_loss_list: list
        A list holding the history validation scores.
    comparator: operator op
        Comparator.

    Returns
    -------
    early_stop : A boolean indicating the early stop.
    """
    early_stop = True
    for old_val_score in val_perf_list:
        early_stop = early_stop and comparator(val_score, old_val_score)

    return early_stop

def get_val_score_rank(val_score, val_perf_rank_list, comparator):
    """
    Compute the rank of the given validation score with the given comparator.

    Here use the most naive method, i.e., scan the entire list once to get the rank.
    For the same value, will treat the given validation score as the next rank. For example, in a
    list [1., 1., 2., 2., 3., 4.], the given value 2 will be ranked to the 5th highest score.

    Later on if need to increase the speed, could use more complex data structure, e.g. LinkedList

    Parameters
    ----------
    val_score: float
        Target validation score.
    val_perf_rank_list: list
        A list holding the history validation scores.
    comparator: operator op
        Comparator

    Returns
    -------
    rank : An integer indicating the rank of the given validation score in the
           existing validation performance rank list.
    """
    rank = 1
    for existing_score in val_perf_rank_list:
        if comparator(val_score, existing_score):
            rank += 1

    return rank


class GSgnnPredictionEvalInterface():
    """ Interface for Prediction evaluation functions.

    The interface set the two abstract methods for prediction tasks, i.e., **Classification**
    and **Regression**, which should be implemented if inherit this interface.

    1. ``evaluate()`` method, which will be called by different **Trainers** in their ``eval()``
    function to provide evaluation results of validation and test sets during training process.

    2. ``compute_score()`` method, which computes the scores for given predictions and labels.
    """

    @abc.abstractmethod
    def evaluate(self, val_pred, test_pred, val_labels, test_labels, total_iters):
        """Evaluate Prediction results on validation and test sets.

        **Classification** and **regression** evaluators should provide both predictions
        and labels of validation and test sets to this method.

        Parameters
        ----------
        val_pred : tensor
            The tensor stores the prediction results on the validation nodes or edges.
        test_pred : tensor
            The tensor stores the prediction results on the test nodes or edges.
        val_labels : tensor
            The tensor stores the labels of the validation nodes or edges.
        test_labels : tensor
            The tensor stores the labels of the test nodes or edges.
        total_iters: int
            The current iteration number.

        Returns
        -----------
        eval_score: dict
            Validation scores of differnet metrics in the format of {metric: val_score}.
        test_score: dict
            Test scores of different metrics in the format of {metric: test_score}.
        """

    @abc.abstractmethod
    def compute_score(self, pred, labels, train=True):
        """ Compute evaluation score of Prediciton results.

        **Classification** and **regression** evaluators should provide both predictions
        and labels to this method.

        Parameters
        ----------
        pred: tensor
            The tensor stores the prediction results.
        labels: tensor
            The tensor stores the labels.
        train: bool
            If in model training.

        Returns
        -------
        dict: Evaluation scores of different metrics in the format of {metric: score}.
        """


class GSgnnLPRankingEvalInterface():
    """ Interface for Link Prediction evaluation functions using ranking methods.

    The interface sets two abstract methods for Link Prediction evaluator classes that use
    ranking method to compute evaluation metrics, such as ``mrr`` (Mean Reciprocal Rank).

    There are two methdos to be implemented if inherite this interface.

    1. ``evaluate()`` method, which will be called by different **Trainer** in their ``eval()``
    function to provide ranking-based evaluation results of validation and test sets during
    training process.

    2. ``compute_score()`` method, which computes the scores for given rankings.
    """

    @abc.abstractmethod
    def evaluate(self, val_rankings, test_rankings, total_iters):
        """Evaluate Link Prediciton results on validation and test sets.

        **Link Prediction** evaluators should provide the ranking of validation and test sets as
        input to this method.

        Parameters
        ----------
        val_rankings: dict of tensors
            The rankings of validation edges for each edge type in the format of {etype: ranking}.
        test_rankings: dict of tensors
            The rankings of testing edges for each edge type in the format of {etype: ranking}.
        total_iters: int
            The current iteration number.

        Returns
        -----------
        eval_score: dict
            Validation score for each edge type in the format of {etype: score}.
        test_score: dict
            Test score for each edge type in the format of {etype: score}.
        """

    @abc.abstractmethod
    def compute_score(self, rankings, train=True):
        """ Compute Link Prediciton evaluation score.

        Ranking-based Link Prediction evaluators should provide ranking values as input
        to this method.

        Parameters
        ----------
        rankings: dict of tensors
            Rankings of positive scores in the format of {etype: ranking}
        train: boolean
            If in model training.

        Returns
        -------
        dict: Ranking-based evaluation scores for each edge type in the format of {etype: score}.
        """


class GSgnnBaseEvaluator():
    """ Base class for GraphStorm Evaluators.

    This class serves as the base for GraphStorm built-in evaluator classes, like
    ``GSgnnClassificationEvaluator``, ``GSgnnRegressionEvaluator``, ``GSgnnMrrLPEvaluator``,
    ``GSgnnPerEtypeMrrLPEvaluator``, and ``GSgnnRconstructFeatRegScoreEvaluator``.

    In order to create customized Evaluators, users can inherite this class and the corresponding
    EvalInteface class, and then implement their two abstract methods, i.e., ``evaluate()``
    and ``compute_score()`` accordingly.

    Parameters
    ----------
    eval_frequency: int
        The frequency (number of iterations) of doing evaluation.
    eval_metric_list: list of string
        Evaluation metrics used for evaluation.
    use_early_stop: bool
        Set true to use early stop.
    early_stop_burnin_rounds: int
        Burn-in rounds (number of evaluations) before starting to check for the early stop
        condition. Default: 0.
    early_stop_rounds: int
        The number of rounds (number of evaluations) for validation scores used to decide early
        stop. Default: 3.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) ``consecutive_increase``, and 2) ``average_increase``.
        Default: ``average_increase``.
    """
    def __init__(self, eval_frequency, eval_metric_list,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        # nodes whose embeddings are used during evaluation
        # if None all nodes are used.
        self._history = []
        self.tracker = None
        self._best_val_score = None
        self._best_test_score = None
        self._best_iter = None
        self.metrics_obj = None # Evaluation metrics obj

        self._metric_list = eval_metric_list
        assert len(self.metric_list) > 0, "At least one metric must be defined, but got 0."
        self._eval_frequency = eval_frequency
        self._do_early_stop = use_early_stop
        if self._do_early_stop:
            self._early_stop_burnin_rounds = early_stop_burnin_rounds
            self._num_early_stop_calls = 0
            self._early_stop_rounds = early_stop_rounds
            self._early_stop_strategy = early_stop_strategy
            self._val_perf_list = []
        # add this list to store all of the performance rank of validation scores for pick top k
        self._val_perf_rank_list = []

    def setup_task_tracker(self, task_tracker):
        """ Setup evaluation task tracker.

        Parameters
        ----------
        task_tracker: GSSageMakerAbc
            A GraphStorm task tracker.
        """
        self.tracker = task_tracker

    def do_eval(self, total_iters, epoch_end=False):
        """ Decide whether to do the evaluation in current iteration or epoch.

        Return `True`, if the current iteration is larger than 0 and is a mutiple of the given
        `eval_frequency`, or is the end of an epoch. Otherwise return `False`.

        Parameters
        ----------
        total_iters: int
            The total number of iterations has been taken.
        epoch_end: bool
            Whether it is the end of an epoch

        Returns
        -------
        bool: Whether to do evaluation.
        """
        if epoch_end:
            return True
        elif self._eval_frequency != 0 and total_iters % self._eval_frequency == 0:
            return True
        return False

    def do_early_stop(self, val_score):
        """ Decide whether to stop the training early.

        Parameters
        ----------
        val_score: dict of list
            Dict of evaluation scores for one metric.

        Returns
        -------
        bool: Whether to stop early.
        """
        if self._do_early_stop is False:
            return False

        assert len(val_score) == 1, \
            f"valudation score should be a signle key value pair but got {val_score}"
        self._num_early_stop_calls += 1
        # Not enough existing validation scores
        if self._num_early_stop_calls <= self._early_stop_burnin_rounds:
            return False

        val_score = list(val_score.values())[0]
        # Not enough validation scores to make early stop decision
        if len(self._val_perf_list) < self._early_stop_rounds:
            self._val_perf_list.append(val_score)
            return False

        # early stop criteria: if the average evaluation value
        # does not improve in the last N evaluation iterations
        if self._early_stop_strategy == EARLY_STOP_AVERAGE_INCREASE_STRATEGY:
            early_stop = early_stop_avg_increase_judge(val_score,
                                                       self._val_perf_list,
                                                       self.get_metric_comparator())
        elif self._early_stop_strategy == EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY:
            early_stop = early_stop_cons_increase_judge(val_score,
                                                        self._val_perf_list,
                                                        self.get_metric_comparator())
        else:
            return False

        self._val_perf_list.pop(0)
        self._val_perf_list.append(val_score)

        return early_stop

    def get_metric_comparator(self):
        """ Return the comparator of the major eval metric.

            We treat the first metric in all evaluation metrics as the major metric, and return
            its corresponding comparator.

            Internal use, and is not released as a public API.
        """
        assert self.metrics_obj is not None, "Evaluation metrics object should not be None."
        metric = self.metric_list[0]
        return self.metrics_obj.metric_comparator[metric]

    def get_val_score_rank(self, val_score):
        """ Get the rank of the given validation score by comparing its value to the
        historical values.

        Parameters
        ----------
        val_score: dict of list
            A dictionary whose key is the metric and the value is a score from evaluator's
            validation computation.

        Returns
        --------
        rank: int
            The rank of the given validation score.
        """
        val_score = list(val_score.values())[0]

        rank = get_val_score_rank(val_score,
                                  self._val_perf_rank_list,
                                  self.get_metric_comparator())
        # after compare, append the score into existing list
        self._val_perf_rank_list.append(val_score)
        return rank

    @property
    def metric_list(self):
        """ Return the evaluation metric list, which is given in class initialization.
        """
        return self._metric_list

    @property
    def best_val_score(self):
        """ Return the best validation score of metrics used in this evaluator in the format
        of {metric: best_val_score}.
        """
        return self._best_val_score

    @property
    def best_test_score(self):
        """ Return the best test score of metrics used in this evaluator in the format
        of {metric: best_test_score}.
        """
        return self._best_test_score

    @property
    def best_iter_num(self):
        """ Return the best iteration number when the best validation score was achieved
        for metrics used in this evaluator in the format of {metric: best_iter_num}.
        """
        return self._best_iter

    @property
    def history(self):
        """ Return a list of evaluation history of training.

        The detailed contents of the list rely on implementations of specific Evaluators.
        For example, ``GSgnnRegressionEvaluator`` and ``GSgnnClassificationEvaluator`` both
        use a tuple of validation and testing score as one list element.
        """
        return self._history

    @property
    def eval_frequency(self):
        """ Return the evaluation frequency, which is given in class initialization.
        """
        return self._eval_frequency

    @property
    def task_tracker(self):
        """ Return the task tracker set from the setup_task_tracker() method.
        """
        return self.tracker

    @property
    def val_perf_rank_list(self):
        """ Return the validation performance rank list.
        """
        return self._val_perf_rank_list


class GSgnnClassificationEvaluator(GSgnnBaseEvaluator, GSgnnPredictionEvalInterface):
    """ Evaluator for classification tasks.

    A built-in evaluator for classification tasks. It uses ``accuracy`` as the default evaluation
    metric.

    This class replaces the ``GSgnnAccEvaluator`` since v0.3.

    Parameters
    ----------
    eval_frequency: int
        The frequency (number of iterations) of doing evaluation.
    eval_metric_list: list of string
        Evaluation metrics used during evaluation. Default: ["accuracy"].
    multilabel: bool
        If set to true, the task is a multi-label classification task. Default: False.
    use_early_stop: bool
        Set true to use early stop. Default: False.
    early_stop_burnin_rounds: int
        Burn-in rounds (number of evaluations) before starting to check for the early stop
        condition. Default: 0.
    early_stop_rounds: int
        The number of rounds (number of evaluations) for validation scores used to decide early
        stop. Default: 3.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) ``consecutive_increase``, and 2) ``average_increase``.
        Default: ``average_increase``.
    """
    def __init__(self, eval_frequency,
                 eval_metric_list=None,
                 multilabel=False,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        # set default metric list
        if eval_metric_list is None:
            eval_metric_list = ["accuracy"]
        super(GSgnnClassificationEvaluator, self).__init__(eval_frequency,
                                                           eval_metric_list,
                                                           use_early_stop,
                                                           early_stop_burnin_rounds,
                                                           early_stop_rounds,
                                                           early_stop_strategy)
        self._multilabel = multilabel
        self._best_val_score = {}
        self._best_test_score = {}
        self._best_iter = {}
        self.metrics_obj = ClassificationMetrics(eval_metric_list,
                                                 multilabel=self._multilabel)

        for metric in self.metric_list:
            self.metrics_obj.assert_supported_metric(metric=metric)
            self._best_val_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_test_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_iter[metric] = 0

    def evaluate(self, val_pred, test_pred, val_labels, test_labels, total_iters):
        """ Compute classificaton metric scores on validation and test sets.

        Parameters
        ----------
        val_pred : tensor
            The tensor stores the prediction results on the validation nodes or edges.
        test_pred : tensor
            The tensor stores the prediction results on the test nodes or edges.
        val_labels : tensor
            The tensor stores the labels of the validation nodes or edges.
        test_labels : tensor
            The tensor stores the labels of the test nodes or edges.
        total_iters: int
            The current iteration number.

        Returns
        -----------
        eval_score: dict
            Validation scores of different classification metrics in the format of
            {metric: val_score}.
        test_score: dict
            Test scores of different classification metrics in the format of {metric: test_score}.
        """
        # exchange preds and labels between runners
        local_rank = get_rank()
        world_size = get_world_size()
        val_pred = broadcast_data(local_rank, world_size, val_pred)
        val_labels = broadcast_data(local_rank, world_size, val_labels)
        test_pred = broadcast_data(local_rank, world_size, test_pred) \
            if test_pred is not None else None
        test_labels = broadcast_data(local_rank, world_size, test_labels) \
            if test_labels is not None else None

        with th.no_grad():
            val_score = self.compute_score(val_pred, val_labels, train=False)
            test_score = self.compute_score(test_pred, test_labels, train=False)

        for metric in self.metric_list:
            # be careful whether > or < it might change per metric.
            if self.metrics_obj.metric_comparator[metric](
                    self._best_val_score[metric], val_score[metric]):
                self._best_val_score[metric] = val_score[metric]
                self._best_test_score[metric] = test_score[metric]
                self._best_iter[metric] = total_iters
        self._history.append((val_score, test_score))

        return val_score, test_score

    def compute_score(self, pred, labels, train=True):
        """ Compute classification evaluation score.

        Parameters
        ----------
        pred: tensor
            The tensor stores the prediction results.
        labels: tensor
            The tensor stores the labels.
        train: bool
            If in model training.

        Returns
        -------
        results: dict
            Evaluation scores of different classification metrics in the format of {metric: score}.
            If either pred or labels are None, the score will be "N/A".
        """
        results = {}
        for metric in self.metric_list:
            if pred is not None and labels is not None:
                if train:
                    # training expects always a single number to be
                    # returned and has a different (potentially) evalution function
                    results[metric] = self.metrics_obj.metric_function[metric](pred, labels)
                else:
                    # validation or testing may have a different
                    # evaluation function, in our case the evaluation code
                    # may return a dictionary with the metric values for each metric
                    results[metric] = self.metrics_obj.metric_eval_function[metric](pred, labels)
            else:
                # if the pred is None or the labels is None the metric can not be computed
                results[metric] = "N/A"
        return results

    @property
    def multilabel(self):
        """ Return if this is a multi-label classification task, which is given in class
        initialization.
        """
        return self._multilabel

class GSgnnRegressionEvaluator(GSgnnBaseEvaluator, GSgnnPredictionEvalInterface):
    """ Evaluator for regression tasks.

    A built-in evaluator for regression tasks. It uses ``rmse`` as the default evaluation metric.

    Parameters
    ----------
    eval_frequency: int
        The frequency (number of iterations) of doing evaluation.
    eval_metric_list: list of string
        Evaluation metric used during evaluation. Default: ["rmse"].
    use_early_stop: bool
        Set true to use early stop. Default: False.
    early_stop_burnin_rounds: int
        Burn-in rounds (number of evaluations) before starting to check for the early stop
        condition. Default: 0.
    early_stop_rounds: int
        The number of rounds (number of evaluations) for validation scores used to decide early
        stop. Default: 3.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) ``consecutive_increase``, and 2) ``average_increase``.
        Default: ``average_increase``.
    """
    def __init__(self, eval_frequency,
                 eval_metric_list=None,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        # set default metric list
        if eval_metric_list is None:
            eval_metric_list = ["rmse"]
        super(GSgnnRegressionEvaluator, self).__init__(eval_frequency,
            eval_metric_list, use_early_stop, early_stop_burnin_rounds,
            early_stop_rounds, early_stop_strategy)
        self._best_val_score = {}
        self._best_test_score = {}
        self._best_iter = {}
        self.metrics_obj = RegressionMetrics()

        for metric in self.metric_list:
            self.metrics_obj.assert_supported_metric(metric=metric)
            self._best_val_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_test_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_iter[metric] = 0

    def evaluate(self, val_pred, test_pred, val_labels, test_labels, total_iters):
        """ Compute regression scores on validation and test sets.

        Parameters
        ----------
        val_pred : tensor
            The tensor stores the prediction results on the validation nodes or edges.
        test_pred : tensor
            The tensor stores the prediction results on the test nodes or edges.
        val_labels : tensor
            The tensor stores the labels of the validation nodes or edges.
        test_labels : tensor
            The tensor stores the labels of the test nodes or edges.
        total_iters: int
            The current iteration number.

        Returns
        -----------
        eval_score: dict
            Validation scores of differnet regression metrics in the format of
            {metric: val_score}.
        test_score: dict
            Test scores of different regression metrics in the format of {metric: test_score}.
        """
        # exchange preds and labels between runners
        local_rank = get_rank()
        world_size = get_world_size()
        val_pred = broadcast_data(local_rank, world_size, val_pred)
        val_labels = broadcast_data(local_rank, world_size, val_labels)
        test_pred = broadcast_data(local_rank, world_size, test_pred) \
            if test_pred is not None else None
        test_labels = broadcast_data(local_rank, world_size, test_labels) \
            if test_labels is not None else None

        with th.no_grad():
            val_score = self.compute_score(val_pred, val_labels)
            test_score = self.compute_score(test_pred, test_labels)

        for metric in self.metric_list:
            # be careful whether > or < it might change per metric.
            if self.metrics_obj.metric_comparator[metric](self._best_val_score[metric],
                                                          val_score[metric]):
                self._best_val_score[metric] = val_score[metric]
                self._best_test_score[metric] = test_score[metric]
                self._best_iter[metric] = total_iters
        self._history.append((val_score, test_score))

        return val_score, test_score

    def compute_score(self, pred, labels, train=True):
        """ Compute regression evaluation score.

        Parameters
        ----------
        pred: tensor
            The tensor stores the prediction results.
        labels: tensor
            The tensor stores the labels.
        train: bool
            If in model training.

        Returns
        -------
        scores: dict
            Evaluation scores of different regression metrics in the format of {metric: score}.
            If either pred or labels are None, the score will be "N/A".
        """
        scores = {}
        for metric in self.metric_list:
            if pred is not None and labels is not None:
                pred = th.squeeze(pred)
                labels = th.squeeze(labels)
                pred = pred.to(th.float32)
                labels = labels.to(th.float32)

                if train:
                    # training expects always a single number to be
                    # returned and has a different (potentially) evluation function
                    scores[metric] = self.metrics_obj.metric_function[metric](pred, labels)
                else:
                    # validation or testing may have a different
                    # evaluation function, in our case the evaluation code
                    # may return a dictionary with the metric values for each metric
                    scores[metric] = self.metrics_obj.metric_eval_function[metric](pred, labels)
            else:
                # if the pred is None or the labels is None the metric can not me computed
                scores[metric] = "N/A"

        return scores

class GSgnnRconstructFeatRegScoreEvaluator(GSgnnRegressionEvaluator):
    """ Evaluator for feature reconstruction tasks using regression scores.

    A built-in evalutor for feature reconstruction tasks. It uses ``mse`` or ``rmse`` as
    evaluation metrics.

    This evaluator requires the prediction results to be a 2D float tensor and
    the label also to be a 2D float tensor, which stores the original features.

    Parameters
    ----------
    eval_frequency: int
        The frequency (number of iterations) of doing evaluation.
    eval_metric_list: list of string
        Evaluation metrics used during evaluation. Default: ["mse"].
    use_early_stop: bool
        Set true to use early stop. Default: False.
    early_stop_burnin_rounds: int
        Burn-in rounds (number of evaluations) before starting to check for the early stop
        condition. Default: 0.
    early_stop_rounds: int
        The number of rounds (number of evaluations) for validation scores used to decide early
        stop. Default: 3.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) ``consecutive_increase``, and 2) ``average_increase``.
        Default: ``average_increase``.
    """
    def __init__(self, eval_frequency,
                 eval_metric_list=None,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        # set default metric list
        if eval_metric_list is None:
            eval_metric_list = ["mse"]

        super(GSgnnRconstructFeatRegScoreEvaluator, self).__init__(
            eval_frequency,
            eval_metric_list,
            use_early_stop,
            early_stop_burnin_rounds,
            early_stop_rounds,
            early_stop_strategy)

    def compute_score(self, pred, labels, train=True):
        """ Compute feature reconstruction evaluation scores.

        Parameters
        ----------
        pred: 2D tensor
            The 2D tensor stores the prediction results.
        labels: 2D tensor
            The 2D tensor stores the labels that are the original node features as this is
            a feature reconstruction task.
        train: bool
            If in model training.

        Returns
        -------
        scores: dict
            Evaluation scores of different feature reconstruction metrics in the format of
            {metric: score}. If either pred or labels are None, the score will be "N/A".
        """
        scores = {}
        for metric in self.metric_list:
            if pred is not None and labels is not None:
                pred = pred.to(th.float32)
                labels = labels.to(th.float32)

                if train:
                    # training expects always a single number to be
                    # returned and has a different (potentially) evluation function
                    scores[metric] = self.metrics_obj.metric_function[metric](pred, labels)
                else:
                    # validation or testing may have a different
                    # evaluation function, in our case the evaluation code
                    # may return a dictionary with the metric values for each metric
                    scores[metric] = self.metrics_obj.metric_eval_function[metric](pred, labels)
            else:
                # if the pred is None or the labels is None the metric can not me computed
                scores[metric] = "N/A"

        return scores

class GSgnnMrrLPEvaluator(GSgnnBaseEvaluator, GSgnnLPRankingEvalInterface):
    """ Evaluator for Link Prediction tasks using ``mrr`` as metric.

    A built-in evaluator for Link Prediction tasks. It uses ``mrr`` as the default eval metric,
    which implements the ``GSgnnLPRankingEvalInterface``.

    To create a customized Link Prediction evaluator that use an evaluation metric other than
    ``mrr``, users might need to 1) define a new evaluation interface if the evaluation method
    requires different input arguments; 2) inherite the new evaluation interface in a
    customized Link Prediction evaluator; 3) define a customized Link Prediction
    Trainer/Inferrer to call the customized Link Prediction evaluator.

    Parameters
    ----------
    eval_frequency: int
        The frequency (number of iterations) of doing evaluation.
    eval_metric_list: list of string
        Evaluation metrics used during evaluation. Default: ["mrr"].
    use_early_stop: bool
        Set true to use early stop. Default: False.
    early_stop_burnin_rounds: int
        Burn-in rounds (number of evaluations) before starting to check for the early stop
        condition. Default: 0.
    early_stop_rounds: int
        The number of rounds (number of evaluations) for validation scores used to decide early
        stop. Default: 3.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) ``consecutive_increase``, and 2) ``average_increase``.
        Default: ``average_increase``.
    """
    def __init__(self, eval_frequency,
                 eval_metric_list=None,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        # set default metric list
        if eval_metric_list is None:
            eval_metric_list = ["mrr"]
        super(GSgnnMrrLPEvaluator, self).__init__(eval_frequency,
            eval_metric_list, use_early_stop, early_stop_burnin_rounds,
            early_stop_rounds, early_stop_strategy)
        self.metrics_obj = LinkPredictionMetrics()

        self._best_val_score = {}
        self._best_test_score = {}
        self._best_iter = {}
        for metric in self.metric_list:
            self._best_val_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_test_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_iter[metric] = 0

    def evaluate(self, val_rankings, test_rankings, total_iters):
        """ ``GSgnnLinkPredictionTrainer`` and ``GSgnnLinkPredictionInferrer`` will call this
        function to compute validation and test ``mrr`` scores.

        Parameters
        ----------
        val_rankings: dict of tensors
            Rankings of positive scores of validation edges for each edge type in the format of
            {etype: ranking}.
        test_rankings: dict of tensors
            Rankings of positive scores of test edges for each edge type in the format of
            {etype: ranking}.
        total_iters: int
            The current iteration number.

        Returns
        -----------
        val_score: dict
            Validation ``mrr`` score in the format of  {"mrr": val_score}. If the ``val_ranking``
            is None, return {"mrr": "N/A"}.
        test_score: dict
            Test ``mrr`` score in the format of {"mrr": test_score}. If the ``test_ranking`` is
            None, return {"mrr": "N/A"}.
        """
        with th.no_grad():
            if test_rankings is not None:
                test_score = self.compute_score(test_rankings)
            else:
                for metric in self.metric_list:
                    test_score = {metric: "N/A"} # Dummy

            if val_rankings is not None:
                val_score = self.compute_score(val_rankings)

                if get_rank() == 0:
                    for metric in self.metric_list:
                        # be careful whether > or < it might change per metric.
                        if self.metrics_obj.metric_comparator[metric](
                            self._best_val_score[metric], val_score[metric]):
                            self._best_val_score[metric] = val_score[metric]
                            self._best_test_score[metric] = test_score[metric]
                            self._best_iter[metric] = total_iters
            else:
                for metric in self.metric_list:
                    val_score = {metric: "N/A"} # Dummy

        self._history.append((val_score, test_score))

        return val_score, test_score

    def compute_score(self, rankings, train=True):
        """ Compute ``mrr`` evaluation score.

        Parameters
        ----------
        rankings: dict of tensors
            Rankings of positive scores in the format of {etype: ranking}
        train: boolean
            If in model training.

        Returns
        -------
        return_metrics: dict
            Evaluation ``mrr`` score of in the format of {"mrr": score}.
        """
        # We calculate global mrr, etype is ignored.
        ranking = []
        for _, rank in rankings.items():
            ranking.append(rank)
        ranking = th.cat(ranking, dim=0)

        # compute ranking value for each metric
        metrics = {}
        for metric in self.metric_list:
            if train:
                # training expects always a single number to be
                # returned and has a different (potentially) evluation function
                metrics[metric] = self.metrics_obj.metric_function[metric](ranking)
            else:
                # validation or testing may have a different
                # evaluation function, in our case the evaluation code
                # may return a dictionary with the metric values for each metric
                metrics[metric] = self.metrics_obj.metric_eval_function[metric](ranking)

        # When world size == 1, we do not need the barrier
        if get_world_size() > 1:
            barrier()
            for _, metric_val in metrics.items():
                th.distributed.all_reduce(metric_val)

        return_metrics = {}
        for metric, metric_val in metrics.items():
            return_metric = metric_val / get_world_size()
            return_metrics[metric] = return_metric.item()

        return return_metrics

class GSgnnPerEtypeMrrLPEvaluator(GSgnnBaseEvaluator, GSgnnLPRankingEvalInterface):
    """ Evaluator for Link Prediction tasks using ``mrr`` as metric,  and
    return per edge type ``mrr`` scores.

    Parameters
    ----------
    eval_frequency: int
        The frequency (number of iterations) of doing evaluation.
    eval_metric_list: list of string
        Evaluation metrics used during evaluation. Default: ["mrr"].
    major_etype: tuple
        A canonical edge type used for selecting the best model. Default: will use the summation
        of ``mrr`` values of all edge types.
    use_early_stop: bool
        Set true to use early stop. Default: False.
    early_stop_burnin_rounds: int
        Burn-in rounds (number of evaluations) before starting to check for the early stop
        condition. Default: 0.
    early_stop_rounds: int
        The number of rounds (number of evaluations) for validation scores used to decide early
        stop. Default: 3.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) ``consecutive_increase``, and 2) ``average_increase``.
        Default: ``average_increase``.
    """
    def __init__(self, eval_frequency,
                 eval_metric_list=None,
                 major_etype = LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        # set default metric list
        if eval_metric_list is None:
            eval_metric_list = ["mrr"]
        super(GSgnnPerEtypeMrrLPEvaluator, self).__init__(eval_frequency,
                                                          eval_metric_list,
                                                          use_early_stop,
                                                          early_stop_burnin_rounds,
                                                          early_stop_rounds,
                                                          early_stop_strategy)
        self.major_etype = major_etype
        self.metrics_obj = LinkPredictionMetrics()

        self._best_val_score = {}
        self._best_test_score = {}
        self._best_iter = {}
        for metric in self.metric_list:
            self._best_val_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_test_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_iter[metric] = 0

    def evaluate(self, val_rankings, test_rankings, total_iters):
        """ ``GSgnnLinkPredictionTrainer`` and ``GSgnnLinkPredictionInferrer`` will call this
        function to compute validation and test ``mrr`` scores.

        Parameters
        ----------
        val_rankings: dict of tensors
            Rankings of positive scores of validation edges for each edge type in the format of
            {etype: ranking}.
        test_rankings: dict of tensors
            Rankings of positive scores of test edges for each edge type in the format of
            {etype: ranking}.
        total_iters: int
            The current iteration number.

        Returns
        -----------
        val_score: dict of dict
            Validation ``mrr`` score in the format of  {"mrr": {etype: val_score}}. If the
            ``val_ranking`` is None, return {"mrr": "N/A"}.
        test_score: dict of dict
            Test ``mrr`` score in the format of {"mrr": {etype: test_score}}. If the
            ``test_ranking`` is None, return {"mrr": "N/A"}.

        """
        with th.no_grad():
            if test_rankings is not None:
                test_score = self.compute_score(test_rankings)
            else:
                for metric in self.metric_list:
                    test_score = {metric: "N/A"} # Dummy

            if val_rankings is not None:
                val_score = self.compute_score(val_rankings)

                if get_rank() == 0:
                    for metric in self.metric_list:
                        # be careful whether > or < it might change per metric.
                        major_val_score = self._get_major_score(val_score[metric])
                        major_test_score = self._get_major_score(test_score[metric])
                        if self.metrics_obj.metric_comparator[metric](
                            self._best_val_score[metric], major_val_score):
                            self._best_val_score[metric] = major_val_score
                            self._best_test_score[metric] = major_test_score
                            self._best_iter[metric] = total_iters
            else:
                for metric in self.metric_list:
                    val_score = {metric: "N/A"} # Dummy

        self._history.append((val_score, test_score))

        return val_score, test_score

    def compute_score(self, rankings, train=True):
        """ Compute per edge type ``mrr`` evaluation score.

        Parameters
        ----------
        rankings: dict of tensors
            Rankings of positive scores in the format of {etype: ranking}.
        train: boolean
            If in model training.

        Returns
        -------
        return_metrics: dict of dict
            Per edge type evaluation ``mrr`` score in the format of {"mrr": {etype: score}}.
        """
        # User can develop its own per etype MRR evaluator
        per_etype_metrics = {}
        for etype, ranking in rankings.items():
            # compute ranking value for each metric
            metrics = {}
            for metric in self.metric_list:
                if train:
                    # training expects always a single number to be
                    # returned and has a different (potentially) evluation function
                    metrics[metric] = self.metrics_obj.metric_function[metric](ranking)
                else:
                    # validation or testing may have a different
                    # evaluation function, in our case the evaluation code
                    # may return a dictionary with the metric values for each metric
                    metrics[metric] = self.metrics_obj.metric_eval_function[metric](ranking)
            per_etype_metrics[etype] = metrics

        # When world size == 1, we do not need the barrier
        if get_world_size() > 1:
            barrier()
            for _, metric in per_etype_metrics.items():
                for _, metric_val in metric.items():
                    th.distributed.all_reduce(metric_val)

        return_metrics = {}
        for etype, metric in per_etype_metrics.items():
            for metric_key, metric_val in metric.items():
                return_metric = metric_val / get_world_size()
                if metric_key not in return_metrics:
                    return_metrics[metric_key] = {}
                return_metrics[metric_key][etype] = return_metric.item()
        return return_metrics

    def _get_major_score(self, score):
        """ Get the score for save best model(s) and early stop
        """
        if isinstance(self.major_etype, str) and \
            self.major_etype == LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL:
            major_score = sum(score.values()) / len(score)
        else:
            major_score = score[self.major_etype]
        return major_score

    def get_val_score_rank(self, val_score):
        """ Get the rank of the validation score of the ``major_etype`` initialized in class
        initialization by comparing its value to the existing historical values. If use
        the default ``major_etype``, will use the summation of validation values of all
        edge types to get the rank.

        Parameters
        ----------
        val_score: dict of dict
            A dict in the format of {"mrr": {etype: score}}.

        Returns
        --------
        rank: int
            The rank of the validation score of the given ``major_etype`` initialized in
            class initialization. If using the default ``major_etype``, the rank will be
            computed based on the summation of validation scores for all edge types.
        """
        val_score = list(val_score.values())[0]
        val_score = self._get_major_score(val_score)

        rank = get_val_score_rank(val_score,
                                  self._val_perf_rank_list,
                                  self.get_metric_comparator())
        # after compare, append the score into existing list
        self._val_perf_rank_list.append(val_score)
        return rank

class GSgnnHitsLPEvaluator(GSgnnBaseEvaluator, GSgnnLPRankingEvalInterface):
    """ Evaluator for Link Prediction tasks using ``hit@k`` as metric.

    A built-in evaluator for Link Prediction tasks. It uses ``hit_at_100`` as the default
    eval metric, which implements the ``GSgnnLPRankingEvalInterface``.

    Parameters
    ----------
    eval_frequency: int
        The frequency (number of iterations) of doing evaluation.
    eval_metric_list: list of string
        Evaluation metric(s) used during evaluation, for example, ["hit_at_10", "hit_at_100"].
        Default: ["hit_at_100"]
    use_early_stop: bool
        Set true to use early stop. Default: False.
    early_stop_burnin_rounds: int
        Burn-in rounds (number of evaluations) before starting to check for the early stop
        condition. Default: 0.
    early_stop_rounds: int
        The number of rounds (number of evaluations) for validation scores used to decide early
        stop. Default: 3.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) ``consecutive_increase``, and 2) ``average_increase``.
        Default: ``average_increase``.
    """
    def __init__(self, eval_frequency,
                 eval_metric_list=None,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        # set default metric list
        if eval_metric_list is None:
            eval_metric_list = [f"{SUPPORTED_HIT_AT_METRICS}_100"]
        super(GSgnnHitsLPEvaluator, self).__init__(eval_frequency,
            eval_metric_list, use_early_stop, early_stop_burnin_rounds,
            early_stop_rounds, early_stop_strategy)
        self.metrics_obj = LinkPredictionMetrics(eval_metric_list)

        self._best_val_score = {}
        self._best_test_score = {}
        self._best_iter = {}
        for metric in self.metric_list:
            self._best_val_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_test_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_iter[metric] = 0

    def evaluate(self, val_rankings, test_rankings, total_iters):
        """ ``GSgnnLinkPredictionTrainer`` and ``GSgnnLinkPredictionInferrer`` will call this
        function to compute validation and test ``hit@k`` scores.

        Parameters
        ----------
        val_rankings: dict of tensors
            Rankings of positive scores of validation edges for each edge type in the format of
            {etype: ranking}.
        test_rankings: dict of tensors
            Rankings of positive scores of test edges for each edge type in the format of
            {etype: ranking}.
        total_iters: int
            The current iteration number.

        Returns
        -----------
        val_score: dict of float
            Validation ``hit@k`` score in the format of  {"hit_at_k": val_score}. If the
            ``val_ranking`` is None, return {"hit_at_k": "N/A"}.
        test_score: dict of float
            Test ``hit@k`` score in the format of {"hit_at_k": test_score}. If the
            ``test_ranking`` is None, return {"hit_at_k": "N/A"}.
        """
        with th.no_grad():
            if test_rankings is not None:
                test_score = self.compute_score(test_rankings)
            else:
                test_score = {}
                for metric in self.metric_list:
                    test_score[metric] = "N/A" # Dummy

            if val_rankings is not None:
                val_score = self.compute_score(val_rankings)

                if get_rank() == 0:
                    for metric in self.metric_list:
                        # be careful whether > or < it might change per metric.
                        if self.metrics_obj.metric_comparator[metric](
                            self._best_val_score[metric], val_score[metric]):
                            self._best_val_score[metric] = val_score[metric]
                            self._best_test_score[metric] = test_score[metric]
                            self._best_iter[metric] = total_iters
            else:
                val_score = {}
                for metric in self.metric_list:
                    val_score[metric] = "N/A" # Dummy

        self._history.append((val_score, test_score))

        return val_score, test_score

    def compute_score(self, rankings, train=True):
        """ Compute ``hit@k`` evaluation score.

        Parameters
        ----------
        rankings: dict of tensors
            Rankings of positive scores in the format of {etype: ranking}
        train: boolean
            If in model training.

        Returns
        -------
        return_metrics: dict of float
            Evaluation ``hit@k`` score of in the format of {"hit_at_k": score}.
        """
        # We calculate global hit@k, etype is ignored.
        ranking = []
        for _, rank in rankings.items():
            ranking.append(rank)
        ranking = th.cat(ranking, dim=0)

        # compute ranking value for each metric
        metrics = {}
        for metric in self.metric_list:
            if train:
                # training expects always a single number to be
                # returned and has a different (potentially) evaluation function
                metrics[metric] = self.metrics_obj.metric_function[metric](ranking)
            else:
                # validation or testing may have a different
                # evaluation function, in our case the evaluation code
                # may return a dictionary with the metric values for each metric
                metrics[metric] = self.metrics_obj.metric_eval_function[metric](ranking)

        # When world size == 1, we do not need the barrier
        if get_world_size() > 1:
            barrier()
            for _, metric_val in metrics.items():
                th.distributed.all_reduce(metric_val)

        return_metrics = {}
        for metric, metric_val in metrics.items():
            return_metric = metric_val / get_world_size()
            return_metrics[metric] = return_metric.item()

        return return_metrics

class GSgnnPerEtypeHitsLPEvaluator(GSgnnBaseEvaluator, GSgnnLPRankingEvalInterface):
    """ Evaluator for Link Prediction tasks using ``hit@k`` as metric,  and
        return per edge type ``hit@k`` scores.

    Parameters
    ----------
    eval_frequency: int
        The frequency (number of iterations) of doing evaluation.
    eval_metric_list: list of string
        Evaluation metric(s) used during evaluation, for example, ["hit_at_10", "hit_at_100"].
        Default: ["hit_at_100"]
    major_etype: tuple
        A canonical edge type used for selecting the best model. Default: will use the summation
        of ``hit@k`` values of all edge types.
    use_early_stop: bool
        Set true to use early stop. Default: False.
    early_stop_burnin_rounds: int
        Burn-in rounds (number of evaluations) before starting to check for the early stop
        condition. Default: 0.
    early_stop_rounds: int
        The number of rounds (number of evaluations) for validation scores used to decide early
        stop. Default: 3.
    early_stop_strategy: str
        1) ``consecutive_increase``, and 2) ``average_increase``.
        Default: ``average_increase``.
    """
    def __init__(self, eval_frequency,
                 eval_metric_list=None,
                 major_etype=LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        # set default metric list
        if eval_metric_list is None:
            eval_metric_list = [f"{SUPPORTED_HIT_AT_METRICS}_100"]
        super(GSgnnPerEtypeHitsLPEvaluator, self).__init__(eval_frequency,
            eval_metric_list, use_early_stop, early_stop_burnin_rounds,
            early_stop_rounds, early_stop_strategy)

        self.major_etype = major_etype
        self.metrics_obj = LinkPredictionMetrics(eval_metric_list)

        self._best_val_score = {}
        self._best_test_score = {}
        self._best_iter = {}
        for metric in self.metric_list:
            self._best_val_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_test_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_iter[metric] = 0

    def evaluate(self, val_rankings, test_rankings, total_iters):
        """ ``GSgnnLinkPredictionTrainer`` and ``GSgnnLinkPredictionInferrer`` will call this
        function to compute validation and test ``hit@k`` scores.

        Parameters
        ----------
        val_rankings: dict of tensors
            Rankings of positive scores of validation edges for each edge type in the format of
            {etype: ranking}.
        test_rankings: dict of tensors
            Rankings of positive scores of test edges for each edge type in the format of
            {etype: ranking}.
        total_iters: int
            The current iteration number.

        Returns
        -----------
        val_score: dict of dict of float
            Validation ``hit@k`` score in the format of  {"hit_at_k": {etype: val_score}}. If the
            ``val_ranking`` is None, return {"hit_at_k": "N/A"}.
        test_score: dict of dict of float
            Test ``hit@k`` score in the format of {"hit_at_k": {etype: test_score}}. If the
            ``test_ranking`` is None, return {"hit_at_k": "N/A"}.
        """
        with th.no_grad():
            if test_rankings is not None:
                test_score = self.compute_score(test_rankings)
            else:
                test_score = {}
                for metric in self.metric_list:
                    test_score[metric] = "N/A" # Dummy

            if val_rankings is not None:
                val_score = self.compute_score(val_rankings)

                if get_rank() == 0:
                    for metric in self.metric_list:
                        # be careful whether > or < it might change per metric.
                        major_val_score = self._get_major_score(val_score[metric])
                        major_test_score = self._get_major_score(test_score[metric])
                        if self.metrics_obj.metric_comparator[metric](
                            self._best_val_score[metric], major_val_score):
                            self._best_val_score[metric] = major_val_score
                            self._best_test_score[metric] = major_test_score
                            self._best_iter[metric] = total_iters
            else:
                val_score = {}
                for metric in self.metric_list:
                    val_score[metric] = "N/A" # Dummy

        self._history.append((val_score, test_score))

        return val_score, test_score

    def _get_major_score(self, score):
        """ Get the score for save best model(s) and early stop
        """
        if isinstance(self.major_etype, str) and \
            self.major_etype == LINK_PREDICTION_MAJOR_EVAL_ETYPE_ALL:
            major_score = sum(score.values()) / len(score)
        else:
            major_score = score[self.major_etype]
        return major_score

    def get_val_score_rank(self, val_score):
        """ Get the rank of the validation score of the ``major_etype`` initialized in class
        initialization by comparing its value to the existing historical values. If using
        the default ``major_etype``, it will compute the rank as the summation of validation
        values of all edge types.

        Parameters
        ----------
        val_score: dict of dict
            A dict in the format of {"hit_at_k": {etype: score}}.

        Returns
        --------
        rank: int
            The rank of the validation score of the given ``major_etype`` initialized in
            class initialization. If using the default ``major_etype``, the rank will be
            computed based on the summation of validation scores for all edge types.
        """
        val_score = list(val_score.values())[0]
        val_score = self._get_major_score(val_score)

        rank = get_val_score_rank(val_score,
                                  self._val_perf_rank_list,
                                  self.get_metric_comparator())
        # after compare, append the score into existing list
        self._val_perf_rank_list.append(val_score)
        return rank

    def compute_score(self, rankings, train=True):
        """ Compute per edge type ``hit@k`` evaluation score.

        Parameters
        ----------
        rankings: dict of tensors
            Rankings of positive scores in the format of {etype: ranking}.
        train: boolean
            If in model training.

        Returns
        -------
        return_metrics: dict of dict of float
            Per edge type evaluation ``hit@k`` score in the format of {"hit_at_k": {etype: score}}.
        """
        # We calculate per etype hit@k
        per_etype_metrics = {}
        for etype, ranking in rankings.items():
            # compute ranking value for each metric
            metrics = {}
            for metric in self.metric_list:
                if train:
                    # training expects always a single number to be
                    # returned and has a different (potentially) evaluation function
                    metrics[metric] = self.metrics_obj.metric_function[metric](ranking)
                else:
                    # validation or testing may have a different
                    # evaluation function, in our case the evaluation code
                    # may return a dictionary with the metric values for each metric
                    metrics[metric] = self.metrics_obj.metric_eval_function[metric](ranking)
            per_etype_metrics[etype] = metrics

        # When world size == 1, we do not need the barrier
        if get_world_size() > 1:
            barrier()
            for _, metric in per_etype_metrics.items():
                for _, metric_val in metric.items():
                    th.distributed.all_reduce(metric_val)

        return_metrics = {}
        for etype, metric in per_etype_metrics.items():
            for metric_key, metric_val in metric.items():
                return_metric = metric_val / get_world_size()
                if metric_key not in return_metrics:
                    return_metrics[metric_key] = {}
                return_metrics[metric_key][etype] = return_metric.item()
        return return_metrics


class GSgnnMultiTaskEvalInterface():
    """ Interface for multi-task evaluation

    The interface set one abstract method
    """
    @abc.abstractmethod
    def evaluate(self, val_results, test_results, total_iters):
        """Evaluate validation and test sets for Prediciton tasks

            GSgnnTrainers will call this function to do evalution in their eval() fuction.

        Parameters
        ----------
        val_results: dict
            Validation results in a format of {task_id: validation results}
        test_results: dict
            Testing results in a format of {task_id: test results}
        total_iters: int
            The current interation number.

        Returns
        -----------
        val_scores: dict
            Validation scores in a format of {task_id: scores}
        test_scores: dict
            Test scores in a format of {task_id: scores}
        """

class GSgnnMultiTaskEvaluator(GSgnnBaseEvaluator, GSgnnMultiTaskEvalInterface):
    """ Multi-task evaluator

    Parameters
    ----------
    eval_frequency: int
        The frequency (number of iterations) of doing evaluation.
    task_evaluators: dict
        Specific evaluators for different tasks. In a format of {task_id:GSgnnBaseEvaluator}
    use_early_stop: bool
        Set true to use early stop.
        Note(xiang): Early stop not implemented. Reserved for future.
    early_stop_burnin_rounds: int
        Burn-in rounds before start checking for the early stop condition.
        Note(xiang): Early stop not implemented. Reserved for future.
    early_stop_rounds: int
        The number of rounds for validation scores used to decide early stop.
        Note(xiang): Early stop not implemented. Reserved for future.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) consecutive_increase and 2) average_increase.
        Note(xiang): Early stop not implemented. Reserved for future.
    """
    # pylint: disable=unused-argument
    # pylint: disable=super-init-not-called
    def __init__(self, eval_frequency, task_evaluators,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        # nodes whose embeddings are used during evaluation
        # if None all nodes are used.
        self._history = []
        self.tracker = None
        self._best_val_score = None
        self._best_test_score = None
        self._best_iter = None

        self._task_evaluators = task_evaluators
        assert len(self.task_evaluators) > 1, \
            "There must be multiple evaluators for different tasks." \
            f"But only get {len(self.task_evaluators)}"

        self._metric_list = {
            task_id: evaluator.metric_list for task_id, evaluator in self.task_evaluators.items()
        }

        self._eval_frequency = eval_frequency
        # TODO(xiang): Support early stop
        assert use_early_stop is False, \
            "GSgnnMultiTaskEvaluator does not support early stop now."
        self._do_early_stop = use_early_stop

        # add this list to store all of the performance rank of validation scores for pick top k
        self._val_perf_rank_list = []


    # pylint: disable=unused-argument
    def do_early_stop(self, val_score):
        """ Decide whether to stop the training

        Note: do not support early stop for multi-task learning.
        Will support it later.

        Parameters
        ----------
        val_score: float
            Evaluation score
        """
        raise RuntimeError("GSgnnMultiTaskEvaluator.do_early_stop is not implemented")

    def get_metric_comparator(self):
        """ Return the comparator of the major eval metric.

            Note: not support now.

        """
        raise RuntimeError("GSgnnMultiTaskEvaluator.get_metric_comparator is not implemented")

    # pylint: disable=unused-argument
    def get_val_score_rank(self, val_score):
        """
        Get the rank of the given validation score by comparing its values to the existing value
        list.

        Note: not support now.

        Parameters
        ----------
        val_score: dict
            A dictionary whose key is the metric and the value is a score from evaluator's
            validation computation.
        """
        raise RuntimeError("GSgnnMultiTaskEvaluator.get_val_score_rank is not implemented")

    @property
    def task_evaluators(self):
        """ Task evaluators
        """
        return self._task_evaluators

    @property
    def best_val_score(self):
        """ Best validation score
        """
        best_val_score = {
            task_id: evaluator.best_val_score \
                for task_id, evaluator in self.task_evaluators.items()
        }
        return best_val_score

    @property
    def best_test_score(self):
        """ Best test score
        """
        best_test_score = {
            task_id: evaluator.best_test_score \
                for task_id, evaluator in self.task_evaluators.items()
        }
        return best_test_score

    @property
    def best_iter_num(self):
        """ Best iteration number
        """
        best_iter_num = {
            task_id: evaluator.best_iter_num \
                for task_id, evaluator in self.task_evaluators.items()
        }
        return best_iter_num

    @property
    def val_perf_rank_list(self):
        raise RuntimeError("GSgnnMultiTaskEvaluator.val_perf_rank_list not supported")

    def evaluate(self, val_results, test_results, total_iters):
        eval_tasks = {}
        val_scores = {}
        test_scores = {}

        if val_results is not None:
            for task_id, val_result in val_results.items():
                eval_tasks[task_id] = [val_result]

        if test_results is not None:
            for task_id, test_result in test_results.items():
                if task_id in eval_tasks:
                    eval_tasks[task_id].append(test_result)
                else:
                    eval_tasks[task_id] = [None, test_result]

        for task_id, eval_task in eval_tasks.items():
            if len(eval_task) == 1:
                # only has validation result
                eval_task.append(None)
            assert len(eval_task) == 2, \
                "An evaluation task is composed of two parts: " \
                f"validation and test, but get {len(eval_task)} parts"
            assert task_id in self._task_evaluators, \
                f"The evaluator of {task_id} is not defined."
            task_evaluator = self._task_evaluators[task_id]

            if isinstance(task_evaluator, GSgnnPredictionEvalInterface):
                val_preds, val_labels = eval_task[0] \
                    if eval_task[0] is not None else (None, None)
                test_preds, test_labels = eval_task[1] \
                    if eval_task[0] is not None else (None, None)
                val_score, test_score = task_evaluator.evaluate(
                    val_preds, test_preds, val_labels, test_labels, total_iters)
            elif isinstance(task_evaluator, GSgnnLPRankingEvalInterface):
                val_rankings = eval_task[0]
                test_rankings = eval_task[1]
                val_score, test_score = task_evaluator.evaluate(
                    val_rankings, test_rankings, total_iters)
            else:
                raise TypeError("Unknown evaluator")

            val_scores[task_id] = val_score
            test_scores[task_id] = test_score

        self._history.append((val_scores, test_scores))

        return val_scores, test_scores
