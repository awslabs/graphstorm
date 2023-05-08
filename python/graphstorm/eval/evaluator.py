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

from .eval_func import ClassificationMetrics, RegressionMetrics, LinkPredictionMetrics
from .utils import broadcast_data
from ..config.config import EARLY_STOP_AVERAGE_INCREASE_STRATEGY
from ..config.config import EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY
from ..utils import get_rank
from .utils import gen_mrr_score

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
    list [1., 1., 2., 2., 3., 4.], the given value 2. will be ranked to the 5th highest score.

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


# TODO(xiangsx): combine GSgnnInstanceEvaluator and GSgnnLPEvaluator
class GSgnnInstanceEvaluator():
    """ Template class for user defined evaluator.

    Parameters
    ----------
    eval_frequency: int
        The frequency (# of iterations) of doing evaluation.
    eval_metric: list of string
        Evaluation metric used during evaluation.
    use_early_stop: bool
        Set true to use early stop.
    early_stop_burnin_rounds: int
        Burn-in rounds before start checking for the early stop condition.
    early_stop_rounds: int
        The number of rounds for validation scores used to decide early stop.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) consecutive_increase and 2) average_increase.
    """
    def __init__(self, eval_frequency, eval_metric,
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

        self._metric = eval_metric
        assert len(self.metric) > 0, \
            "At least one metric must be defined"
        self.eval_frequency = eval_frequency
        self._do_early_stop = use_early_stop
        if self._do_early_stop:
            self._early_stop_burnin_rounds = early_stop_burnin_rounds
            self._num_early_stop_calls = 0
            self._early_stop_rounds = early_stop_rounds
            self._early_stop_strategy = early_stop_strategy
            self._val_perf_list = []
        # add this list to store
        self._val_perf_rank_list = []

    def setup_task_tracker(self, task_tracker):
        """ Setup evaluation tracker

            Parameters
            ----------
            client:
                tracker client
        """
        self.tracker = task_tracker

    @abc.abstractmethod
    def evaluate(self, val_pred, test_pred, val_labels, test_labels, total_iters):
        """
        GSgnnLinkPredictionModel.fit() will call this function to do user defined evalution.

        Parameters
        ----------
        val_pred : tensor
            The tensor stores the prediction results on the validation nodes.
        test_pred : tensor
            The tensor stores the prediction results on the test nodes.
        val_labels : tensor
            The tensor stores the labels of the validation nodes.
        test_labels : tensor
            The tensor stores the labels of the test nodes.
        total_iters: int
            The current interation number.

        Returns
        -----------
        eval_score: float
            Validation score
        test_score: float
            Test score
        """

    def do_eval(self, total_iters, epoch_end=False):
        """ Decide whether to do the evaluation in current iteration or epoch

            Parameters
            ----------
            total_iters: int
                The total number of iterations has been taken.
            epoch_end: bool
                Whether it is the end of an epoch

            Returns
            -------
            Whether do evaluation: bool
        """
        if epoch_end:
            return True
        elif self.eval_frequency != 0 and total_iters % self.eval_frequency == 0:
            return True
        return False


    @abc.abstractmethod
    def compute_score(self, pred, labels):
        """ Compute evaluation score

            Parameters
            ----------
            pred:
                Rediction result
            labels:
                Label
        """

    def print_history(self):
        """ Print history eval info
        """
        for val_score, test_score in self._history:
            print(f"val {self.metric}: {val_score:.3f}, test {self.metric}: {test_score:.3f}")

    def do_early_stop(self, val_score):
        """ Decide whether to stop the training

            Parameters
            ----------
            val_score: float
                Evaluation score
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
                self._val_perf_list, self.get_metric_comparator())
        elif self._early_stop_strategy == EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY:
            early_stop = early_stop_cons_increase_judge(val_score,
                self._val_perf_list, self.get_metric_comparator())
        else:
            return False

        self._val_perf_list.pop(0)
        self._val_perf_list.append(val_score)

        return early_stop

    def get_metric_comparator(self):
        """ Return the comparator of the major eval metric.

            We treat the first metric in all evaluation metrics as the major metric.
        """
        assert self.metrics_obj is not None, \
            "Evaluation metrics object should not be None"
        metric = self.metric[0]
        return self.metrics_obj.metric_comparator[metric]

    def get_val_score_rank(self, val_score):
        """
        Get the rank of the given validation score by comparing its values to the existing value
        list.

        Parameters
        ----------
        val_score: dict
            A dictionary whose key is the metric and the value is a score from evaluator's
            validation computation.
        """
        val_score = list(val_score.values())[0]

        rank = get_val_score_rank(val_score,
                                  self._val_perf_rank_list,
                                  self.get_metric_comparator())
        # after compare, append the score into existing list
        self._val_perf_rank_list.append(val_score)
        return rank

    @property
    def metric(self):
        """ evaluation metrics
        """
        return self._metric

    @property
    def best_val_score(self):
        """ Best validation score
        """
        return self._best_val_score

    @property
    def best_test_score(self):
        """ Best test score
        """
        return self._best_test_score

    @property
    def best_iter_num(self):
        """ Best iteration number
        """
        return self._best_iter

class GSgnnRegressionEvaluator(GSgnnInstanceEvaluator):
    """ The class for user defined evaluator.

    Parameters
    ----------
    eval_frequency: int
        The frequency (# of iterations) of doing evaluation.
    eval_metric: list of string
        Evaluation metric used during evaluation.
    use_early_stop: bool
        Set true to use early stop.
    early_stop_burnin_rounds: int
        Burn-in rounds before start checking for the early stop condition.
    early_stop_rounds: int
        The number of rounds for validation scores used to decide early stop.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) consecutive_increase and 2) average_increase.
    """
    def __init__(self, eval_frequency,
                 eval_metric,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        super(GSgnnRegressionEvaluator, self).__init__(eval_frequency,
            eval_metric, use_early_stop, early_stop_burnin_rounds,
            early_stop_rounds, early_stop_strategy)
        self._best_val_score = {}
        self._best_test_score = {}
        self._best_iter = {}
        self.metrics_obj = RegressionMetrics()

        for metric in self.metric:
            self.metrics_obj.assert_supported_metric(metric=metric)
            self._best_val_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_test_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_iter[metric] = 0

    def evaluate(self, val_pred, test_pred, val_labels, test_labels, total_iters):
        """ Compute scores on validation and test predictions.

            Parameters
            ----------
            val_pred : tensor
                The tensor stores the prediction results on the validation nodes.
            test_pred : tensor
                The tensor stores the prediction results on the test nodes.
            val_labels : tensor
                The tensor stores the labels of the validation nodes.
            test_labels : tensor
                The tensor stores the labels of the test nodes.
            total_iters: int
                The current interation number.
            Returns
            -----------
            float
                Validation MSE
            float
                Test MSE
        """
        # exchange preds and labels between runners
        local_rank = th.distributed.get_rank()
        world_size = th.distributed.get_world_size()
        val_pred = broadcast_data(local_rank, world_size, val_pred)
        val_labels = broadcast_data(local_rank, world_size, val_labels)
        test_pred = broadcast_data(local_rank, world_size, test_pred)
        test_labels = broadcast_data(local_rank, world_size, test_labels)

        with th.no_grad():
            val_score = self.compute_score(val_pred, val_labels)
            test_score = self.compute_score(test_pred, test_labels)

        for metric in self.metric:
            # be careful whether > or < it might change per metric.
            if self.metrics_obj.metric_comparator[metric](self._best_val_score[metric],
                                                          val_score[metric]):
                self._best_val_score[metric] = val_score[metric]
                self._best_test_score[metric] = test_score[metric]
                self._best_iter[metric] = total_iters
        self._history.append((val_score, test_score))

        return val_score, test_score

    def compute_score(self, pred, labels):
        """ Compute evaluation score

            Parameters
            ----------
            pred:
                Rediction result
            labels:
                Label

            Returns
            -------
            Evaluation metric values: dict
        """
        scores = {}
        pred = th.squeeze(pred)
        labels = th.squeeze(labels)
        for metric in self.metric:
            scores[metric] = self.metrics_obj.metric_function[metric](pred, labels) \
                    if pred is not None and labels is not None else -1
        return scores

class GSgnnAccEvaluator(GSgnnInstanceEvaluator):
    """ The class for user defined evaluator.

    Parameters
    ----------
    eval_frequency: int
        The frequency (# of iterations) of doing evaluation.
    eval_metric: list of string
        Evaluation metric used during evaluation.
    multilabel: bool
        If set to true, the task is a multi-label classification task.
    use_early_stop: bool
        Set true to use early stop.
    early_stop_burnin_rounds: int
        Burn-in rounds before start checking for the early stop condition.
    early_stop_rounds: int
        The number of rounds for validation scores used to decide early stop.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) consecutive_increase and 2) average_increase.
    """
    def __init__(self, eval_frequency,
                 eval_metric, multilabel,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY): # pylint: disable=unused-argument
        super(GSgnnAccEvaluator, self).__init__(eval_frequency,
            eval_metric, use_early_stop, early_stop_burnin_rounds,
            early_stop_rounds, early_stop_strategy)
        self.multilabel = multilabel
        self._best_val_score = {}
        self._best_test_score = {}
        self._best_iter = {}
        self.metrics_obj = ClassificationMetrics(multilabel=self.multilabel)

        for metric in self.metric:
            self.metrics_obj.assert_supported_metric(metric=metric)
            self._best_val_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_test_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_iter[metric] = 0

    def evaluate(self, val_pred, test_pred, val_labels, test_labels, total_iters):
        """ Compute scores on validation and test predictions.

            Parameters
            ----------
            val_pred : tensor
                The tensor stores the prediction results on the validation nodes.
            test_pred : tensor
                The tensor stores the prediction results on the test nodes.
            val_labels : tensor
                The tensor stores the labels of the validation nodes.
            test_labels : tensor
                The tensor stores the labels of the test nodes.
            total_iters: int
                The current interation number.
            Returns
            -----------
            float
                Validation Score
            float
                Test Score
        """
        # exchange preds and labels between runners
        local_rank = th.distributed.get_rank()
        world_size = th.distributed.get_world_size()
        val_pred = broadcast_data(local_rank, world_size, val_pred)
        val_labels = broadcast_data(local_rank, world_size, val_labels)
        test_pred = broadcast_data(local_rank, world_size, test_pred)
        test_labels = broadcast_data(local_rank, world_size, test_labels)

        with th.no_grad():
            val_score = self.compute_score(val_pred, val_labels, train=False)
            test_score = self.compute_score(test_pred, test_labels, train=False)

        for metric in self.metric:
            # be careful whether > or < it might change per metric.
            if self.metrics_obj.metric_comparator[metric](
                self._best_val_score[metric],val_score[metric]):
                self._best_val_score[metric] = val_score[metric]
                self._best_test_score[metric] = test_score[metric]
                self._best_iter[metric] = total_iters
        self._history.append((val_score, test_score))

        return val_score, test_score

    def compute_score(self, pred, labels, train=True):
        """ Compute evaluation score

            Parameters
            ----------
            pred:
                Rediction result
            labels:
                Label

            Returns
            -------
            Evaluation metric values: dict
        """
        results = {}
        for metric in self.metric:
            if pred is not None and labels is not None:
                if train:
                    # training expects always a single number to be
                    # returned and has a different (potentially) function
                    results[metric] = self.metrics_obj.metric_function[metric](pred, labels)
                else:
                    # validation or testing may have a different
                    # evaluation function, in our case the evaluation code
                    # may return a dictionary with the metric values for each label
                    results[metric] = self.metrics_obj.metric_eval_function[metric](pred, labels)
            else:
                # if the pred is None or the labels is None the metric can not me computed
                results[metric] = -1
        return results

class GSgnnLPEvaluator():
    """ Template class for user defined evaluator.

    Parameters
    ----------
    eval_frequency: int
        The frequency (# of iterations) of doing evaluation.
    eval_metric: list of string
        Evaluation metric used during evaluation.
    use_early_stop: bool
        Set true to use early stop.
    early_stop_burnin_rounds: int
        Burn-in rounds before start checking for the early stop condition.
    early_stop_rounds: int
        The number of rounds for validation scores used to decide early stop.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) consecutive_increase and 2) average_increase.
    """
    def __init__(self, eval_frequency, eval_metric,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        # nodes whose embeddings are used during evaluation
        # if None all nodes are used.
        self._target_nidx = None
        self.tracker = None
        self._best_val_score = None
        self._best_test_score = None
        self._best_iter = None
        self.metrics_obj = None # Evaluation metrics obj

        self._metric = eval_metric
        assert len(self.metric) > 0, "At least one metric must be defined"
        self.eval_frequency = eval_frequency
        self._do_early_stop = use_early_stop
        if self._do_early_stop:
            self._early_stop_burnin_rounds = early_stop_burnin_rounds
            self._num_early_stop_calls = 0
            self._early_stop_rounds = early_stop_rounds
            self._early_stop_strategy = early_stop_strategy
            self._val_perf_list = []
        # add this list to store all of the performance rank of validation scores for pick top k
        self._val_perf_rank_list = []

    def setup_task_tracker(self, client):
        """ Setup evaluation tracker

            Parameters
            ----------
            client:
                tracker client
        """
        self.tracker = client

    @abc.abstractmethod
    def evaluate(self, val_scores, test_scores, total_iters):
        """
        GSgnnLinkPredictionModel.fit() will call this function to do user defined evalution.

        Note: Make sure each trainer will get the same validation scores.
        The early stop and model saving progress rely on certain scores.

        Parameters
        ----------
        val_scores: dict of tensors
            The rankings of validation edges for each edge type.
        test_scores: dict of tensors
            The rankings of testing edges for each edge type.
        total_iters: int
            The current interation number.

        Returns
        -----------
        eval_score: float
            Validation score
        test_score: float
            Test score
        """

    def do_eval(self, total_iters, epoch_end=False):
        """ Decide whether to do the evaluation in current iteration or epoch

            Parameters
            ----------
            epoch: int
                The epoch number
            total_iters: int
                The total number of iterations has been taken.
            epoch_end: bool
                Whether it is the end of an epoch

            Returns
            -------
            Whether do evaluation: bool
        """
        if epoch_end:
            return True
        elif self.eval_frequency != 0 and \
            total_iters % self.eval_frequency == 0:
            return True
        return False

    def do_early_stop(self, val_score):
        """ Decide whether to stop the training

            Parameters
            ----------
            val_score: float
                Evaluation score
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
                self._val_perf_list, self.get_metric_comparator())
        elif self._early_stop_strategy == EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY:
            early_stop = early_stop_cons_increase_judge(val_score,
                self._val_perf_list, self.get_metric_comparator())

        self._val_perf_list.pop(0)
        self._val_perf_list.append(val_score)

        return early_stop

    def get_metric_comparator(self):
        """ Return the comparator of the major eval metric.

            We treat the first metric in all evaluation metrics as the major metric.
        """

        assert self.metrics_obj is not None, \
            "Evaluation metrics object should not be None"
        metric = self.metric[0]
        return self.metrics_obj.metric_comparator[metric]

    def get_val_score_rank(self, val_score):
        """
        Get the rank of the given val score by comparing its values to the existing value list.

        Parameters
        ----------
        val_score: dict
            A dictionary whose key is the metric and the value is a score from evaluator's
            validation computation.
        """
        val_score = list(val_score.values())[0]

        rank = get_val_score_rank(val_score,
                                  self._val_perf_rank_list,
                                  self.get_metric_comparator())
        # after compare, append the score into existing list
        self._val_perf_rank_list.append(val_score)
        return rank

    @property
    def target_nidx(self):
        """ target_nidx
        """
        return self._target_nidx

    @property
    def metric(self):
        """ evaluation metrics
        """
        return self._metric

    @property
    def best_val_score(self):
        """ Best validation score
        """
        return self._best_val_score

    @property
    def best_test_score(self):
        """ Best test score
        """
        return self._best_test_score

    @property
    def best_iter_num(self):
        """ Best iteration number
        """
        return self._best_iter

    @property
    def val_perf_rank_list(self):
        """ validation performance rank list
        """
        return self._val_perf_rank_list


class GSgnnMrrLPEvaluator(GSgnnLPEvaluator):
    """ The class for user defined evaluator.

    Parameters
    ----------
    eval_frequency: int
        The frequency (# of iterations) of doing evaluation.
    data: GSgnnEdgeData
        The processed dataset
    num_negative_edges_eval: int
        Number of negative edges sampled for each positive edge in evalation.
    lp_decoder_type: str
        Link prediction decoder type.
    use_early_stop: bool
        Set true to use early stop.
    early_stop_burnin_rounds: int
        Burn-in rounds before start checking for the early stop condition.
    early_stop_rounds: int
        The number of rounds for validation scores used to decide early stop.
    early_stop_strategy: str
        The early stop strategy. GraphStorm supports two strategies:
        1) consecutive_increase and 2) average_increase.
    """
    def __init__(self, eval_frequency, data,
                 num_negative_edges_eval, lp_decoder_type,
                 use_early_stop=False,
                 early_stop_burnin_rounds=0,
                 early_stop_rounds=3,
                 early_stop_strategy=EARLY_STOP_AVERAGE_INCREASE_STRATEGY):
        eval_metric = ["mrr"]
        super(GSgnnMrrLPEvaluator, self).__init__(eval_frequency,
            eval_metric, use_early_stop, early_stop_burnin_rounds,
            early_stop_rounds, early_stop_strategy)
        self.train_idxs = data.train_idxs
        self.val_idxs = data.val_idxs
        self.test_idxs = data.test_idxs
        self.num_negative_edges_eval = num_negative_edges_eval
        self.lp_decoder_type = lp_decoder_type

        self.metrics_obj = LinkPredictionMetrics()

        self._best_val_score = {}
        self._best_test_score = {}
        self._best_iter = {}
        for metric in self.metric:
            self._best_val_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_test_score[metric] = self.metrics_obj.init_best_metric(metric=metric)
            self._best_iter[metric] = 0

    def compute_score(self, rankings, train=False): # pylint:disable=unused-argument
        """ Compute evaluation score

            Parameters
            ----------
            rankings: dict of tensors
                Rankings of positive scores in format of {etype: ranking}
            train: bool
                TODO: Reversed for future use cases when we want to use different
                way to generate scores for train (more efficient but less accurate)
                and test.

            Returns
            -------
            Evaluation metric values: dict
        """
        # We calculate global mrr, etype is ignored.
        # User can develop its own per etype MRR evaluator
        ranking = []
        for _, rank in rankings.items():
            ranking.append(rank)
        ranking = th.cat(ranking, dim=0)

        metrics = gen_mrr_score(ranking)

        # When world size == 1, we do not need the barrier
        if th.distributed.get_world_size() > 1:
            th.distributed.barrier()
        for _, metric_val in metrics.items():
            th.distributed.all_reduce(metric_val)
        return_metrics = {}
        for metric, metric_val in metrics.items():
            return_metric = \
                metric_val / th.distributed.get_world_size()
            return_metrics[metric] = return_metric.item()
        return return_metrics

    def evaluate(self, val_scores, test_scores, total_iters):
        """ GSgnnLinkPredictionModel.fit() will call this function to do user defined evalution.

        Parameters
        ----------
        val_scores: dict of tensors
            Rankings of positive scores of validation edges for each edge type.
        test_scores: dict of tensors
            Rankings of positive scores of test edges for each edge type..
        total_iters: int
            The current interation number.

        Returns
        -----------
        val_mrr: float
            Validation mrr
        test_mrr: float
            Test mrr
        """
        with th.no_grad():
            test_score = self.compute_score(test_scores)

            if val_scores is not None:
                val_score = self.compute_score(val_scores)

                if get_rank() == 0:
                    for metric in self.metric:
                        # be careful whether > or < it might change per metric.
                        if self.metrics_obj.metric_comparator[metric](
                            self._best_val_score[metric], val_score[metric]):
                            self._best_val_score[metric] = val_score[metric]
                            self._best_test_score[metric] = test_score[metric]
                            self._best_iter[metric] = total_iters
            else:
                val_score = {"mrr": -1} # Dummy

        return val_score, test_score
