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

    GraphStorm task tracker
"""
import abc

class GSTaskTrackerAbc():
    """ Template class for user defined task tracker, i.e., logging system.

        Parameters
        ----------
        config: GSConfig
            Configurations. Users can add their own configures in the yaml config file.
        rank: int
            Task rank
    """
    def __init__(self, config, rank):
        self._rank = rank
        self._report_frequency = config.log_report_frequency # Can be None if not provided

    @abc.abstractmethod
    def log_metric(self, metric_name, metric_value, step, force_report=False):
        """ log validation or test metric

        Parameters
        ----------
        metric_name: str
            Validation or test metric name
        metric_value:
            Validation or test metric value
        step: int
            The corresponding step/iteration in the training loop.
        force_report: bool
            If true, report the metric
        """

    @abc.abstractmethod
    def log_iter_metrics(self, eval_metrics, val_score, test_score,
        best_val_score, best_test_score, best_iter_num, train_score=None,
        eval_time=-1, total_steps=1):
        """ log evaluation metrics for a specific iteration.

        Parameters
        ----------
        eval_metrics : list of str
            The evaluation metrics.
        val_score: dict
            Validation score
        test_score: dict
            Test score
        best_val_score: dict
            Best validation score
        best_test_score: dict
            Best test score corresponding to the best_val_score
        best_iter_num: dict
            The iteration number corresponding to the best_val_score
        train_score: dict
            Training score
        eval_time:
            Total evaluation time
        total_steps: int
            The corresponding step/iteration
        """

    def keep_alive(self, report_step):
        """ Dummy log, send keep alive message to tracker server

        Parameters
        ----------
        step: int
            Current exec step. Used to decide whether send dummy info
        """
        # Do nothing

    @abc.abstractmethod
    def log_param(self, param_name, param_value):
        """ Log parameters

        Parameters
        ----------
        param_name: str
            Parameter name
        param_value:
            Parameter value
        """

    def log_params(self, param_value):
        """ Log parameters

        Parameters
        ----------
        param_value: dict
            A dictionary of <name, value> pairs storing parameters
        """
        for name, value in param_value.items():
            self.log_param(name, value)

    def log_mean_forward_time(self, forward_time):
        """ Log average forward time

        Parameters
        ----------
        forward_time: float
            Average forward time
        """
        # By default do nothing

    def log_mean_backward_time(self, backward_time):
        """ Log average backward time

        Parameters
        ----------
        backward_time: float
            Average backward time
        """
        # By default do nothing

    def log_train_time(self, train_time):
        """ Log total training time

        Parameters
        ----------
        train_time: float
            Total trianing time
        """
        # By default do nothing

    def log_valid_time(self, valid_time):
        """ Log total validation time

        Parameters
        ----------
        valid_time: float
            Total validation time
        """
        # By default do nothing

    @property
    def rank(self):
        """ Task rank in a distributed training/inference cluster
        """
        return self._rank
