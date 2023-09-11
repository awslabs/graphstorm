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

    SageMaker task tracker
"""
import numbers
import logging

from .graphstorm_tracker import GSTaskTrackerAbc

class GSSageMakerTaskTracker(GSTaskTrackerAbc):
    """ GraphStorm builtin SageMaker task tracker

        Parameters
        ----------
        config: GSConfig
            Configurations. Users can add their own configures in the yaml config file.
        rank: int
            Task rank
    """

    def _do_report(self, step):
        """ Whether report the metric

        Parameters
        ----------
        step: int
            Current step
        """
        return step % self._report_frequency == 0

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
        if force_report or self._do_report(step):
            if metric_value is not None:
                if isinstance(metric_value, str):
                    logging.info("Step %d | %s: %s", step, metric_name, metric_value)
                elif isinstance(metric_value, numbers.Number):
                    logging.info("Step %d | %s: %.4f", step, metric_name, metric_value)
                else:
                    logging.info("Step %d | %s: %s", step, metric_name, str(metric_value))

    def log_train_metric(self, metric_name, metric_value, step, force_report=False):
        """ Log train metric

            Parameters
            ----------
            metric_name: str
                Train metric name
            metric_value:
                Train metric value
            step: int
                The corresponding step/iteration in the training loop.
            force_report: bool
                If true, report the metric
        """
        metric_name = f"Train {metric_name}"
        self.log_metric(metric_name, metric_value, step, force_report)

    def log_best_test(self, metric_name, metric_value, step, force_report=False):
        """ Log best test score

            Parameters
            ----------
            metric_name: str
                Test metric name
            metric_value:
                Test metric value
            step: int
                The corresponding step/iteration in the training loop.
            force_report: bool
                If true, report the metric
        """
        metric_name = f"Best Test {metric_name}"
        self.log_metric(metric_name, metric_value, step, force_report)

    def log_test_metric(self, metric_name, metric_value, step, force_report=False):
        """ Log test metric

            Parameters
            ----------
            metric_name: str
                Test metric name
            metric_value:
                Test metric value
            step: int
                The corresponding step/iteration in the training loop.
            force_report: bool
                If true, report the metric
        """
        metric_name = f"Test {metric_name}"
        self.log_metric(metric_name, metric_value, step, force_report)

    def log_best_valid(self, metric_name, metric_value, step, force_report=False):
        """ Log best validation score

            Parameters
            ----------
            metric_name: str
                Validation metric name
            metric_value:
                Validation metric value
            step: int
                The corresponding step/iteration in the training loop.
            force_report: bool
                If true, report the metric
        """
        metric_name = f"Best Validation {metric_name}"
        self.log_metric(metric_name, metric_value, step, force_report)

    def log_valid_metric(self, metric_name, metric_value, step, force_report=False):
        """ Log validation metric

            Parameters
            ----------
            metric_name: str
                Validation metric name
            metric_value: float
                Validation metric value
            step: int
                The corresponding step/iteration in the training loop.
            force_report: bool
                If true, report the metric
        """
        metric_name = f"Validation {metric_name}"
        self.log_metric(metric_name, metric_value, step, force_report)

    def log_best_iter(self, metric_name, best_iter, step, force_report=False):
        """ Log best iteration

            Parameters
            ----------
            metric_name: str
                Metric name
            iter:
                Best iteration number
            step: int
                The corresponding step/iteration in the training loop.
            force_report: bool
                If true, report the metric
        """
        metric_name = f"Best Iteration {metric_name}"
        self.log_metric(metric_name, best_iter, step, force_report)

    def log_mean_forward_time(self, forward_time):
        """ Log average forward time

        Parameters
        ----------
        forward_time: float
            Average forward time
        """
        # {'Name': 'Mean Forward Time', 'Regex': 'Mean forward time: ([0-9\\.]+)'}
        logging.info("Mean forward time: %.4f.", forward_time)

    def log_mean_backward_time(self, backward_time):
        """ Log average backward time

        Parameters
        ----------
        backward_time: float
            Average backward time
        """
        # {'Name': 'Mean Backward Time', 'Regex': 'Mean backward time: ([0-9\\.]+)'}
        logging.info("Mean backword time: %.4f.", backward_time)

    def log_train_time(self, train_time):
        """ Log total training time

        Parameters
        ----------
        train_time: float
            Total trianing time
        """
        # 'Name': 'Train Time', 'Regex': 'Total train Time: ([0-9\\.]+)'
        logging.info("Total train Time: %.4f.", train_time)

    def log_valid_time(self, valid_time):
        """ Log total validation time

        Parameters
        ----------
        valid_time: float
            Total validation time
        """
        # 'Name': 'Validation Time', 'Regex': 'Total validatoin Time: ([0-9\\.]+)'
        logging.info("Total validatoin Time: %.4f.", valid_time)

    def log_param(self, param_name, param_value):
        logging.info("%s: %s", param_name, str(param_value))

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
        for eval_metric in eval_metrics:
            train_score_metric = train_score[eval_metric] if train_score is not None else None
            val_score_metric = val_score[eval_metric]
            test_score_metric = test_score[eval_metric]
            best_val_score_metric = best_val_score[eval_metric]
            best_test_score_metric = best_test_score[eval_metric]
            best_iter_metric = best_iter_num[eval_metric]

            # Each metric has only one score.
            self.log_per_metric(eval_metric,
                                train_score_metric,
                                val_score_metric,
                                test_score_metric,
                                eval_time,
                                total_steps,
                                best_val_score_metric,
                                best_test_score_metric,
                                best_iter_metric)

    def log_per_metric(self, metric, train_score, val_score, test_score,
        dur_eval, total_steps, best_val_score, best_test_score, best_iter_num):
        """ Log information of a evaluation metric

            Parameters
            ----------
            metric: str
                Evaluation metric
            train_score: float
                Training score
            val_score: float
                Validation score
            test_score: float
                Test score
            dur_eval: float
                Evaluation time
            total_steps: int
                Current step
            best_val_score: float
                Best validation score
            best_test_score: float
                Best test score
            best_iter_num: int
                Best iteration number
        """
        self.log_train_metric(metric, train_score, total_steps, force_report=True)
        self.log_valid_metric(metric, val_score, total_steps, force_report=True)
        self.log_test_metric(metric, test_score, total_steps, force_report=True)
        self.log_best_valid(metric, best_val_score, total_steps, force_report=True)
        self.log_best_test(metric, best_test_score, total_steps, force_report=True)
        self.log_best_iter(metric, best_iter_num, total_steps, force_report=True)
        logging.info(" Eval time: %.4f, Evaluation step: %d.", dur_eval, total_steps)
