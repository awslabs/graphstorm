"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    TensorBoard task tracker
"""
import numbers
import logging
import importlib

from ..utils import get_rank
from .sagemaker_tracker import GSSageMakerTaskTracker

class GSTensorBoardTracker(GSSageMakerTaskTracker):
    """ GraphStorm builtin TensorBoard task tracker.

        GSTensorBoardTracker inherits from GSSageMakerTaskTracker.
        It follows the same logic as GSSageMakerTaskTracker to print logs.
        It uses torch.utils.tensorboard.SummaryWriter to
        dump training losses, validation and test
        scores into log files.

        Parameters
        ----------
        log_report_frequency: int
            The frequency of reporting model performance metrics through task_tracker.
            The frequency is defined by using number of iterations, i.e., every N iterations
            the evaluation metrics will be reported.
        log_dir: str
            Save directory location. The default setting is
            runs/**CURRENT_DATETIME_HOSTNAME**, which changes after each run.
            Use hierarchical folder structure to compare
            between runs easily. e.g. pass in 'runs/exp1', 'runs/exp2', etc.
            See https://pytorch.org/docs/stable/tensorboard.html for more detials.
            Default: None.

        .. versionadded:: 0.4.1
            The :py:class:`GSTensorBoardTracker`.
    """
    def __init__(self, log_report_frequency, log_dir=None):
        super().__init__(log_report_frequency, log_dir)
        try:
            tensorboard = importlib.import_module("torch.utils.tensorboard")
        except ImportError as err:
            msg =  (
                "GSTensorBoardTracker requires tensorboard to run. "
                "Please install the tensorboard Python package.")
            raise ImportError(msg) from err
        self._writer  = tensorboard.SummaryWriter(log_dir)

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
                    # Only rank 0 will write log to TensorBoard
                    if get_rank() == 0:
                        self._writer.add_text(metric_name, metric_value, step)
                    logging.info("Step %d | %s: %s", step, metric_name, metric_value)
                elif isinstance(metric_value, numbers.Number):
                    # Only rank 0 will write log to TensorBoard
                    if get_rank() == 0:
                        self._writer.add_scalar(metric_name, metric_value, step)
                    logging.info("Step %d | %s: %.4f", step, metric_name, metric_value)
                else:
                    # Only rank 0 will write log to TensorBoard
                    if get_rank() == 0:
                        self._writer.add_text(metric_name, str(metric_value), step)
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
        metric_name = f"{metric_name}/Train"
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
        metric_name = f"{metric_name}/Best Test"
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
        metric_name = f"{metric_name}/Test"
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
        metric_name = f"{metric_name}/Best Validation"
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
        metric_name = f"{metric_name}/Validation"
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
        metric_name = f"{metric_name}/Best Iteration"
        self.log_metric(metric_name, best_iter, step, force_report)
