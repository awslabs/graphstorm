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

    Inference framework.
"""
from ..tracker import GSSageMakerTaskTracker

class GSInferrer():
    """ Generic GSgnn Inferrer.

    Parameters
    ----------
    model : GSgnnModel
        This model could be one of the internal GraphStorm GNN models, i.e.,
        ``GSgnnNodeModel``, ``GSgnnEdgeModel``, ``GSgnnLinkPredictionModel``, or a model
        class that inherit them.  
        For customized GNN models, they should be the concrete implementation that inherits
        one of the ``GSgnnNodeModelBase``, ``GSgnnEdgeModelBase``, and
        ``GSgnnLinkPredictionModelBase`` classes.
    """
    def __init__(self, model):
        self._model = model
        self._device = -1
        self._evaluator = None
        self._task_tracker = None

    def setup_device(self, device):
        """ Set up the device for the inferrer.

        The CUDA device is set up based on the local rank.

        Parameters
        ----------
        device :
            The device for inferrer.
        """
        self._device = device
        self._model = self._model.to(self.device)

    def setup_task_tracker(self, task_tracker):
        """ Set the task tracker.

        Parameters
        ----------
        task_tracker : GSTaskTracker
            The task tracker
        """
        self._task_tracker = task_tracker

    def setup_evaluator(self, evaluator):
        """ Setup the evaluator

        If the evaluator has its own task tracker, just setup the evaluator. But if the evaluator
        has no task tracker, will use this Trainer's task tracker to setup the evaluator. When there
        is no self task tracker, will create a new one by using the given evaluator's evaluation
        frequency.
        """
        if evaluator.task_tracker is None:
            if self.task_tracker is None:
                self.setup_task_tracker(GSSageMakerTaskTracker(evaluator.eval_frequency))

            evaluator.setup_task_tracker(self.task_tracker)

        self._evaluator = evaluator

    def log_print_metrics(self, val_score, test_score, dur_eval, total_steps, train_score=None):
        """
        This function prints and logs all the metrics for evaluation

        Parameters
        ----------
        train_score: dict
            Training score
        val_score: dict
            Validation score
        test_score: dict
            Test score
        dur_eval:
            Total evaluation time
        total_steps: int
            The corresponding step/iteration
        """
        if self.task_tracker is None:
            return

        best_val_score = self.evaluator.best_val_score
        best_test_score = self.evaluator.best_test_score
        best_iter_num = self.evaluator.best_iter_num
        self.task_tracker.log_iter_metrics(self.evaluator.metric_list,
                                           train_score=train_score,
                                           val_score=val_score,
                                           test_score=test_score,
                                           best_val_score=best_val_score,
                                           best_test_score=best_test_score,
                                           best_iter_num=best_iter_num,
                                           eval_time=dur_eval,
                                           total_steps=total_steps)

    @property
    def evaluator(self):
        """ Get the evaluator associated with the inferrer.
        """
        return self._evaluator

    @property
    def task_tracker(self):
        """ Get the task tracker associated with the inferrer.
        """
        return self._task_tracker

    @property
    def device(self):
        """ The device associated with the inferrer.
        """
        return self._device
