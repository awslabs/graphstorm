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

    Infererence framework.
"""
import torch as th

class GSInfer():
    """ Generic GSgnn infer.


    Parameters
    ----------
    model : GSgnnNodeModel
        The GNN model for node prediction.
    rank : int
        The rank.
    """
    def __init__(self, model, rank):
        self._model = model
        self._rank = rank
        self._dev_id = -1
        self._evaluator = None
        self._task_tracker = None

    def setup_cuda(self, dev_id):
        """ Set up the CUDA device of this trainer.

        The CUDA device is set up based on the local rank.

        Parameters
        ----------
        dev_id : int
            The device ID for model training.
        """
        # setup cuda env
        use_cuda = th.cuda.is_available()
        assert use_cuda, "Only support GPU training"
        th.cuda.set_device(dev_id)
        self._dev_id = dev_id
        self._model = self._model.to(self.dev_id)

    def setup_task_tracker(self, task_tracker):
        """ Set the task tracker.

        Parameters
        ----------
        task_tracker : GSTaskTracker
            The task tracker
        """
        if self.evaluator is not None:
            self.evaluator.setup_task_tracker(task_tracker)
        self._task_tracker = task_tracker

    def setup_evaluator(self, evaluator):
        """ Set the evaluator
        """
        if self.task_tracker is not None:
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
        self.task_tracker.log_iter_metrics(self.evaluator.metric,
                train_score=train_score, val_score=val_score,
                test_score=test_score, best_val_score=best_val_score,
                best_test_score=best_test_score, best_iter_num=best_iter_num,
                eval_time=dur_eval, total_steps=total_steps)

    @property
    def evaluator(self):
        """ Get the evaluator associated with the inference.
        """
        return self._evaluator

    @property
    def task_tracker(self):
        """ Get the task tracker associated with the inference.
        """
        return self._task_tracker

    @property
    def dev_id(self):
        """ Get the device ID associated with the inference.
        """
        return self._dev_id

    @property
    def rank(self):
        """ Get the rank the inference.
        """
        return self._rank
