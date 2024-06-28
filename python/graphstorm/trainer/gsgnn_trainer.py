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

    GraphStorm trainer base
"""
import os
import logging

from ..model import GSOptimizer
from ..model import GSgnnModel, GSgnnModelBase
from ..model.utils import TopKList
from ..model.utils import remove_saved_models as remove_gsgnn_models
from ..model.utils import save_model_results_json
from ..config import GRAPHSTORM_MODEL_ALL_LAYERS
from ..tracker import GSSageMakerTaskTracker
from ..utils import barrier, get_rank, is_distributed

class GSgnnTrainer():
    """ Generic GSgnn trainer.

    This class is used as a mixin for classes that implement trainers
    for various learning tasks at the node and edge level.

    It contains functions that can be used in the implementing classes'
    `fit` and `eval` functions.

    To implement your own trainers, extend this class and add implementations
    for the `fit` and `eval` functions.

    Parameters
    ----------
    model : GSgnnModel
        The GNN model.
    topk_model_to_save : int
        The top K model to save.
    """
    def __init__(self, model, topk_model_to_save=1):
        super(GSgnnTrainer, self).__init__()
        self._model = model
        optimizer = model.create_optimizer()
        assert optimizer is not None, "The model cannot provide an optimizer"
        if not isinstance(optimizer, GSOptimizer):
            if get_rank() == 0:
                logging.warning("the optimizer is not GSOptimizer. Convert it to GSOptimizer.")
            optimizer = GSOptimizer([optimizer])
        self._optimizer = optimizer
        self._evaluator = None
        self._best_model_path = None

        assert topk_model_to_save >= 0
        self._topklist = TopKList(topk_model_to_save)    # A list to store the top k best
                                                        # perf epoch+iteration for
                                                        # saving/removing models.
        self._task_tracker = None

    def setup_device(self, device):
        """ Set up the device of this trainer.

        The CUDA device is set up based on the local rank.

        Parameters
        ----------
        device :
            The device for model training.
        """
        self._device = device
        self._model = self._model.to(self.device)
        self._optimizer.move_to_device(self._model.device)

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

    def log_metric(self, metric_name, metric_value, step):
        """ log evaluation metric

        Parameters
        ----------
        metric_name: str
            Evaluation metric name
        metric_value: float
            Value
        step: int
            Current step
        """
        if self.task_tracker is None:
            return

        self.task_tracker.log_metric(metric_name, metric_value, step)

    def keep_alive(self, report_step):
        """ Dummy log, send keep alive message to mlflow server

        Parameters
        ----------
        report_step: int
            Current exec step. Used to decide whether send dummy info
        """
        if self.task_tracker is None:
            return

        self.task_tracker.keep_alive(report_step)

    def log_param(self, param_name, param_value):
        """ Log parameters

        Parameters
        ----------
        param_name: str
            Parameter name
        param_value:
            Parameter value
        """
        if self.task_tracker is None:
            return

        self.task_tracker.log_param(param_name, param_value)

    def log_params(self, param_value):
        """ Log a dict of parameters

        Parameter
        ---------
        param_value: dict
            Key value pairs of parameters to log
        """
        if self.task_tracker is None:
            return

        self.task_tracker.log_params(param_value)

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
                train_score=train_score, val_score=val_score,
                test_score=test_score, best_val_score=best_val_score,
                best_test_score=best_test_score, best_iter_num=best_iter_num,
                eval_time=dur_eval, total_steps=total_steps)

    def save_model(self, model, epoch, i, save_model_path):
        '''Save the model for a certain iteration in an epoch.
        '''
        barrier()
        if save_model_path is not None:
            module = model.module if is_distributed() else model
            assert isinstance(module, (GSgnnModel, GSgnnModelBase)), \
                "Please make sure the model derives from GSgnnModel or GSgnnModelBase, " \
                "which provides a scalable model saving implementation."
            save_model_path = self._gen_model_path(save_model_path, epoch, i)
            module.save_model(save_model_path)
            self.optimizer.save_opt_state(save_model_path)

        # make sure each trainer finishes its own model saving task.
        barrier()

    def remove_saved_model(self, epoch, i, save_model_path):
        """ remove previously saved model, which may not be the best K performed or other reasons.
            This function will remove the entire folder.

        Parameters
        ----------
        epoch: int
            The number of training epoch.
        i: int
            The number of iteration in a training epoch.
        save_model_path : str
            The path where the model is saved.
        """
        if save_model_path is not None and get_rank() == 0:
            # construct model path
            saved_model_path = self._gen_model_path(save_model_path, epoch, i)

            # remove the folder that contains saved model files.
            remove_status = remove_gsgnn_models(saved_model_path)
            if remove_status == 0:
                logging.debug('Successfully removed the saved model files in %s',
                              saved_model_path)

    def save_topk_models(self, model, epoch, i, val_score, save_model_path):
        """ Based on the given val_score, decided if save the current model trained in the i_th
            iteration and the epoch_th epoch.

        Parameters
        ----------
        model : pytorch model
            The GNN model.
        epoch: int
            The number of training epoch.
        i: int
            The number of iteration in a training epoch.
        val_score: dict or None
            A dictionary contains scores from evaluator's validation function. It could be None
            that means there is either no evluator or not do validation. In that case, just set
            the score rank as 1st to save all models or the last k models.
        save_model_path : str
            The path where the model is saved.
        """

        # compute model validation score rank in evaluator
        if val_score is None:
            score_rank = 1
        else:
            score_rank = self.evaluator.get_val_score_rank(val_score)

        insert_success, (return_epoch, return_i) = self._topklist.insert(score_rank, (epoch, i))

        if insert_success:
            # if success, should always save this epoch and/or iteration models, and remove the
            # previous worst model saved if the return_epoch or return_i is different from current
            # epoch and i
            if return_epoch != epoch or return_i != i:
                # here the return_epoch and return_i are the epoch and iteration number that
                # performan worst in the previous top k list.
                self.remove_saved_model(return_epoch, return_i, save_model_path)

            # save this epoch and iteration's model and node embeddings
            # All trainers will sync in save_model before start saving a model.
            self.save_model(model, epoch, i, save_model_path)

            # If this is the best model
            if score_rank == 1 and save_model_path is not None:
                self._best_model_path = self._gen_model_path(save_model_path, epoch, i)

    def get_best_model_path(self):
        """ Return the path of the best model.
        """
        assert self._best_model_path is not None, "Cannot get the best model from the trainer."
        assert os.path.exists(self._best_model_path), \
                f"The model path {self._best_model_path} does not exist." \
                + "Please make sure that the model is saved in a shared filesystem."
        return self._best_model_path

    def _gen_model_path(self, base_path, epoch, i):
        """
        Generate the model path for both saving and removing a folder that contains model files.
        """
        model_path = os.path.join(base_path, 'epoch-' + str(epoch))
        if i is not None:
            model_path = model_path + '-iter-' + str(i)

        return model_path

    def save_model_results_to_file(self, test_model_performance, save_perf_results_path):
        """Save model's performance results to a local JSON file.
        """
        # cast value to str to avoid serialization error from certain classes.
        conf = {k: str(v) for k, v in self.__dict__.items()}
        save_model_results_json(conf=conf,
                                test_model_performance=test_model_performance,
                                save_perf_results_path=save_perf_results_path)

    def restore_model(self, model_path, model_layer_to_load=None):
        """ Restore a GNN model and the optimizer.

        Parameters
        ----------
        model_path : str
            The path where the model and the optimizer state has been saved.
        model_layer_to_load: list of str
            list of model layers to load. Supported layers include
            'gnn', 'embed', 'decoder'
        """
        self._model.restore_model(model_path, model_layer_to_load)

        # If we only load part of a saved model for model fine-tuning,
        # we do not load optimizer states as the model states of
        # two models (pre-training and fine-tuning) are not 100%
        # compatible.
        if model_layer_to_load == GRAPHSTORM_MODEL_ALL_LAYERS:
            self._optimizer.load_opt_state(model_path, self._model.device)

    def can_do_validation(self, val_dataloader):
        """ A unified method to judge if a trainer can do model evaluation

        Parameters
        ----------
        val_dataloader: Dataloader
            GraphStorm Dataloader for validation

        Return
        -------
        True or False
            Whether do model validation.
        """
        # check if have evaluator or if have validation dataloader
        if self.evaluator is None or val_dataloader is None:
            return False
        else:
            return True

    @property
    def evaluator(self):
        """ The evaluator associated with the trainer.
        """
        return self._evaluator

    @property
    def optimizer(self):
        """ The optimizer associated with the trainer.
        """
        return self._optimizer

    @property
    def task_tracker(self):
        """ The task tracker associated with the trainer.
        """
        return self._task_tracker

    @property
    def device(self):
        """ The device associated with the trainer.
        """
        return self._device
