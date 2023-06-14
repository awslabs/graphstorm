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
import psutil
import torch as th

from ..model import GSOptimizer
from ..model import GSgnnModel, GSgnnModelBase
from ..model.utils import TopKList
from ..model.utils import remove_saved_models as remove_gsgnn_models
from ..model.utils import save_model_results_json
from ..config import GRAPHSTORM_MODEL_ALL_LAYERS

class GSgnnTrainer():
    """ Generic GSgnn trainer.

    Parameters
    ----------
    model : GSgnnModel
        The GNN model.
    rank : int
        The rank.
    topk_model_to_save : int
        The top K model to save.
    """
    def __init__(self, model, rank, topk_model_to_save=1):
        super(GSgnnTrainer, self).__init__()
        self._model = model
        optimizer = model.create_optimizer()
        assert optimizer is not None, "The model cannot provide an optimizer"
        if not isinstance(optimizer, GSOptimizer):
            if rank == 0:
                print("Warining: the optimizer is not GSOptimizer. "
                        + "Convert it to GSOptimizer.")
            optimizer = GSOptimizer([optimizer])
        self._optimizer = optimizer
        self._rank = rank
        self._dev_id = -1
        self._evaluator = None
        self._task_tracker = None
        self._best_model_path = None

        assert topk_model_to_save >= 0
        self._topklist = TopKList(topk_model_to_save)    # A list to store the top k best
                                                        # perf epoch+iteration for
                                                        # saving/removing models.

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
        self._optimizer.move_to_device(self._model.device)

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
        self.task_tracker.log_iter_metrics(self.evaluator.metric,
                train_score=train_score, val_score=val_score,
                test_score=test_score, best_val_score=best_val_score,
                best_test_score=best_test_score, best_iter_num=best_iter_num,
                eval_time=dur_eval, total_steps=total_steps)

    def save_model(self, model, epoch, i, save_model_path):
        '''Save the model for a certain iteration in an epoch.
        '''
        th.distributed.barrier()
        if save_model_path is not None:
            assert isinstance(model.module, (GSgnnModel, GSgnnModelBase)), \
                "Please make sure the model derives from GSgnnModel or GSgnnModelBase, " \
                "which provides a scalable model saving implementation."
            save_model_path = self._gen_model_path(save_model_path, epoch, i)
            model.module.save_model(save_model_path)
            self.optimizer.save_opt_state(save_model_path)

        # make sure each trainer finishes its own model saving task.
        th.distributed.barrier()

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
        if save_model_path is not None and self.rank == 0:
            # construct model path
            saved_model_path = self._gen_model_path(save_model_path, epoch, i)

            # remove the folder that contains saved model files.
            remove_status = remove_gsgnn_models(saved_model_path)
            if remove_status == 0:
                print(f'Successfully removed the saved model files in {saved_model_path}')

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

    def print_info(self, epoch, i, num_input_nodes, compute_time):
        ''' Print basic information during training

        Parameters:
        epoch: int
            The epoch number
        i: int
            The current iteration
        num_input_nodes: int
            number of input nodes
        compute_time: tuple of ints
            A tuple of (forward time and backward time)
        '''
        gnn_forward_time, back_time = compute_time
        device = 'cuda:%d' % self.dev_id

        print("Epoch {:05d} | Batch {:03d} | GPU Mem reserved: {:.4f} MB | Peak Mem: {:.4f} MB".
                format(epoch, i,
                    th.cuda.memory_reserved(device) / 1024 / 1024,
                    th.cuda.max_memory_allocated(device) / 1024 /1024))
        print('Epoch {:05d} | Batch {:03d} | RAM memory {} used | Avg input nodes per iter {}'.
                format(epoch, i, psutil.virtual_memory(), num_input_nodes))
        print('Epoch {:05d} | Batch {:03d} | forward {:05f} | Backward {:05f}'.format(
            epoch, i, gnn_forward_time, back_time))

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
    def dev_id(self):
        """ The device associated with the trainer.
        """
        return self._dev_id

    @property
    def rank(self):
        """ The rank of the trainer.
        """
        return self._rank
