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

    GraphStorm trainer for node prediction.
"""
import time
import resource
import logging
import torch as th
from torch.nn.parallel import DistributedDataParallel

from ..model.node_gnn import node_mini_batch_gnn_predict, node_mini_batch_predict
from ..model.node_gnn import GSgnnNodeModelInterface
from ..model import do_full_graph_inference, GSgnnModelBase, GSgnnModel
from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker, rt_profiler, print_mem, get_rank
from ..utils import barrier, is_distributed, get_backend

class GSgnnNodePredictionTrainer(GSgnnTrainer):
    """ Trainer for node prediction tasks.

    ``GSgnnNodePredictionTrainer`` is used to train models for node prediction tasks,
    such as node classification and node regression. ``GSgnnNodePredictionTrainer``
    defines two main functions: 

    * ``fit()``: performs the training for the model provided to this trainer
      when the object is initialized, and;
    * ``eval()``: evaluates the provided model against test and validation dataset.

    Example
    --------

    .. code:: python

        from graphstorm.dataloading import GSgnnNodeDataLoader
        from graphstorm.dataset import GSgnnData
        from graphstorm.model import GSgnnNodeModel
        from graphstorm.trainer import GSgnnNodePredictionTrainer

        np_data = GSgnnData("...")
        target_idx = np_data.get_node_train_set('ntype1')
        train_loader = GSgnnNodeDataLoader(
            np_dataset, target_idx, fanout=[10], batch_size=1024,
            label_field="label", node_feats="feat", train_task=True)
        model = GSgnnNodeModel(alpha_l2norm=0.0)

        trainer = GSgnnNodePredictionTrainer(model)

        trainer.fit(train_loader, num_epochs=2)

    Parameters
    ----------
    model: GSgnnNodeModelBase
        The GNN model for node prediction, which could be a model class inherited from the
        ``GSgnnNodeModelBase``, or a model class that inherits both the ``GSgnnModelBase``
        and the ``GSgnnNodeModelInterface``.
    topk_model_to_save: int
        The top `K` model to be saved based on evaluation results. Default: 1.
    """
    def __init__(self, model, topk_model_to_save=1):
        super(GSgnnNodePredictionTrainer, self).__init__(model, topk_model_to_save)
        assert isinstance(model, GSgnnNodeModelInterface) and isinstance(model, GSgnnModelBase), \
                "The input model is not a node model. Please implement GSgnnNodeModelBase."

    def fit(self, train_loader, num_epochs,
            val_loader=None,
            test_loader=None,
            use_mini_batch_infer=True,
            save_model_path=None,
            save_model_frequency=-1,
            save_perf_results_path=None,
            freeze_input_layer_epochs=0,
            max_grad_norm=None,
            grad_norm_type=2.0):
        """ Fit function for node prediction training.

        This function performs the training for the given node prediction model.
        It iterates over the training batches provided by the ``train_loader``
        to compute the loss, and then performs the backward steps using trainer's
        own optimizer. 

        If an evaluator and a validation dataloader are added to this trainer, during
        training, the trainer will perform model evaluation in three cases:

        * At the end of each epoch.
        * At the evaluation frequency (number of iterations) defined in the evaluator.
        * Before saving a model checkpoint.

        Parameters
        ----------
        train_loader: GSgnnNodeDataLoader
            Node dataloader for mini-batch sampling the training set.
        num_epochs: int
            The max number of epochs used to train the model.
        val_loader: GSgnnNodeDataLoader
            Node dataloader for mini-batch sampling the validation set. Default: None.
        test_loader: GSgnnNodeDataLoader
            Node dataloader for mini-batch sampling the test set. Default: None.
        use_mini_batch_infer: bool
            Whether to use mini-batch for inference. Default: True.
        save_model_path: str
            The path where trained model checkpoints are saved. If is None, will not
            save model checkpoints.
            Default: None.
        save_model_frequency: int
            The number of iterations to train the model before saving a model checkpoint. 
            Default: -1, meaning only save model after each epoch.
        save_perf_results_path: str
            The path of the file where the performance results are saved. Default: None.
        freeze_input_layer_epochs: int
            The number of epochs to freeze the input layer from updating trainable
            parameters. This is commonly used when the input layer contains language models.
            Default: 0.
        max_grad_norm: float
            A value used to clip the gradient, which can enhance training stability.
            More explanation of this argument can be found
            in `torch.nn.utils.clip_grad_norm_ <https://pytorch.org/docs/2.1/generated/
            torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_>`__.
            Default: None.
        grad_norm_type: float
            Norm type for the gradient clip. More explanation of this argument can be found
            in `torch.nn.utils.clip_grad_norm_ <https://pytorch.org/docs/2.1/generated/
            torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_>`__.
            Default: 2.0.
        """
        # Check the correctness of configurations.
        if self.evaluator is not None:
            assert val_loader is not None, \
                    "The evaluator is provided but validation set is not provided."
        if not use_mini_batch_infer:
            assert isinstance(self._model, GSgnnModel), \
                    "Only GSgnnModel supports full-graph inference."

        # with freeze_input_layer_epochs is 0, computation graph will not be changed.
        static_graph = freeze_input_layer_epochs == 0
        on_cpu = self.device == th.device('cpu')
        if is_distributed():
            model = DistributedDataParallel(self._model,
                                            device_ids=None if on_cpu else [self.device],
                                            output_device=None if on_cpu else self.device,
                                            find_unused_parameters=True,
                                            static_graph=static_graph)
        else:
            model = self._model
        device = model.device
        data = train_loader.data

        # Preparing input layer for training or inference.
        # The input layer can pre-compute node features in the preparing step if needed.
        # For example pre-compute all BERT embeddings
        if freeze_input_layer_epochs > 0:
            self._model.freeze_input_encoder(data)
        # TODO(xiangsx) Support freezing gnn encoder and decoder

        # training loop
        total_steps = 0
        early_stop = False # used when early stop is True
        sys_tracker.check('start training')
        g = data.g
        for epoch in range(num_epochs):
            model.train()
            epoch_start = time.time()
            if freeze_input_layer_epochs <= epoch:
                self._model.unfreeze_input_encoder()
            # TODO(xiangsx) Support unfreezing gnn encoder and decoder

            # TODO(zhengda) the dataloader should return node features and labels directly.
            rt_profiler.start_record()
            batch_tic = time.time()
            for i, (input_nodes, seeds, blocks) in enumerate(train_loader):
                rt_profiler.record('train_sample')
                total_steps += 1

                if not isinstance(input_nodes, dict):
                    # This happens on a homogeneous graph.
                    assert len(g.ntypes) == 1, \
                        "The graph should be a homogeneous graph, " \
                        f"but it has multiple node types {g.ntypes}"
                    input_nodes = {g.ntypes[0]: input_nodes}
                nfeat_fields = train_loader.node_feat_fields
                label_field = train_loader.label_field
                input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)
                lbl = data.get_node_feats(seeds, label_field, device)
                rt_profiler.record('train_node_feats')

                blocks = [block.to(device) for block in blocks]
                rt_profiler.record('train_graph2GPU')

                # TODO(zhengda) we don't support edge features for now.
                loss = model(blocks, input_feats, None, lbl, input_nodes)
                rt_profiler.record('train_forward')

                self.optimizer.zero_grad()
                loss.backward()
                rt_profiler.record('train_backward')
                self.optimizer.step()
                rt_profiler.record('train_step')

                if max_grad_norm is not None:
                    th.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, grad_norm_type)
                self.log_metric("Train loss", loss.item(), total_steps)

                if i % 20 == 0 and get_rank() == 0:
                    rt_profiler.print_stats()
                    logging.info("Part %d | Epoch %05d | Batch %03d | Loss: %.4f | Time: %.4f",
                                 get_rank(), epoch, i,  loss.item(), time.time() - batch_tic)

                val_score = None
                if self.can_do_validation(val_loader) and self.evaluator.do_eval(total_steps):
                    val_score = self.eval(model.module if is_distributed() else model,
                                          val_loader, test_loader,
                                          use_mini_batch_infer, total_steps, return_proba=False)

                    if self.evaluator.do_early_stop(val_score):
                        early_stop = True

                # In every save_model_frequency iterations, check to save the top k models.
                # If has validation score, will save the best top k. If no validation, will
                # either save the last k model or all models depends on the setting of top k.
                if save_model_frequency > 0 and \
                    total_steps % save_model_frequency == 0 and \
                    total_steps != 0:
                    if val_score is None:
                        # not in the same eval_frequncy iteration
                        if self.can_do_validation(val_loader):
                            # for model saving, force to do evaluation if can
                            val_score = self.eval(model.module if is_distributed() else model,
                                                val_loader, test_loader, use_mini_batch_infer,
                                                total_steps, return_proba=False)
                    # We will save the best model when
                    # 1. If not do evaluation, we will keep the latest K models.
                    # 2. If do evaluaiton, we need to follow the guidance of validation score.
                    self.save_topk_models(model, epoch, i, val_score, save_model_path)

                rt_profiler.record('train_eval')
                batch_tic = time.time()
                # early_stop, exit current interation.
                if early_stop is True:
                    break

            # ------- end of an epoch -------

            barrier()
            epoch_time = time.time() - epoch_start
            if get_rank() == 0:
                logging.info("Epoch %d take %.3f seconds", epoch, epoch_time)

            val_score = None
            # do evaluation and mode saving after each epoch if can
            if self.can_do_validation(val_loader):
                val_score = self.eval(model.module if is_distributed() else model,
                                      val_loader, test_loader,
                                      use_mini_batch_infer, total_steps, return_proba=False)
                if self.evaluator.do_early_stop(val_score):
                    early_stop = True

            # After each epoch, check to save the top k models. If has validation score, will save
            # the best top k. But if no validation, will either save the last k model or all models
            # depends on the setting of top k. To show this is after epoch save, set the iteration
            # to be None, so that we can have a determistic model folder name for testing and debug.
            self.save_topk_models(model, epoch, None, val_score, save_model_path)
            # make sure saving model finishes properly before the main process kills this training
            barrier()

            # early_stop, exit training
            if early_stop is True:
                break

        rt_profiler.save_profile()
        print_mem(device)
        if get_rank() == 0 and self.evaluator is not None:
            output = {'best_test_score': self.evaluator.best_test_score,
                       'best_val_score': self.evaluator.best_val_score,
                       'peak_GPU_mem_alloc_MB': th.cuda.max_memory_allocated(device) / 1024 / 1024,
                       'peak_RAM_mem_alloc_MB': \
                           resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       'best validation iteration': \
                           self.evaluator.best_iter_num[self.evaluator.metric_list[0]],
                       'best model path': \
                           self.get_best_model_path() if save_model_path is not None else None}
            self.log_params(output)

            if save_perf_results_path is not None:
                self.save_model_results_to_file(self.evaluator.best_test_score,
                                                save_perf_results_path)

    def eval(self, model, val_loader, test_loader, use_mini_batch_infer, total_steps,
             return_proba=True):
        """ Do model evaluation using the validation set, or test set if provided.

        Parameters
        ----------
        model: GSgnnNodeModelBase
            The GNN model for node prediction, which could be a model class inherited from the
            ``GSgnnNodeModelBase``, or a model class that inherits both the ``GSgnnModelBase``
            and the ``GSgnnNodeModelInterface``.
        val_loader: GSgnnNodeDataLoader
            Node dataloader for mini-batch sampling the validation set. Default: None.
        test_loader: GSgnnNodeDataLoader
            Node dataloader for mini-batch sampling the test set. Default: None.
        use_mini_batch_infer: bool
            Whether to use mini-batch for inference. Default: True.
        total_steps: int
            The total number of iterations.
        return_proba: bool
            Whether to return the prediction results or the argmax results for
            classification tasks.

        Returns
        -------
        val_score: dict
            Validation scores of differnet metrics in the format of {metric: val_score}.
        """
        teval = time.time()
        sys_tracker.check('before prediction')

        metric = set(self.evaluator.metric_list)
        need_proba = metric.intersection({'roc_auc', 'per_class_roc_auc', 'precision_recall'})
        need_label_pred = metric.intersection({'accuracy', 'f1_score', 'per_class_f1_score'})
        assert len(need_proba) == 0 or len(need_label_pred) == 0, \
            f"{need_proba} requires return_proba==True, \
                         but {need_label_pred} requires return_proba==False."
        if len(need_proba) > 0 and return_proba is False:
            return_proba = True
            logging.warning("%s requires return_proba==True. \
                Set return_proba to True.", need_proba)

        if use_mini_batch_infer:
            val_pred, _, val_label = node_mini_batch_gnn_predict(model, val_loader, return_proba,
                                                                 return_label=True)
            sys_tracker.check('after_val_score')
            if test_loader is not None:
                test_pred, _, test_label = \
                    node_mini_batch_gnn_predict(model, test_loader, return_proba,
                                                return_label=True)
            else: # there is no test set
                test_pred = None
                test_label = None
            sys_tracker.check('after_test_score')
        else:
            emb = do_full_graph_inference(model, val_loader.data, fanout=val_loader.fanout,
                                          task_tracker=self.task_tracker)
            sys_tracker.check('after_full_infer')
            val_pred, val_label = node_mini_batch_predict(model, emb, val_loader, return_proba,
                                                          return_label=True)
            sys_tracker.check('after_val_score')
            if test_loader is not None:
                test_pred, test_label = \
                    node_mini_batch_predict(model, emb, test_loader, return_proba,
                                            return_label=True)
            else:
                # there is no test set
                test_pred = None
                test_label = None
            sys_tracker.check('after_test_score')
        sys_tracker.check('predict')

        # TODO(wlcong) we only support node prediction on one node type for evaluation now
        assert len(val_label) == 1, "We only support prediction on one node type for now."
        ntype = list(val_label.keys())[0]
        # We need to have val and label (test and test label) data in GPU
        # when backend is nccl, as we need to use nccl.all_reduce to exchange
        # data between GPUs
        val_pred = val_pred[ntype].to(self.device) \
            if is_distributed() and get_backend() == "nccl" else val_pred[ntype]
        val_label = val_label[ntype].to(self.device) \
            if is_distributed() and get_backend() == "nccl" else val_label[ntype]
        if test_pred is not None:
            test_pred = test_pred[ntype].to(self.device) \
                if is_distributed() and get_backend() == "nccl" else test_pred[ntype]
            test_label = test_label[ntype].to(self.device) \
                if is_distributed() and get_backend() == "nccl" else test_label[ntype]
        val_score, test_score = self.evaluator.evaluate(val_pred, test_pred,
                                                        val_label, test_label, total_steps)
        sys_tracker.check('evaluate')
        if get_rank() == 0:
            self.log_print_metrics(val_score=val_score,
                                    test_score=test_score,
                                    dur_eval=time.time() - teval,
                                    total_steps=total_steps)
        return val_score
