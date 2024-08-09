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

    GraphStorm GLEM trainer for node prediction.
"""

import time
import logging
import torch as th
from torch.nn.parallel import DistributedDataParallel

from ..model.node_gnn import GSgnnNodeModelInterface
from ..model.node_glem import GLEM
from .np_trainer import GSgnnNodePredictionTrainer

from ..utils import sys_tracker, rt_profiler, print_mem
from ..utils import barrier, get_rank, is_distributed
from ..dataloading import GSgnnNodeSemiSupDataLoader

class GLEMNodePredictionTrainer(GSgnnNodePredictionTrainer):
    r""" Trainer for node prediction tasks using the `GLEM
    framework <https://arxiv.org/abs/2210.14709>`__.

    This class is used to train models for node prediction tasks,
    such as node classification and node regression using the GLEM.

    GLEM iteratively trains a GNN model and a language model in
    turn, allowing it to make better use of textual features in graphs.

    Parameters
    ----------
    model : GSgnnNodeModel
        The GNN model for node prediction. Need to be an instance of
        `model.node_glem.GLEM`.
    topk_model_to_save : int
        The top K model to save.

    Example
    -------

    .. code:: python

        from graphstorm.dataloading import GSgnnNodeDataLoader
        from graphstorm.dataset import GSgnnData
        from graphstorm.model.node_glem import GLEM
        from graphstorm.trainer import GLEMNodePredictionTrainer

        my_dataset = GSgnnData("/path/to/part_config")
        target_idx = {"my_node_type": target_nodes_tensor}
        my_data_loader = GSgnnNodeDataLoader(
            my_dataset, target_idx, fanout=[10], batch_size=1024,
            label_field="label", node_feats="feat", device='cpu')
        my_model = GLEM(alpha_l2norm=0.0, target_ntype="my_node_type")

        trainer =  GLEMNodePredictionTrainer(my_model, topk_model_to_save=1)

        trainer.fit(my_data_loader, num_epochs=2)
    """
    def __init__(self, model, topk_model_to_save=1):
        super(GLEMNodePredictionTrainer, self).__init__(model, topk_model_to_save)
        assert isinstance(model, GSgnnNodeModelInterface) and isinstance(model, GLEM), \
                "The input model is not a GLEM node model. Please implement GLEM."
        self.early_stop = False # used when early stop is True
        self.num_pretrain_epochs = model.num_pretrain_epochs

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
        """ The fit function for node prediction.

        The fit method alternates between training a GNN model and
        a Language Model at every epoch, and can run in fully supervised
        or semi-supervised mode.

        If an evaluator has been assigned to the
        trainer, it will run evaluation at the end of every epoch.

        Parameters
        ----------
        train_loader : GSgnnNodeDataLoader
            The mini-batch sampler for training.
        num_epochs : int
            The max number of epochs to train the model.
        val_loader : GSgnnNodeDataLoader
            The mini-batch sampler for computing validation scores. The validation scores
            are used for selecting models.
        test_loader : GSgnnNodeDataLoader
            The mini-batch sampler for computing test scores.
        use_mini_batch_infer : bool
            Whether or not to use mini-batch inference.
        save_model_path : str
            The path where the model is saved.
        save_model_frequency : int
            The number of iteration to train the model before saving the model.
        save_perf_results_path : str
            The path of the file where the performance results are saved.
        freeze_input_layer_epochs: int
            Freeze the input layer for N epochs. This is commonly used when
            the input layer contains language models.
            Default: 0, no freeze.
        max_grad_norm: float
            Clip the gradient by the max_grad_norm to ensure stability.
            Default: None, no clip.
        grad_norm_type: float
            Norm type for the gradient clip
            Default: 2.0
        """
        # Check the correctness of configurations.
        if self.evaluator is not None:
            assert val_loader is not None, \
                    "The evaluator is provided but validation set is not provided."

        # computation graph will be changed during training.
        on_cpu = self.device == th.device('cpu')
        model = DistributedDataParallel(self._model, device_ids=None if on_cpu else [self.device],
                                        output_device=None if on_cpu else self.device,
                                        find_unused_parameters=True,
                                        static_graph=False)
        device = model.device
        data = train_loader.data

        # training loop
        dur = []
        total_steps = 0

        sys_tracker.check('start training')
        g = data.g # dgl.DistGraph
        for epoch in range(num_epochs):
            t0 = time.time()
            rt_profiler.start_record()

            use_gnn = self._model.em_order_gnn_first
            # `use_gnn`` determines which part to train, if `em_order_gnn_first`
            # 1st round: train GNN, fix LM; 2nd round: train LM fix gnn
            for _ in range(2):
                stage_start_time = time.time()
                part_to_train = 'gnn' if use_gnn else 'lm'
                self._model.toggle(part_to_train, data)

                self._fit_one_epoch(use_gnn, model, g, data, train_loader, val_loader, test_loader,
                                    device, rt_profiler,
                                    epoch, total_steps, use_mini_batch_infer,
                                    save_model_path, save_model_frequency, max_grad_norm,
                                    grad_norm_type)
                stage_finish_time = time.time()
                if get_rank() == 0:
                    logging.info("Epoch %d: %s takes %.2f seconds",
                                 epoch, part_to_train, stage_finish_time-stage_start_time)
                use_gnn = not use_gnn

            # early_stop, exit training
            if self.early_stop is True:
                break

            epoch_time = time.time() - t0
            dur.append(epoch_time)

        rt_profiler.save_profile()
        print_mem(device)
        if get_rank() == 0 and self.evaluator is not None:
            output = {'best_test_score': self.evaluator.best_test_score,
                       'best_val_score': self.evaluator.best_val_score,
                       'peak_mem_alloc_MB': th.cuda.max_memory_allocated(device) / 1024 / 1024}
            self.log_params(output)

            if save_perf_results_path is not None:
                self.save_model_results_to_file(self.evaluator.best_test_score,
                                                save_perf_results_path)

    def _fit_one_epoch(self, use_gnn, model, g, data, train_loader,
                       val_loader, test_loader, device, profiler,
                       epoch, total_steps,
                       use_mini_batch_infer=True,
                       save_model_path=None,
                       save_model_frequency=-1,
                       max_grad_norm=None, grad_norm_type=2.0):
        """Fit model for one epoch
        """
        def _prepare_batch(input_nodes, seeds, blocks, is_labeled=True):
            """Prepare a batch of graph data from the data loader, by retrieving features,
            moving blocks to device, and get labels if `is_labeled` is True.
            """
            if not isinstance(input_nodes, dict):
                assert len(g.ntypes) == 1
                input_nodes = {g.ntypes[0]: input_nodes}

            nfeat_fields = train_loader.node_feat_fields
            input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)
            profiler.record('train_node_feats')
            lbl = None
            if is_labeled:
                label_field = train_loader.label_field
                lbl = data.get_node_feats(seeds, label_field, device)
            blocks = [block.to(device) for block in blocks]
            profiler.record('train_graph2GPU')
            return input_nodes, input_feats, blocks, lbl

        profiler.start_record()
        for i, batch in enumerate(train_loader):
            if isinstance(train_loader, GSgnnNodeSemiSupDataLoader):
                # semi-supervised setting
                input_nodes, input_feats, blocks, lbl = _prepare_batch(*batch[0])
                input_nodes_u, input_feats_u, blocks_u, _ = _prepare_batch(
                    *batch[1], is_labeled=False)
            else:
                # supervised setting, no unlabeled data
                input_nodes, input_feats, blocks, lbl = _prepare_batch(*batch)
                input_nodes_u = input_feats_u = blocks_u = None
            profiler.record('train_sample')
            total_steps += 1
            batch_tic = time.time()
            # Run forward function to compute loss:
            loss = model(blocks, input_feats, None, lbl, input_nodes, use_gnn=use_gnn,
                         no_pl=epoch < self.num_pretrain_epochs,
                         blocks_u=blocks_u, node_feats_u=input_feats_u, edge_feats_u=None,
                         input_nodes_u=input_nodes_u)
            profiler.record('train_forward')
            module = model.module if is_distributed() else model
            self.optimizer.zero_grad(optimize_sparse_params=module.training_sparse_embed)
            loss.backward()
            profiler.record('train_backward')
            self.optimizer.step(optimize_sparse_params=module.training_sparse_embed)
            profiler.record('train_step')

            if max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, grad_norm_type)
            self.log_metric("Train loss", loss.item(), total_steps)

            if i % 20 == 0 and get_rank() == 0:
                rt_profiler.print_stats()
                logging.info("Part %d | Epoch %05d | Batch %03d | Loss: %.4f | Time: %.4f",
                             get_rank(), epoch, i,  loss.item(), time.time() - batch_tic)

            val_score = None
            if self.can_do_validation(val_loader) and self.evaluator.do_eval(total_steps):
                val_score = self.eval(model.module, val_loader, test_loader,
                                      use_mini_batch_infer, total_steps, return_proba=False)

                if self.evaluator.do_early_stop(val_score):
                    self.early_stop = True

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
                        val_score = self.eval(model.module, val_loader, test_loader,
                                            use_mini_batch_infer, total_steps, return_proba=False)
                # We will save the best model when
                # 1. If not do evaluation, we will keep the latest K models.
                # 2. If do evaluaiton, we need to follow the guidance of validation score.
                self.save_topk_models(model, epoch, i, val_score, save_model_path)

            rt_profiler.record('train_eval')
            # early_stop, exit current interation.
            if self.early_stop is True:
                break

        # ------- end of an epoch -------

        barrier()

        val_score = None
        # do evaluation and model saving after each epoch if can
        if self.can_do_validation(val_loader):
            val_score = self.eval(model.module, val_loader, test_loader,
                                  use_mini_batch_infer, total_steps, return_proba=False)
            if self.evaluator.do_early_stop(val_score):
                self.early_stop = True

        # After each epoch, check to save the top k models. If has validation score, will save
        # the best top k. But if no validation, will either save the last k model or all models
        # depends on the setting of top k. To show this is after epoch save, set the iteration
        # to be None, so that we can have a determistic model folder name for testing and debug.
        self.save_topk_models(model, epoch, None, val_score, save_model_path)
        # make sure saving model finishes properly before the main process kills this training
        barrier()
