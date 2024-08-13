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

    GraphStorm trainer for link prediction
"""
import time
import resource
import logging
import torch as th
from torch.nn.parallel import DistributedDataParallel
import dgl

from ..model.lp_gnn import GSgnnLinkPredictionModelInterface
from ..model.lp_gnn import lp_mini_batch_predict
from ..model.gnn_with_reconstruct import GNNEncoderWithReconstructedEmbed
from ..model import (do_full_graph_inference,
                     do_mini_batch_inference,
                     GSgnnModelBase,
                     GSgnnModel)
from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker, rt_profiler, print_mem, get_rank
from ..utils import barrier, is_distributed

class GSgnnLinkPredictionTrainer(GSgnnTrainer):
    """ Trainer for link prediction tasks.

    ``GSgnnLinkPredictionTrainer`` is a high-level trainer wrapper that can be used directly
    to train a link prediction model. ``GSgnnLinkPredictionTrainer`` define two main functions:

    * ``fit()``: performs the training for the model provided to this trainer
      when the object is initialized, and;
    * ``eval()``: evaluates the provided model against test and validation dataset.

    Example
    -------

    .. code:: python

        from graphstorm.dataloading import GSgnnLinkPredictionDataLoader
        from graphstorm.dataset import GSgnnData
        from graphstorm.model import GSgnnLinkPredictionModel
        from graphstorm.trainer import GSgnnLinkPredictionTrainer

        lp_data = GSgnnData("...")
        target_idx = lp_data.get_edge_train_set([("src_ntype1", "etype1", "dst_ntype1)])
        train_loader = GSgnnLinkPredictionDataLoader(
            lp_data, target_idx, fanout=[10], batch_size=1024,
            num_negative_edges=10, node_feats="feat", train_task=True)
        model = GSgnnLinkPredictionModel(alpha_l2norm=0.0)

        trainer = GSgnnLinkPredictionTrainer(model)

        trainer.fit(train_loader, num_epochs=2)

    Parameters
    ----------
    model : GSgnnLinkPredictionModelBase
        The GNN model for link prediction, which could be a model class inherited from the
        ``GSgnnLinkPredictionModelBase``, or a model class that inherits both the
        ``GSgnnModelBase`` and the ``GSgnnLinkPredictionModelInterface`` class.
    topk_model_to_save : int
        The top `K` model to be saved based on evaluation results. Default: 1.
    """
    def __init__(self, model, topk_model_to_save=1):
        super(GSgnnLinkPredictionTrainer, self).__init__(model, topk_model_to_save)
        assert isinstance(model, GSgnnLinkPredictionModelInterface) \
                and isinstance(model, GSgnnModelBase), \
                "The input model is not an edge model. Please implement GSgnnEdgeModelBase."

    def fit(self, train_loader, num_epochs,
            val_loader=None,            # pylint: disable=unused-argument
            test_loader=None,           # pylint: disable=unused-argument
            use_mini_batch_infer=True,      # pylint: disable=unused-argument
            save_model_path=None,
            save_model_frequency=-1,
            save_perf_results_path=None,
            edge_mask_for_gnn_embeddings='train_mask',
            freeze_input_layer_epochs=0,
            max_grad_norm=None,
            grad_norm_type=2.0):
        """ Fit function for link prediction.

        This function performs the training for the given link prediction model.
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
        train_loader: GSgnnLinkPredictionDataLoader
            LinkPrediction dataloader for mini-batch sampling the training set.
        num_epochs: int
            The max number of epochs used to train the model.
        val_loader: GSgnnLinkPredictionDataLoader
            LinkPrediction dataloader for mini-batch sampling the validation set. Default: None.
        test_loader: GSgnnLinkPredictionDataLoader
            LinkPrediction dataloader for mini-batch sampling the test set. Default: None.
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
        edge_mask_for_gnn_embeddings : str
            The mask that indicates the edges used for computing GNN embeddings for model
            evaluation. By default, it uses the edges in the training graph to compute
            GNN embeddings for evaluation. Default: "train_mask".
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
        if not use_mini_batch_infer:
            assert isinstance(self._model, GSgnnModel), \
                    "Only GSgnnModel supports full-graph inference."

        # assert not use GNNEncoderWithReconstructedEmbed when use_mini_batch_infer=True
        if self._model.gnn_encoder is not None:
            assert not (isinstance(self._model.gnn_encoder, GNNEncoderWithReconstructedEmbed) and \
                use_mini_batch_infer), 'GraphStorm GNNEncoderWithReconstructedEmbed encoder' + \
                    ' dose not support mini-batch inference. Please set ' + \
                        'use_mini_batch_infer to be false.'

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
        for epoch in range(num_epochs):
            model.train()
            epoch_start = time.time()

            if freeze_input_layer_epochs <= epoch:
                self._model.unfreeze_input_encoder()
            # TODO(xiangsx) Support unfreezing gnn encoder and decoder

            rt_profiler.start_record()
            batch_tic = time.time()
            for i, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(train_loader):
                rt_profiler.record('train_sample')
                total_steps += 1

                if not isinstance(input_nodes, dict):
                    assert len(pos_graph.ntypes) == 1
                    input_nodes = {pos_graph.ntypes[0]: input_nodes}
                nfeat_fields = train_loader.node_feat_fields
                input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)
                if train_loader.pos_graph_edge_feat_fields is not None:
                    input_edges = {etype: pos_graph.edges[etype].data[dgl.EID] \
                        for etype in pos_graph.canonical_etypes}
                    pos_graph_feats = data.get_edge_feats(input_edges,
                                                          train_loader.pos_graph_edge_feat_fields,
                                                          device)
                else:
                    pos_graph_feats = None
                rt_profiler.record('train_node_feats')

                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [blk.to(device) for blk in blocks]
                rt_profiler.record('train_graph2GPU')

                # TODO(zhengda) we don't support edge features for now.
                loss = model(blocks, pos_graph, neg_graph,
                             node_feats=input_feats,
                             edge_feats=None,
                             pos_edge_feats=pos_graph_feats,
                             input_nodes=input_nodes)
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
                    logging.info("Epoch %05d | Batch %03d | Train Loss: %.4f | Time: %.4f",
                                 epoch, i, loss.item(), time.time() - batch_tic)

                val_score = None
                if self.can_do_validation(val_loader) and self.evaluator.do_eval(total_steps):
                    val_score = self.eval(model.module if is_distributed() else model,
                                          data, val_loader, test_loader, total_steps,
                                          edge_mask_for_gnn_embeddings, use_mini_batch_infer)
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
                                                data, val_loader, test_loader, total_steps,
                                                edge_mask_for_gnn_embeddings, use_mini_batch_infer)
                    # We will save the best model when
                    # 1. If not do evaluation, we will keep the latest K models.
                    # 2. If do evaluaiton, we need to follow the guidance of validation score.
                    self.save_topk_models(model, epoch, i, val_score, save_model_path)

                batch_tic = time.time()
                rt_profiler.record('train_eval')
                # early_stop, exit current interation.
                if early_stop is True:
                    break

            # ------- end of an epoch -------

            barrier()
            epoch_time = time.time() - epoch_start
            if get_rank() == 0:
                logging.info("Epoch %d take %.3f seconds", epoch, epoch_time)

            val_score = None
            # do evaluation and model saving after each epoch if can
            if self.can_do_validation(val_loader):
                val_score = self.eval(model.module if is_distributed() else model,
                                      data, val_loader, test_loader, total_steps,
                                      edge_mask_for_gnn_embeddings, use_mini_batch_infer)

                if self.evaluator.do_early_stop(val_score):
                    early_stop = True

            # After each epoch, check to save the top k models. If has validation score, will save
            # the best top k. But if no validation, will either save the last k model or all models
            # depends on the setting of top k. To show this is after epoch save, set the iteration
            # to be None, so that we can have a determistic model folder name for testing and debug.
            self.save_topk_models(model, epoch, None, val_score, save_model_path)
            rt_profiler.print_stats()
            # make sure saving model finishes properly before the main process kills this training
            barrier()

            # early_stop, exit training
            if early_stop is True:
                break

        rt_profiler.save_profile()
        print_mem(device)
        if get_rank() == 0 and self.evaluator is not None:
            output = {'best_test_score': self.evaluator.best_test_score,
                       'best_val_score':self.evaluator.best_val_score,
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

    def eval(self, model, data, val_loader, test_loader,
             total_steps, edge_mask_for_gnn_embeddings,
             use_mini_batch_infer=False):
        """ Do model evaluation using the validation set, or test set if provided.

        Parameters
        ----------
        model : GSgnnLinkPredictionModelBase
            The GNN model for link prediction, which could be a model class inherited from the
            ``GSgnnLinkPredictionModelBase``, or a model class that inherits both the
            ``GSgnnModelBase`` and the ``GSgnnLinkPredictionModelInterface``.
        data : GSgnnData
            The ``GSgnnData`` associated with dataloaders.
        val_loader: GSgnnLinkPredictionDataLoader
            Link prediction dataloader for mini-batch sampling the validation set. Default: None.
        test_loader: GSgnnLinkPredictionDataLoader
            Link prediction dataloader for mini-batch sampling the test set. Default: None.
        total_steps: int
            The total number of iterations.
        edge_mask_for_gnn_embeddings : str
            The mask that indicates the edges used for computing GNN embeddings for model
            evaluation.
        use_mini_batch_infer: bool
            Whether to use mini-batch for inference. Default: True.

        Returns
        -------
        val_score: dict
            Validation scores of differnet metrics in the format of {metric: val_score}.
        """
        test_start = time.time()
        sys_tracker.check('before prediction')
        model.eval()

        if use_mini_batch_infer:
            emb = do_mini_batch_inference(model, data, fanout=val_loader.fanout,
                                          edge_mask=edge_mask_for_gnn_embeddings,
                                          task_tracker=self.task_tracker)
        else:
            emb = do_full_graph_inference(model, data, fanout=val_loader.fanout,
                                          edge_mask=edge_mask_for_gnn_embeddings,
                                          task_tracker=self.task_tracker)
        sys_tracker.check('compute embeddings')
        val_scores = lp_mini_batch_predict(model, emb, val_loader, self.device) \
            if val_loader is not None else None
        sys_tracker.check('after_val_score')
        if test_loader is not None:
            test_scores = lp_mini_batch_predict(model, emb, test_loader, self.device)
        else:
            test_scores = None
        sys_tracker.check('after_test_score')
        val_score, test_score = self.evaluator.evaluate(
            val_scores, test_scores, total_steps)
        sys_tracker.check('evaluate validation/test')
        model.train()

        if get_rank() == 0:
            self.log_print_metrics(val_score=val_score,
                                   test_score=test_score,
                                   dur_eval=time.time() - test_start,
                                   total_steps=total_steps)
        return val_score
