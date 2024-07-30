"""
    Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GraphStorm trainer for multi-task learning.
"""

import time
import resource
import logging
import torch as th
from torch.nn.parallel import DistributedDataParallel
import dgl

from ..config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                      BUILTIN_TASK_NODE_REGRESSION,
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      BUILTIN_TASK_EDGE_REGRESSION,
                      BUILTIN_TASK_LINK_PREDICTION,
                      BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
from ..model import (do_full_graph_inference,
                     do_mini_batch_inference,
                     GSgnnModelBase, GSgnnModel,
                     GSgnnMultiTaskModelInterface,
                     multi_task_mini_batch_predict,
                     gen_emb_for_nfeat_reconstruct)
from ..model.lp_gnn import run_lp_mini_batch_predict
from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker, rt_profiler, print_mem, get_rank
from ..utils import barrier, is_distributed

def prepare_node_mini_batch(data, task_info, mini_batch, device):
    """ Prepare mini-batch for node classification and regression tasks.

        The input is a mini-batch sampled by a node sampler.
        The output ia a prepared input following the
        input arguments of GSgnnNodeModelInterface.forward.

    Parameters
    ----------
    data: GSgnnData
        Graph data
    task_info: TaskInfo
        Task meta information
    mini_batch: tuple
        Mini-batch info
    device: torch.device
        Device

    Return
    ------
    tuple: mini-batch
    """
    g = data.g
    input_nodes, seeds, blocks = mini_batch
    if not isinstance(input_nodes, dict):
        # This happens on a homogeneous graph.
        assert len(g.ntypes) == 1, \
            "The graph should be a homogeneous graph, " \
            f"but it has multiple node types {g.ntypes}"
        input_nodes = {g.ntypes[0]: input_nodes}

    nfeat_fields = task_info.dataloader.node_feat_fields
    label_field = task_info.dataloader.label_field
    input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)
    lbl = data.get_node_feats(seeds, label_field, device)
    blocks = [block.to(device) for block in blocks] \
        if blocks is not None else None

    # Order follow GSgnnNodeModelInterface.forward
    # TODO: we don't support edge features for now.
    return (blocks, input_feats, None, lbl, input_nodes)

def prepare_edge_mini_batch(data, task_info, mini_batch, device):
    """ Prepare mini-batch for edge classification and regression tasks.

        The input is a mini-batch sampled by an edge sampler.
        The output ia a prepared input following the
        input arguments of GSgnnEdgeModelInterface.forward.

    Parameters
    ----------
    data: GSgnnData
        Graph data
    task_info: TaskInfo
        Task meta information
    mini_batch: tuple
        Mini-batch info
    device: torch.device
        Device

    Return
    ------
    tuple: mini-batch
    """
    input_nodes, batch_graph, blocks = mini_batch
    if not isinstance(input_nodes, dict):
        assert len(batch_graph.ntypes) == 1, \
            "The graph should be a homogeneous graph, " \
            f"but it has multiple node types {batch_graph.ntypes}"
        input_nodes = {batch_graph.ntypes[0]: input_nodes}

    nfeat_fields = task_info.dataloader.node_feat_fields
    node_feats = data.get_node_feats(input_nodes, nfeat_fields, device)

    if task_info.dataloader.decoder_edge_feat_fields is not None:
        # There are edge features used in decoder.
        input_edges = {etype: batch_graph.edges[etype].data[dgl.EID] \
                for etype in batch_graph.canonical_etypes}
        edge_decoder_feats = \
            data.get_edge_feats(input_edges,
                                task_info.dataloader.decoder_edge_feat_fields,
                                device)
        edge_decoder_feats = {etype: feat.to(th.float32) \
            for etype, feat in edge_decoder_feats.items()}
    else:
        edge_decoder_feats = None

    # retrieving seed edge id from the graph to find labels
    assert len(batch_graph.etypes) == 1, \
        "Edge classification/regression tasks only support " \
        "conducting prediction on one edge type."
    target_etype = batch_graph.canonical_etypes[0]
    seeds = batch_graph.edges[target_etype].data[dgl.EID]
    label_field = task_info.dataloader.label_field
    lbl = data.get_edge_feats({target_etype: seeds}, label_field, device)

    blocks = [block.to(device) for block in blocks] \
        if blocks is not None else None
    batch_graph = batch_graph.to(device)
    rt_profiler.record('train_graph2GPU')

    # Order follow GSgnnEdgeModelInterface.forward
    # TODO(zhengda) we don't support edge features for now.
    return (blocks, batch_graph, node_feats, None,
            edge_decoder_feats, lbl, input_nodes)

def prepare_link_predict_mini_batch(data, task_info, mini_batch, device):
    """ Prepare mini-batch for link prediction tasks.

        The input is a mini-batch sampled by an edge sampler.
        The output ia a prepared input following the
        input arguments of GSgnnLinkPredictionModelInterface.forward.

    Parameters
    ----------
    data: GSgnnData
        Graph data
    task_info: TaskInfo
        Task meta information
    mini_batch: tuple
        Mini-batch info
    device: torch.device
        Device

    Return
    ------
    tuple: mini-batch
    """
    input_nodes, pos_graph, neg_graph, blocks = mini_batch

    if not isinstance(input_nodes, dict):
        assert len(pos_graph.ntypes) == 1, \
            "The graph should be a homogeneous graph, " \
            f"but it has multiple node types {pos_graph.ntypes}"
        input_nodes = {pos_graph.ntypes[0]: input_nodes}

    nfeat_fields = task_info.dataloader.node_feat_fields
    node_feats = data.get_node_feats(input_nodes, nfeat_fields, device)

    if task_info.dataloader.pos_graph_edge_feat_fields is not None:
        input_edges = {etype: pos_graph.edges[etype].data[dgl.EID] \
            for etype in pos_graph.canonical_etypes}
        pos_graph_feats = data.get_edge_feats(input_edges,
                                                task_info.dataloader.pos_graph_edge_feat_fields,
                                                device)
    else:
        pos_graph_feats = None

    pos_graph = pos_graph.to(device)
    neg_graph = neg_graph.to(device)
    blocks = [blk.to(device) for blk in blocks] \
        if blocks is not None else None

    # follow the interface of GSgnnLinkPredictionModelInterface.forward
    return (blocks, pos_graph, neg_graph, node_feats, None, \
            pos_graph_feats, None, input_nodes)

def prepare_reconstruct_node_feat(data, task_info, mini_batch, device):
    """ Prepare mini-batch for node feature reconstruction.

        The input is a mini-batch sampled by a node sampler.
        The output ia a prepared input following the
        input arguments of GSgnnNodeModelInterface.forward.

    Parameters
    ----------
    data: GSgnnData
        Graph data
    task_info: TaskInfo
        Task meta information
    mini_batch: tuple
        Mini-batch info
    device: torch.device
        Device

    Return
    ------
    tuple: mini-batch
    """
    # same are preparing node regression data
    # Note: We may add some argumentation in the future
    # So keep a different prepare func for node feature reconstruction.
    return prepare_node_mini_batch(data, task_info, mini_batch, device)

class GSgnnMultiTaskLearningTrainer(GSgnnTrainer):
    r""" A trainer for multi-task learning

    This class is used to train models for multi-task learning.

    It makes use of the functions provided by `GSgnnTrainer`
    to define two main functions: `fit` that performs the training
    for the model that is provided when the object is created,
    and `eval` that evaluates a provided model against test and
    validation data.

    Parameters
    ----------
    model : GSgnnMultiTaskModel
        The GNN model for prediction.
    topk_model_to_save : int
        The top K model to save.
    """
    def __init__(self, model, topk_model_to_save=1):
        super(GSgnnMultiTaskLearningTrainer, self).__init__(model, topk_model_to_save)
        assert isinstance(model, GSgnnMultiTaskModelInterface) \
            and isinstance(model, GSgnnModelBase), \
                "The input model is not a GSgnnModel model "\
                "or not implement the GSgnnMultiTaskModelInterface." \
                "Please implement GSgnnModelBase."

    def _prepare_mini_batch(self, data, task_info, mini_batch, device):
        """ prepare mini batch for a single task

        Parameters
        ----------
        data: GSgnnData
            Graph data
        model: GSgnnModel
            Model
        task_info: TaskInfo
            Task meta information
        mini_batch: tuple
            Mini-batch info
        device: torch.device
            Device

        Return
        ------
        tuple: mini-batch
        """
        if task_info.task_type in \
            [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
            return prepare_node_mini_batch(data,
                                           task_info,
                                           mini_batch,
                                           device)
        elif task_info.task_type in \
            [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
            return prepare_edge_mini_batch(data,
                                           task_info,
                                           mini_batch,
                                           device)
        elif task_info.task_type == BUILTIN_TASK_LINK_PREDICTION:
            return prepare_link_predict_mini_batch(data,
                                                   task_info,
                                                   mini_batch,
                                                   device)
        elif task_info.task_type == BUILTIN_TASK_RECONSTRUCT_NODE_FEAT:
            return prepare_reconstruct_node_feat(data,
                                                 task_info,
                                                 mini_batch,
                                                 device)
        else:
            raise TypeError(f"Unknown task {task_info}", )

    # pylint: disable=unused-argument
    def fit(self, train_loader,
            num_epochs,
            val_loader=None,
            test_loader=None,
            use_mini_batch_infer=True,
            save_model_path=None,
            save_model_frequency=-1,
            save_perf_results_path=None,
            freeze_input_layer_epochs=0,
            max_grad_norm=None,
            grad_norm_type=2.0):
        """ The fit function for multi-task learning.

        Performs the training for `self.model`. Iterates over all the tasks
        and run one mini-batch for each task in an iteration. The loss will be
        accumulated. Performs the backwards step using `self.optimizer`.
        If an evaluator has been assigned to the trainer, it will run evaluation
        at the end of every epoch.

        Parameters
        ----------
        train_loader : GSgnnMultiTaskDataLoader
            The mini-batch sampler for training.
        num_epochs : int
            The max number of epochs to train the model.
        val_loader : GSgnnMultiTaskDataLoader
            The mini-batch sampler for computing validation scores. The validation scores
            are used for selecting models.
        test_loader : GSgnnMultiTaskDataLoader
            The mini-batch sampler for computing test scores.
        use_mini_batch_infer : bool
            Whether or not to use mini-batch inference.
        save_model_path : str
            The path where the model is saved.
        save_model_frequency : int
            The number of iteration to train the model before saving the model.
        save_perf_results_path : str
            The path of the file where the performance results are saved.
            TODO(xiangsx): Add support for saving performance results on disk.
            Reserved for future.
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
        if not use_mini_batch_infer:
            assert isinstance(self._model, GSgnnModel), \
                    "Only GSgnnModel supports full-graph inference."

        # with freeze_input_layer_epochs is 0, computation graph will not be changed.
        on_cpu = self.device == th.device('cpu')
        if is_distributed():
            model = DistributedDataParallel(self._model,
                                            device_ids=None if on_cpu else [self.device],
                                            output_device=None if on_cpu else self.device,
                                            find_unused_parameters=True,
                                            static_graph=False)
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
        sys_tracker.check('start training')
        for epoch in range(num_epochs):
            model.train()
            epoch_start = time.time()
            if freeze_input_layer_epochs <= epoch:
                self._model.unfreeze_input_encoder()
            # TODO(xiangsx) Support unfreezing gnn encoder and decoder

            rt_profiler.start_record()
            batch_tic = time.time()
            for i, task_mini_batches in enumerate(train_loader):
                rt_profiler.record('train_sample')
                total_steps += 1

                mini_batches = []
                for (task_info, mini_batch) in task_mini_batches:
                    mini_batches.append((task_info, \
                        self._prepare_mini_batch(data, task_info, mini_batch, device)))

                loss, task_losses = model(mini_batches)

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
                    per_task_loss = {}
                    for mini_batch, task_loss in zip(mini_batches, task_losses):
                        task_info, _ = mini_batch
                        per_task_loss[task_info.task_id] = task_loss[0].item()
                    logging.info("Epoch %05d | Batch %03d | Train Loss: %.4f | Time: %.4f",
                                 epoch, i, loss.item(), time.time() - batch_tic)
                    logging.debug("Per task Loss: %s", per_task_loss)

                val_score = None
                if self.can_do_validation(val_loader) and self.evaluator.do_eval(total_steps):
                    val_score = self.eval(model.module if is_distributed() else model,
                                          data, val_loader, test_loader, total_steps)
                    # TODO(xiangsx): Add early stop support

                # Every n iterations, save the model and keep
                # the last k models.
                # TODO(xiangsx): support saving the best top k model.
                if save_model_frequency > 0 and \
                    total_steps % save_model_frequency == 0 and \
                    total_steps != 0:
                    if val_score is None:
                        # not in the same eval_frequncy iteration
                        if self.can_do_validation(val_loader):
                            # for model saving, force to do evaluation if can
                            val_score = self.eval(model.module if is_distributed() else model,
                                                data, val_loader, test_loader, total_steps)
                    # We will save the best model when
                    # 1. There is no evaluation, we will keep the
                    #    latest K models.
                    # 2. (TODO) There is evaluaiton, we need to follow the
                    #    guidance of validation score.
                    # So here reset val_score to be None
                    val_score = None
                    self.save_topk_models(model, epoch, i, val_score, save_model_path)

                batch_tic = time.time()
                rt_profiler.record('train_eval')

            # ------- end of an epoch -------

            barrier()
            epoch_time = time.time() - epoch_start
            if get_rank() == 0:
                logging.info("Epoch %d take %.3f seconds", epoch, epoch_time)

            val_score = None
            # do evaluation and model saving after each epoch if can
            if self.can_do_validation(val_loader):
                val_score = self.eval(model.module if is_distributed() else model,
                                      data, val_loader, test_loader, total_steps)

            # After each epoch, check to save the top k models.
            # Will either save the last k model or all models
            # depends on the setting of top k.
            self.save_topk_models(model, epoch, None, None, save_model_path)
            rt_profiler.print_stats()
            # make sure saving model finishes properly before the main process kills this training
            barrier()

        rt_profiler.save_profile()
        print_mem(device)
        if get_rank() == 0 and self.evaluator is not None:
            # final evaluation
            output = {'best_test_score': self.evaluator.best_test_score,
                       'best_val_score':self.evaluator.best_val_score,
                       'peak_GPU_mem_alloc_MB': th.cuda.max_memory_allocated(device) / 1024 / 1024,
                       'peak_RAM_mem_alloc_MB': \
                           resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       'best validation iteration': \
                           self.evaluator.best_iter_num,
                       'best model path': \
                           self.get_best_model_path() if save_model_path is not None else None}
            self.log_params(output)

    def eval(self, model, data, mt_val_loader, mt_test_loader, total_steps,
        use_mini_batch_infer=False, return_proba=True):
        """ do the model evaluation using validation and test sets

        Parameters
        ----------
        model : Pytorch model
            The GNN model.
        data : GSgnnData
            The training dataset
        mt_val_loader: GSgnnMultiTaskDataLoader
            The dataloader for validation data
        mt_test_loader : GSgnnMultiTaskDataLoader
            The dataloader for test data.
        total_steps: int
            Total number of iterations.
        use_mini_batch_infer: bool
            Whether do mini-batch inference
        return_proba: bool
            Whether to return all the predictions or the maximum prediction.

        Returns
        -------
        dict: validation score
        """
        test_start = time.time()
        sys_tracker.check('before prediction')
        model.eval()

        if mt_val_loader is None and mt_test_loader is None:
            # no need to do validation and test
            # do nothing.
            return None

        val_dataloaders = mt_val_loader.dataloaders \
            if mt_val_loader is not None else None
        test_dataloaders = mt_test_loader.dataloaders \
            if mt_test_loader is not None else None
        task_infos = mt_val_loader.task_infos \
            if mt_val_loader is not None else mt_test_loader.task_infos
        if val_dataloaders is None:
            val_dataloaders = [None] * len(task_infos)
        if test_dataloaders is None:
            test_dataloaders = [None] * len(task_infos)

        # All the tasks share the same GNN encoder so the fanouts are same
        # for different tasks.
        fanout = None
        if mt_val_loader is not None:
            for task_fanout in mt_val_loader.fanout:
                if task_fanout is not None:
                    fanout = task_fanout
                    break
        else:
            for task_fanout in mt_test_loader.fanout:
                if task_fanout is not None:
                    fanout = task_fanout
                    break
        assert fanout is not None, \
            "There is no validation dataloader.eval() function should not be called"

        # Node prediction and edge prediction
        # do not have information leakage problem
        predict_tasks = []
        predict_val_loaders = []
        predict_test_loaders = []
        # For link prediction tasks, we need to
        # exclude valid and test edges during message
        # passk
        lp_tasks = []
        lp_val_loaders = []
        lp_test_loaders = []
        # For node feature reconstruction tasks,
        # we need to avoid self-loop in the last
        # GNN layer
        nfeat_recon_tasks = []
        nfeat_recon_val_loaders = []
        nfeat_recon_test_loaders = []

        for val_loader, test_loader, task_info \
            in zip(val_dataloaders, test_dataloaders, task_infos):

            if val_loader is None and test_loader is None:
                # For this task, these is no need to do compute test or val score
                # skip this task
                continue

            if task_info.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION,
                                       BUILTIN_TASK_NODE_REGRESSION,
                                       BUILTIN_TASK_EDGE_CLASSIFICATION,
                                       BUILTIN_TASK_EDGE_REGRESSION]:
                predict_tasks.append(task_info)
                predict_val_loaders.append(val_loader)
                predict_test_loaders.append(test_loader)

            if task_info.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
                lp_tasks.append(task_info)
                lp_val_loaders.append(val_loader)
                lp_test_loaders.append(test_loader)

            if task_info.task_type in [BUILTIN_TASK_RECONSTRUCT_NODE_FEAT]:
                nfeat_recon_tasks.append(task_info)
                nfeat_recon_val_loaders.append(val_loader)
                nfeat_recon_test_loaders.append(test_loader)

        def gen_embs(edge_mask=None):
            """ Compute node embeddings
            """
            if use_mini_batch_infer:
                emb = do_mini_batch_inference(model, data,
                                              fanout=fanout,
                                              edge_mask=edge_mask,
                                              task_tracker=self.task_tracker)
            else:
                emb = do_full_graph_inference(model, data,
                                              fanout=fanout,
                                              edge_mask=edge_mask,
                                              task_tracker=self.task_tracker)
            return emb

        embs = None
        val_results = None
        test_results = None
        if len(predict_tasks) > 0:
            # do validation and test for prediciton tasks.
            sys_tracker.check('compute embeddings')
            embs = gen_embs()
            val_results = \
                multi_task_mini_batch_predict(
                    model,
                    emb=embs,
                    dataloaders=predict_val_loaders,
                    task_infos=predict_tasks,
                    device=self.device,
                    return_proba=return_proba,
                    return_label=True) \
                if len(predict_val_loaders) > 0 else None

            test_results = \
                multi_task_mini_batch_predict(
                    model,
                    emb=embs,
                    dataloaders=predict_test_loaders,
                    task_infos=predict_tasks,
                    device=self.device,
                    return_proba=return_proba,
                    return_label=True) \
                if len(predict_test_loaders) > 0 else None

        if len(lp_tasks) > 0:
            for lp_val_loader, lp_test_loader, task_info \
                in zip(lp_val_loaders, lp_test_loaders, lp_tasks):
                # For link prediction, do evaluation task
                # by task.
                lp_test_embs = gen_embs(edge_mask=task_info.task_config.train_mask)
                # normalize the node embedding if needed.
                # we can do inplace normalization as embeddings are generated
                # per lp task.
                lp_test_embs = model.normalize_task_node_embs(task_info.task_id,
                                                              lp_test_embs,
                                                              inplace=True)

                decoder = model.task_decoders[task_info.task_id]
                val_scores = run_lp_mini_batch_predict(decoder,
                                                       lp_test_embs,
                                                       lp_val_loader,
                                                       self.device) \
                    if lp_val_loader is not None else None
                test_scores = run_lp_mini_batch_predict(decoder,
                                                        lp_test_embs,
                                                        lp_test_loader,
                                                        self.device) \
                    if lp_test_loader is not None else None

                if val_results is not None:
                    val_results[task_info.task_id] = val_scores
                else:
                    val_results = {task_info.task_id: val_scores}
                if test_results is not None:
                    test_results[task_info.task_id] = test_scores
                else:
                    test_results = {task_info.task_id: test_scores}

        if len(nfeat_recon_tasks) > 0:
            def nfrecon_gen_embs(skip_last_self_loop=False, node_embs=embs):
                """ Generate node embeddings for node feature reconstruction
                """
                if skip_last_self_loop is True:
                    # Turn off the last layer GNN's self-loop
                    # to compute node embeddings.
                    model.gnn_encoder.skip_last_selfloop()
                    new_embs = gen_embs()
                    model.gnn_encoder.reset_last_selfloop()
                    return new_embs
                else:
                    # If skip_last_self_loop is False
                    # we will not change the way we compute
                    # node embeddings.
                    if node_embs is not None:
                        # The embeddings have been computed
                        return node_embs
                    else:
                        return gen_embs()

            nfeat_embs = gen_emb_for_nfeat_reconstruct(model, nfrecon_gen_embs)

            nfeat_recon_val_results = \
                multi_task_mini_batch_predict(
                    model,
                    emb=nfeat_embs,
                    dataloaders=nfeat_recon_val_loaders,
                    task_infos=nfeat_recon_tasks,
                    device=self.device,
                    return_proba=return_proba,
                    return_label=True) \
                if len(nfeat_recon_val_loaders) > 0 else None

            nfeat_recon_test_results = \
                multi_task_mini_batch_predict(
                    model,
                    emb=nfeat_embs,
                    dataloaders=nfeat_recon_test_loaders,
                    task_infos=nfeat_recon_tasks,
                    device=self.device,
                    return_proba=return_proba,
                    return_label=True) \
                if len(nfeat_recon_test_loaders) > 0 else None

            if val_results is None:
                val_results = nfeat_recon_val_results
            else:
                if nfeat_recon_val_results is not None:
                    val_results.update(nfeat_recon_val_results)

            if test_results is None:
                test_results = nfeat_recon_test_results
            else:
                if nfeat_recon_test_results is not None:
                    test_results.update(nfeat_recon_test_results)


        sys_tracker.check('after_test_score')
        val_score, test_score = self.evaluator.evaluate(
                val_results, test_results, total_steps)
        sys_tracker.check('evaluate validation/test')
        model.train()

        if get_rank() == 0:
            self.log_print_metrics(val_score=val_score,
                                   test_score=test_score,
                                   dur_eval=time.time() - test_start,
                                   total_steps=total_steps)
        return val_score
