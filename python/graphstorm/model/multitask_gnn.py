"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GNN model for multi-task learning in GraphStorm
"""
import abc
import logging
import torch as th
from torch import nn
import dgl

from ..config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                      BUILTIN_TASK_NODE_REGRESSION,
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      BUILTIN_TASK_EDGE_REGRESSION,
                      BUILTIN_TASK_LINK_PREDICTION,
                      BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
from .gnn import GSgnnModel
from .gnn_encoder_base import GSgnnGNNEncoderInterface

from .node_gnn import run_node_mini_batch_predict
from .edge_gnn import run_edge_mini_batch_predict
from .lp_gnn import run_lp_mini_batch_predict
from .utils import LazyDistTensor
from .utils import normalize_node_embs, get_data_range
from ..utils import (
    get_rank,
    get_world_size,
    barrier,
    create_dist_tensor
)

class GSgnnMultiTaskModelInterface:
    """ The interface for GraphStorm multi-task learning.

    This interface defines two main methods for training and inference.
    """
    @abc.abstractmethod
    def forward(self, task_mini_batches):
        """ The forward function for multi-task learning

        This method is used for training, It runs model forword
        on a mini-batch for one task at a time.
        The loss of the model in the mini-batch is returned.

        Parameters
        ----------
        task_mini_batches: list
            Mini-batches.

        Return
        ------
        The loss of prediction.
        """

    @abc.abstractmethod
    def predict(self, task_id, mini_batch):
        """ The forward function for multi-task prediction.

        This method is used for inference, It runs model forword
        on a mini-batch for one task at a time.
        The prediction result is returned.

        Parameters
        ----------
        task_id: str
            Task ID.
        mini_batch: tuple
            Mini-batch info.

        Returns
        -------
        Tensor or dict of Tensor:
            the prediction results.
        """

class GSgnnMultiTaskSharedEncoderModel(GSgnnModel, GSgnnMultiTaskModelInterface):
    """ GraphStorm GNN model for multi-task learning
    with a shared encoder model and separate decoder models for each task.

    Parameters
    ----------
    alpha_l2norm : float
        The alpha for L2 normalization.
    """
    def __init__(self, alpha_l2norm):
        super(GSgnnMultiTaskSharedEncoderModel, self).__init__()
        self._alpha_l2norm = alpha_l2norm
        self._task_pool = {}
        self._decoder = nn.ModuleDict()
        self._loss_fn = nn.ModuleDict()
        self._node_embed_norm_methods = {}
        self._warn_printed = False

    def normalize_task_node_embs(self, task_id, embs, inplace=False):
        """ Normalize node embeddings when needed.

            normalize_task_node_embs should be called when embs stores embeddings
            of every node.

        Parameters
        ----------
        task_id: str
            Task ID.
        embs: dict of Tensors
            A dict of node embeddings.
        inplace: bool
            Whether to do inplace normalization.

        Returns
        -------
        new_embs: dict of Tensors
            Normalized node embeddings.
        """
        if self._node_embed_norm_methods[task_id] is not None:
            new_embs = {}
            rank = get_rank()
            world_size = get_world_size()
            for key, emb in embs.items():
                if isinstance(emb, (dgl.distributed.DistTensor, LazyDistTensor)):
                    # If emb is a distributed tensor, multiple processes are doing
                    # embdding normalization concurrently. We need to split
                    # the task. (From full_graph_inference)
                    start, end = get_data_range(rank, world_size, len(emb))
                    new_emb = emb if inplace else \
                        create_dist_tensor(emb.shape,
                                           emb.dtype,
                                           name=f"{emb.name}_task_id",
                                           part_policy=emb.part_policy,
                                           persistent=True)
                else:
                    # If emb is just a torch Tensor. do normalization directly.
                    # (From mini_batch_inference)
                    start, end = 0, len(emb)
                    new_emb = emb if inplace else th.clone(emb)
                idx = start
                while idx + 1024 < end:
                    new_emb[idx:idx+1024] = \
                        self.minibatch_normalize_task_node_embs(
                            task_id,
                            {key:emb[idx:idx+1024]})[key]
                    idx += 1024
                new_emb[idx:end] = \
                    self.minibatch_normalize_task_node_embs(
                        task_id,
                        {key:emb[idx:end]})[key]
                barrier()
                new_embs[key] = new_emb
            return new_embs
        else:
            # If normalization method is None
            # do nothing.
            new_embs = embs
            return new_embs

    # pylint: disable = arguments-differ
    def minibatch_normalize_task_node_embs(self, task_id, embs):
        """ Normalize node embeddings when needed for a mini-batch.

            minibatch_normalize_task_node_embs should be called in
            forward() and predict().

        Parameters
        ----------
        task_id: str
            Task ID.
        embs: dict of Tensors
            A dict of node embeddings.

        Returns
        -------
        embs: dict of Tensors
            Normalized node embeddings.
        """
        if self._node_embed_norm_methods[task_id] is not None:
            return normalize_node_embs(embs, self._node_embed_norm_methods[task_id])
        else:
            return embs

    @property
    def node_embed_norm_methods(self):
        """ Get per task node embedding normalization method

        Returns
        -------
        dict of strings:
            Normalization methods
        """
        return self._node_embed_norm_methods

    def add_task(self, task_id, task_type,
                 decoder, loss_func,
                 embed_norm_method=None):
        """ Add a task into the multi-task pool

        Parameters
        ----------
        task_id: str
            Task ID.
        task_type: str
            Task type.
        decoder: GSNodeDecoder or
                 GSEdgeDecoder or
                 LinkPredictNoParamDecoder or
                 LinkPredictLearnableDecoder
            Task decoder.
        loss_func: func
            Loss function.
        embed_norm_method: str
            Node embedding normalization method.
        """
        assert task_id not in self._task_pool, \
            f"Task {task_id} already exists"
        logging.info("Setup task %s", task_id)
        self._task_pool[task_id] = (task_type, loss_func)
        self._decoder[task_id] = decoder
        # add loss func in nn module
        self._loss_fn[task_id] = loss_func
        self._node_embed_norm_methods[task_id] = embed_norm_method

    @property
    def alpha_l2norm(self):
        """Get parameter norm params
        """
        return self._alpha_l2norm

    @property
    def task_pool(self):
        """ Get task pool
        """
        return self._task_pool

    @property
    def task_decoders(self):
        """ Get task decoders
        """
        return self._decoder

    def _run_mini_batch(self, task_info, mini_batch):
        """ Run mini_batch forward.
        """
        if task_info.task_type in \
            [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
            # Order follow GSgnnNodeModelInterface.forward
            blocks, input_feats, edge_feats, lbl,input_nodes = mini_batch
            loss = self._forward(task_info.task_id,
                                 (blocks, input_feats, edge_feats, input_nodes),
                                 lbl)

        elif task_info.task_type in \
            [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
            # Order follow GSgnnEdgeModelInterface.forward
            blocks, target_edges, node_feats, edge_feats, \
                edge_decoder_feats, lbl, input_nodes = mini_batch
            loss = self._forward(task_info.task_id,
                                 (blocks, node_feats, None, input_nodes),
                                 (target_edges, edge_decoder_feats, lbl))

        elif task_info.task_type == BUILTIN_TASK_LINK_PREDICTION:
            # Order follow GSgnnLinkPredictionModelInterface.forward
            blocks, pos_graph, neg_graph, node_feats, edge_feats, \
            pos_edge_feats, neg_edge_feats, input_nodes = mini_batch

            loss = self._forward(task_info.task_id,
                                 (blocks, node_feats, edge_feats, input_nodes),
                                 (pos_graph, neg_graph, pos_edge_feats, neg_edge_feats))
        elif task_info.task_type == BUILTIN_TASK_RECONSTRUCT_NODE_FEAT:
            # Order follow GSgnnNodeModelInterface.forward
            blocks, input_feats, edge_feats, lbl, input_nodes = mini_batch
            loss = self._forward(task_info.task_id,
                                 (blocks, input_feats, edge_feats, input_nodes),
                                 lbl)
        else:
            raise TypeError(f"Unknown task {task_info}")

        return loss, task_info.task_config.task_weight

    def forward(self, task_mini_batches):
        """ The forward function for multi-task learning
            It will iterate over the mini-batches and call
            forward for each task.

            Return
            mt_loss: overall loss
            losses: per task loss (used for debug)
        """
        losses = []
        for (task_info, mini_batch) in task_mini_batches:
            loss, weight = self._run_mini_batch(task_info, mini_batch)
            losses.append((loss, weight))

        reg_loss = th.tensor(0.).to(losses[0][0].device)
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()
        alpha_l2norm = self.alpha_l2norm

        mt_loss = reg_loss * alpha_l2norm
        for loss, weight in losses:
            mt_loss += loss * weight

        return mt_loss, losses

    # pylint: disable=unused-argument
    def _forward(self, task_id, encoder_data, decoder_data):
        """ The forward function to run forward for a specific
            task with task_id.

        Parameters
        ----------
        task_id: str
            Task ID.
        encoder_data: tuple
            The input data for the encoder.
        decoder_data: tuple
            The input for the decoder.
        """
        assert task_id in self.task_pool, \
            f"Unknown task: {task_id} in multi-task learning." \
            f"Existing tasks are {self.task_pool.keys()}"

        # message passing graph, node features, edge features, seed nodes
        blocks, node_feats, _, input_nodes = encoder_data
        task_type, loss_func = self.task_pool[task_id]
        task_decoder = self.decoder[task_id]

        if blocks is None or len(blocks) == 0:
            # no GNN message passing
            if task_type == BUILTIN_TASK_RECONSTRUCT_NODE_FEAT:
                logging.warning("Reconstruct node feature with only " \
                                "input embedding layer may not work.")
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            # GNN message passing
            if task_type == BUILTIN_TASK_RECONSTRUCT_NODE_FEAT:
                if isinstance(self.gnn_encoder, GSgnnGNNEncoderInterface):
                    if self.has_sparse_params():
                        # When there are learnable embeddings, we can not
                        # just simply skip the last layer self-loop.
                        # It may break the sparse optimizer backward code logic
                        # keep the self-loop and print a warning insetead
                        encode_embs = self.compute_embed_step(
                            blocks, node_feats, input_nodes)
                        if self._warn_printed is False:
                            logging.warning("When doing %s training, we need to "
                                            "avoid adding self loop in the last GNN layer "
                                            "to avoid the potential node "
                                            "feature leakage issue. "
                                            "When there are learnable embeddings on "
                                            "nodes, GraphStorm can not automatically"
                                            "skip the last layer self-loop"
                                            "Please set use_self_loop to False",
                                            BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
                            self._warn_printed = True
                    else:
                        # skip the selfloop of the last layer to
                        # avoid information leakage.
                        self.gnn_encoder.skip_last_selfloop()
                        encode_embs = self.compute_embed_step(
                            blocks, node_feats, input_nodes)
                        self.gnn_encoder.reset_last_selfloop()
                else:
                    if self._warn_printed is False:
                        # Only print warning once to avoid overwhelming the log.
                        logging.warning("The gnn encoder %s does not support skip "
                                        "the last self-loop operation"
                                        "(skip_last_selfloop). There is a potential "
                                        "node feature leakage risk when doing %s training.",
                                        type(self.gnn_encoder),
                                        BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
                        self._warn_printed = True
                    encode_embs = self.compute_embed_step(
                        blocks, node_feats, input_nodes)
            else:
                encode_embs = self.compute_embed_step(blocks, node_feats, input_nodes)

        # Call emb normalization.
        encode_embs = self.minibatch_normalize_task_node_embs(task_id, encode_embs)

        if task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
            labels = decoder_data
            assert len(labels) == 1, \
                "In multi-task learning, only support do prediction " \
                "on one node type for a single node task."
            pred_loss = 0
            target_ntype = list(labels.keys())[0]

            assert target_ntype in encode_embs, f"Node type {target_ntype} not in encode_embs"
            assert target_ntype in labels, f"Node type {target_ntype} not in labels"
            emb = encode_embs[target_ntype]
            ntype_labels = labels[target_ntype]
            ntype_logits = task_decoder(emb)
            pred_loss = loss_func(ntype_logits, ntype_labels)

            return pred_loss
        elif task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
            batch_graph, target_edge_feats, labels = decoder_data
            assert len(labels) == 1, \
                "In multi-task learning, only support do prediction " \
                "on one edge type for a single edge task."
            pred_loss = 0
            target_etype = list(labels.keys())[0]
            logits = task_decoder(batch_graph, encode_embs, target_edge_feats)
            pred_loss = loss_func(logits, labels[target_etype])

            return pred_loss
        elif task_type == BUILTIN_TASK_LINK_PREDICTION:
            pos_graph, neg_graph, pos_edge_feats, neg_edge_feats = decoder_data

            pos_score = task_decoder(pos_graph, encode_embs, pos_edge_feats)
            neg_score = task_decoder(neg_graph, encode_embs, neg_edge_feats)
            assert pos_score.keys() == neg_score.keys(), \
                "Positive scores and Negative scores must have edges of same" \
                f"edge types, but get {pos_score.keys()} and {neg_score.keys()}"
            pred_loss = loss_func(pos_score, neg_score)
            return pred_loss
        elif task_type == BUILTIN_TASK_RECONSTRUCT_NODE_FEAT:
            labels = decoder_data
            assert len(labels) == 1, \
                "In multi-task learning, only support do prediction " \
                "on one node type for a single node task."
            pred_loss = 0
            target_ntype = list(labels.keys())[0]

            assert target_ntype in encode_embs, f"Node type {target_ntype} not in encode_embs"
            assert target_ntype in labels, f"Node type {target_ntype} not in labels"
            emb = encode_embs[target_ntype]
            ntype_labels = labels[target_ntype]
            ntype_logits = task_decoder(emb)
            pred_loss = loss_func(ntype_logits, ntype_labels)

            return pred_loss
        else:
            raise TypeError(f"Unknow task type {task_type}")

    def predict(self, task_id, mini_batch, return_proba=False):
        """ The forward function for multi-task inference
        """
        assert task_id in self.task_pool, \
            f"Unknown task: {task_id} in multi-task learning." \
            f"Existing tasks are {self.task_pool.keys()}"

        encoder_data, decoder_data = mini_batch
        # message passing graph, node features, edge features, seed nodes
        blocks, node_feats, _, input_nodes = encoder_data
        if blocks is None or len(blocks) == 0:
            # no GNN message passing
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            # GNN message passing
            encode_embs = self.compute_embed_step(blocks, node_feats, input_nodes)

        # Call emb normalization.
        encode_embs = self.minibatch_normalize_task_node_embs(task_id, encode_embs)

        task_type, _ = self.task_pool[task_id]
        task_decoder = self.decoder[task_id]

        if task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
            assert len(encode_embs) == 1, \
                "In multi-task learning, only support do prediction " \
                "on one node type for a single node task."
            target_ntype = list(encode_embs.keys())[0]
            predicts = {}
            if return_proba:
                predicts[target_ntype] = task_decoder.predict_proba(encode_embs[target_ntype])
            else:
                predicts[target_ntype] = task_decoder.predict(encode_embs[target_ntype])
            return predicts
        elif task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
            batch_graph, target_edge_feats, _ = decoder_data
            if return_proba:
                return task_decoder.predict_proba(batch_graph, encode_embs, target_edge_feats)
            return task_decoder.predict(batch_graph, encode_embs, target_edge_feats)
        elif task_type == BUILTIN_TASK_LINK_PREDICTION:
            logging.warning("Prediction for link prediction is not implemented")
            return None
        elif task_type == BUILTIN_TASK_RECONSTRUCT_NODE_FEAT:
            logging.warning("Prediction for node feature reconstruction is not supported")
            return None
        else:
            raise TypeError(f"Unknow task type {task_type}")

def multi_task_mini_batch_predict(
    model, emb, dataloaders, task_infos, device, return_proba=True, return_label=False):
    """ conduct mini batch prediction on multiple tasks.

        The task infos are passed in as task_infos.
        The task dataloaders are passed in as dataloaders.

    Parameters
    ----------
    model: GSgnnMultiTaskModelInterface, GSgnnModel
        Multi-task learning model
    emb : dict of Tensor
        The GNN embeddings
    dataloaders: list
        List of val or test dataloaders.
    task_infos: list
        List of task info
    device: th.device
        Device used to compute test scores.
    return_proba: bool
        Whether to return all the predictions or the maximum prediction.
    return_label : bool
        Whether or not to return labels.

    Returns
    -------
    dict: prediction results of each task
    """
    task_decoders = model.task_decoders
    res = {}
    with th.no_grad():
        for dataloader, task_info in zip(dataloaders, task_infos):
            # normalize the node embedding if needed.
            # input emb is shared across different tasks
            # so that we can not do inplace normalization.
            #
            # Note(xiangsx): Currently node embedding normalization
            # only supports link prediction tasks.
            # model.normalize_task_node_embs does nothing
            # for node and edge prediction tasks.
            # TODO(xiangsx): Need a more memory efficient design when
            # node embedding normalization supports node and edge
            # prediction tasks.
            emb = model.normalize_task_node_embs(task_info.task_id, emb, inplace=False)
            if task_info.task_type in \
            [BUILTIN_TASK_NODE_CLASSIFICATION,
             BUILTIN_TASK_NODE_REGRESSION,
             BUILTIN_TASK_RECONSTRUCT_NODE_FEAT]:
                if dataloader is None:
                    # In cases when there is no validation or test set.
                    # set pred and labels to None
                    res[task_info.task_id] = (None, None)
                else:
                    decoder = task_decoders[task_info.task_id]
                    preds, labels = \
                        run_node_mini_batch_predict(decoder,
                                                    emb,
                                                    dataloader,
                                                    device,
                                                    return_proba,
                                                    return_label)
                    assert not labels or len(labels) == 1, \
                        "In multi-task learning, for each training task, " \
                        "we only support prediction on one node type." \
                        "For multiple node types, please treat them as " \
                        "different training tasks."
                    ntype = list(preds.keys())[0]
                    res[task_info.task_id] = (preds[ntype], labels[ntype] \
                        if labels else None)
            elif task_info.task_type in \
            [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
                if dataloader is None:
                    # In cases when there is no validation or test set.
                    # set pred and labels to None
                    res[task_info.task_id] = (None, None)
                else:
                    decoder = task_decoders[task_info.task_id]
                    preds, labels = \
                        run_edge_mini_batch_predict(decoder,
                                                    emb,
                                                    dataloader,
                                                    device,
                                                    return_proba,
                                                    return_label)
                    assert not labels or len(labels) == 1, \
                        "In multi-task learning, for each training task, " \
                        "we only support prediction on one edge type." \
                        "For multiple edge types, please treat them as " \
                        "different training tasks."
                    etype = list(preds.keys())[0]
                    res[task_info.task_id] = (preds[etype], labels[etype] \
                        if labels else None)
            elif task_info.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
                if dataloader is None:
                    # In cases when there is no validation or test set.
                    res[task_info.task_id] = None
                else:
                    decoder = task_decoders[task_info.task_id]
                    ranking = run_lp_mini_batch_predict(decoder, emb, dataloader, device)
                    res[task_info.task_id] = ranking
            else:
                raise TypeError(f"Unsupported task {task_info}")

    return res

def gen_emb_for_nfeat_reconstruct(model, gen_embs):
    """ Generate node embeddings for node feature reconstruction.
        In theory, we should skip the self-loop of the last GNN layer.
        However, there are some exceptions. This function handles
        those exceptions.

    Parameters
    ----------
    model: GSgnnMultiTaskSharedEncoderModel
        Multi-task model
    gen_embs: func
        The function used to generate node embeddings.
        It should accept a bool flag indicating whether
        the last GNN layer self-loop should be removed.

    Return
    ------
    embs: node embeddings
    """
    if isinstance(model.gnn_encoder, GSgnnGNNEncoderInterface):
        if model.has_sparse_params():
            # When there are learnable embeddings, we can not
            # just simply skip the last layer self-loop.
            # Keep the self-loop and print a warning
            # we will use the computed embs directly
            logging.warning("When doing %s inference, we need to "
                            "avoid adding self loop in the last GNN layer "
                            "to avoid the potential node "
                            "feature leakage issue. "
                            "When there are learnable embeddings on "
                            "nodes, GraphStorm can not automatically"
                            "skip the last layer self-loop"
                            "Please set use_self_loop to False",
                            BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
            embs = gen_embs(skip_last_self_loop=False)
        else:
            # skip the selfloop of the last layer to
            # avoid information leakage.
            embs = gen_embs(skip_last_self_loop=True)
    else:
        # we will use the computed embs directly
        logging.warning("The gnn encoder %s does not support skip "
                        "the last self-loop operation"
                        "(skip_last_selfloop). There is a potential "
                        "node feature leakage risk when doing %s training.",
                        type(model.gnn_encoder),
                        BUILTIN_TASK_RECONSTRUCT_NODE_FEAT)
        embs = gen_embs(skip_last_self_loop=False)
    return embs
