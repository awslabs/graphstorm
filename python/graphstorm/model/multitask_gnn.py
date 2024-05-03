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
import time
import torch as th
import dgl

from ..config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                      BUILTIN_TASK_NODE_REGRESSION,
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      BUILTIN_TASK_EDGE_REGRESSION,
                      BUILTIN_TASK_LINK_PREDICTION)
from .gnn import GSgnnModel, GSgnnModelBase


class GSgnnMultiTaskModelInterface:
    """ The interface for GraphStorm multi-task learning.

    This interface defines two main methods for training and inference.
    """
    @abc.abstractmethod
    def forward(self, task_id, mini_batch):
        """ The forward function for multi-task learning

        This method is used for training, It runs model forword
        on a mini-batch for one task at a time.
        The loss of the model in the mini-batch is returned.

        Parameters
        ----------
        task_id: str
            ID of the task.
        mini_batch: tuple
            Mini-batch info


        Return
        ------
        The loss of prediction.
        """

    @abc.abstractmethod
    def predict(self, task_info, mini_batch):
        """ The forward function for multi-task prediction.

        This method is used for inference, It runs model forword
        on a mini-batch for one task at a time.
        The prediction result is returned.

        Parameters
        ----------
        task_info: TaskInfo
            task meta information
        mini_batch: tuple
            mini-batch info

        Returns
        -------
        Tensor or dict of Tensor:
            the prediction results.
        """

class GSgnnMultiTaskSharedEncoderModel(GSgnnModel, GSgnnMultiTaskModelInterface):
    """ GraphStorm GNN model for multi-task learning
    with a shared encoder model and separate decoder models.

    Parameters
    ----------
    alpha_l2norm : float
        The alpha for L2 normalization.
    """
    def __init__(self, alpha_l2norm):
        super(GSgnnMultiTaskSharedEncoderModel, self).__init__()
        self._alpha_l2norm = alpha_l2norm
        self._task_pool = {}

    def add_task(self, task_id, task_type,
                 decoder, loss_func, weight):
        """ Add a task into the multi-task pool
        """
        assert task_id not in self._task_pool, \
            f"Task {task_id} already exists"
        logging.info("Setup task %s", task_id)
        self._task_pool[task_id] = (task_type, decoder, loss_func, weight)

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

    # pylint: disable=unused-argument
    def forward(self, task_id, mini_batch):
        """ The forward function for multi-task learning
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
        encode_embs = self.normalize_node_embs(encode_embs)

        task_type, decoder, loss_func, weight = self.task_pool[task_id]

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
            ntype_logits = decoder(emb)
            pred_loss = loss_func(ntype_logits, ntype_labels)

            return pred_loss, weight
        elif task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
            target_edges, target_edge_feats, labels = decoder_data
            assert len(labels) == 1, \
                "In multi-task learning, only support do prediction " \
                "on one edge type for a single edge task."
            pred_loss = 0
            target_etype = list(labels.keys())[0]
            logits = decoder(target_edges, encode_embs, target_edge_feats)
            pred_loss = loss_func(logits, labels[target_etype])

            return pred_loss, weight
        elif task_type == BUILTIN_TASK_LINK_PREDICTION:
            pos_graph, neg_graph, pos_edge_feats, neg_edge_feats = decoder_data

            pos_score = decoder(pos_graph, encode_embs, pos_edge_feats)
            neg_score = decoder(neg_graph, encode_embs, neg_edge_feats)
            assert pos_score.keys() == neg_score.keys(), \
                "Positive scores and Negative scores must have edges of same" \
                f"edge types, but get {pos_score.keys()} and {neg_score.keys()}"
            pred_loss = loss_func(pos_score, neg_score)
            return pred_loss, weight
        else:
            raise TypeError("Unknow task type %s", task_type)


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
        encode_embs = self.normalize_node_embs(encode_embs)

        task_type, decoder, _, _ = self.task_pool[task_id]

        if task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
            assert len(encode_embs) == 1, \
                "In multi-task learning, only support do prediction " \
                "on one node type for a single node task."
            target_ntype = list(encode_embs.keys())[0]
            predicts = {}
            if return_proba:
                predicts[target_ntype] = decoder.predict_proba(encode_embs[target_ntype])
            else:
                predicts[target_ntype] = decoder.predict(encode_embs[target_ntype])
            return predicts
        elif task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
            target_edges, target_edge_feats, _ = decoder_data
            if return_proba:
                return decoder.predict_proba(target_edges, encode_embs, target_edge_feats)
            return decoder.predict(target_edges, encode_embs, target_edge_feats)
        elif task_type == BUILTIN_TASK_LINK_PREDICTION:
            logging.warning("Prediction for link prediction is not implemented")
            return None
        else:
            raise TypeError("Unknow task type %s", task_type)
