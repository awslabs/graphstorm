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

    GNN model for edge prediction s in GraphStorm
"""
import abc
import logging
import time
import torch as th
import dgl

from .gnn import GSgnnModel, GSgnnModelBase
from .gnn_encoder_base import prepare_for_wholegraph
from .utils import append_to_dict

from ..utils import barrier, is_distributed, get_rank, is_wholegraph

class GSgnnEdgeModelInterface:
    """ The interface for GraphStorm edge prediction model.

    This interface defines two main methods: ``forward()`` for training and ``predict()``
    for inference. Edge GNN models should inherite this interface and implement the two
    methods.
    """
    @abc.abstractmethod
    def forward(self, blocks, target_edges, node_feats, edge_feats,
                target_edge_feats, labels, input_nodes=None):
        """ The forward function for edge prediction.

        This method is used for training. It takes a list of DGL message flow graphs (MFGs),
        node features, edge features, and edge labels of a mini-batch as inputs, and
        computes the loss of the model in the mini-batch as the return value. More
        detailed information about DGL MFG can be found in `DGL Neighbor Sampling
        Overview
        <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.

        Parameters
        ----------
        blocks: list of DGL MFGs
            Sampled subgraph in the list of DGL message flow graph (MFG) format. More
            detailed information about DGL MFG can be found in `DGL Neighbor Sampling
            Overview
            <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.
        target_edges : a DGLGraph
            The graph that stores target edges to run edge prediction.
        node_feats : dict of Tensors
            The input node features of the message passing graph.
        edge_feats : dict of Tensors
            The input edge features of the message passing graph.
        target_edge_feats: dict of Tensors
            The edge features of target edges
        labels: dict of Tensor
            The labels of the target edges.
        input_nodes: dict of Tensors
            The input nodes of the mini-batch.

        Returns
        -------
        float: The loss of prediction of this mini-batch.
        """

    @abc.abstractmethod
    def predict(self, blocks, target_edges, node_feats, edge_feats,
                target_edge_feats, input_nodes, return_proba):
        """ Make prediction on the target edges.

        Parameters
        ----------
        blocks: list of DGL MFGs
            Sampled subgraph in the list of DGL message flow graph (MFG) format. More
            detailed information about DGL MFG can be found in `DGL Neighbor Sampling
            Overview
            <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.
        target_edges : a DGLGraph
            The graph that stores target edges to run edge prediction.
        node_feats : dict of Tensors
            The node features of the message passing graph.
        edge_feats : dict of Tensors
            The edge features of the message passing graph.
        target_edge_feats: dict of Tensors
            The edge features of target edges.
        input_nodes: dict of Tensors
            The input nodes of the mini-batch.
        return_proba : bool
            Whether to return the predicted results, or only return the argmaxed ones in
            classification models.

        Returns
        -------
        Tensor, or dict of Tensor:
            GNN prediction results. Return results of all dimensions when ``return_proba``
            is True, otherwise return the argmaxed results.
        """

# pylint: disable=abstract-method
class GSgnnEdgeModelBase(GSgnnModelBase, GSgnnEdgeModelInterface):
    """ GraphStorm GNN model base class for edge prediction tasks.

    This base class extends GraphStorm ``GSgnnModelBase`` and ``GSgnnEdgeModelInterface``.
    When users want to define a customized edge prediction GNN model and train the model
    in GraphStorm, the model class needs to inherit from this base class, and implement
    the required methods including ``forward()``, ``predict()``, ``save_model()``,
    ``restore_model()`` and ``create_optimizer()``.
    """

class GSgnnEdgeModel(GSgnnModel, GSgnnEdgeModelInterface):
    """ GraphStorm GNN model for edge prediction tasks

    Parameters
    ----------
    alpha_l2norm : float
        The alpha for L2 normalization.
    """
    def __init__(self, alpha_l2norm):
        super(GSgnnEdgeModel, self).__init__()
        self.alpha_l2norm = alpha_l2norm

    # pylint: disable=unused-argument
    def forward(self, blocks, target_edges, node_feats, edge_feats, target_edge_feats,
        labels, input_nodes=None):
        """ The forward function for edge prediction.

        This GNN model doesn't support edge features right now.
        """
        alpha_l2norm = self.alpha_l2norm
        if blocks is None or len(blocks) == 0:
            # no GNN message passing
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            encode_embs = self.compute_embed_step(blocks, node_feats, input_nodes)
        # Call emb normalization.
        # the default behavior is doing nothing.
        encode_embs = self.normalize_node_embs(encode_embs)

        # TODO(zhengda) we only support prediction on one edge type now
        assert len(labels) == 1, "We only support prediction on one edge type for now."
        target_etype = list(labels.keys())[0]

        logits = self.decoder(target_edges, encode_embs, target_edge_feats)
        pred_loss = self.loss_func(logits, labels[target_etype])

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        return pred_loss + alpha_l2norm * reg_loss

    def predict(self, blocks, target_edges, node_feats, edge_feats,
                target_edge_feats, input_nodes, return_proba=False):
        """ Make prediction on edges.
        """
        if blocks is None or len(blocks) == 0:
            # no GNN message passing in encoder
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            encode_embs = self.compute_embed_step(blocks, node_feats, input_nodes)

        # Call emb normalization.
        # the default behavior is doing nothing.
        encode_embs = self.normalize_node_embs(encode_embs)
        if return_proba:
            return self.decoder.predict_proba(target_edges, encode_embs, target_edge_feats)
        return self.decoder.predict(target_edges, encode_embs, target_edge_feats)

def edge_mini_batch_gnn_predict(model, loader, return_proba=True, return_label=False):
    """ Perform mini-batch prediction on a GNN model.

    Parameters
    ----------
    model : GSgnnModel
        The GraphStorm GNN model
    loader : GSgnnEdgeDataLoader
        The GraphStorm dataloader
    return_proba: bool
        Whether to return all the predictions or the maximum prediction
    return_label : bool
        Whether or not to return labels

    Returns
    -------
    dict of Tensor : GNN prediction results. Return all the results when return_proba is true
        otherwise return the maximum result.
    dict of Tensor : labels if return_labels is True
    """
    if get_rank() == 0:
        logging.debug("Perform mini-batch inference for edge prediction.")
    device = model.device
    data = loader.data
    g = data.g
    preds = {}
    labels = {}
    model.eval()

    len_dataloader = max_num_batch = len(loader)
    num_batch = th.tensor([len_dataloader], device=device)
    if is_distributed():
        th.distributed.all_reduce(num_batch, op=th.distributed.ReduceOp.MAX)
        max_num_batch = num_batch[0]
    dataloader_iter = iter(loader)

    with th.no_grad():
        # WholeGraph does not support imbalanced batch numbers across processes/trainers
        # TODO (IN): Fix dataloader to have the same number of minibatches
        for iter_l in range(max_num_batch):
            iter_start = time.time()
            tmp_node_keys = tmp_edge_keys = []
            blocks = target_edge_graph = None
            input_nodes = {}
            input_edges = {}
            if iter_l < len_dataloader:
                input_nodes, target_edge_graph, blocks = next(dataloader_iter)
                if not isinstance(input_nodes, dict):
                    assert len(g.ntypes) == 1
                    input_nodes = {g.ntypes[0]: input_nodes}
                if loader.decoder_edge_feat_fields is not None:
                    input_edges = {etype: target_edge_graph.edges[etype].data[dgl.EID] \
                        for etype in target_edge_graph.canonical_etypes}
            if is_wholegraph():
                tmp_node_keys = [ntype for ntype in g.ntypes if ntype not in input_nodes]
                if loader.decoder_edge_feat_fields is not None:
                    tmp_edge_keys = [etype for etype in g.canonical_etypes \
                        if etype not in input_edges]
                prepare_for_wholegraph(g, input_nodes, input_edges)
            nfeat_fields = loader.node_feat_fields
            input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)
            if blocks is None:
                continue
            # Remove additional keys (ntypes) added for WholeGraph compatibility
            for ntype in tmp_node_keys:
                del input_nodes[ntype]
            if loader.decoder_edge_feat_fields is not None:
                edge_decoder_feats = \
                    data.get_edge_feats(input_edges,
                                        loader.decoder_edge_feat_fields,
                                        device)
                # Remove additional keys (etypes) added for WholeGraph compatibility
                for etype in tmp_edge_keys:
                    del input_edges[etype]
                edge_decoder_feats = {etype: feat.to(th.float32) \
                    for etype, feat in edge_decoder_feats.items()}
            else:
                edge_decoder_feats = None
            blocks = [block.to(device) for block in blocks]
            target_edge_graph = target_edge_graph.to(device)
            pred = model.predict(blocks, target_edge_graph, input_feats,
                                 None, edge_decoder_feats, input_nodes,
                                 return_proba)

            # TODO expand code for multiple edge types
            assert len(target_edge_graph.etypes) == 1, \
                "GraphStorm does not support multi-task training on " \
                "different edge types now."
            target_etype = target_edge_graph.canonical_etypes[0]

            if return_label:
                # retrieving seed edge id from the graph to find labels
                # TODO(zhengda) the data loader should return labels directly.
                seeds = target_edge_graph.edges[target_etype].data[dgl.EID]
                label_field = loader.label_field
                lbl = data.get_edge_feats({target_etype: seeds}, label_field)
                assert len(lbl) == 1
                append_to_dict(lbl, labels)
            if isinstance(pred, dict):
                append_to_dict(pred, preds)
            else: # model.predict return a tensor instead of a dict
                append_to_dict({target_etype: pred}, preds)
            if get_rank() == 0 and iter_l % 20 == 0:
                logging.debug("iter %d out of %d: takes %.3f seconds",
                              iter_l, max_num_batch, time.time() - iter_start)

    model.train()
    for target_etype, pred in preds.items():
        preds[target_etype] = th.cat(pred)
    if return_label:
        for target_etype, label in labels.items():
            labels[target_etype] = th.cat(label)
        return preds, labels
    else:
        return preds, None

def edge_mini_batch_predict(model, emb, loader, return_proba=True, return_label=False):
    """ Perform mini-batch prediction.

    This function usually follows full-grain GNN embedding inference. After having
    the GNN embeddings, we need to perform mini-batch computation to make predictions
    on the GNN embeddings.

    Parameters
    ----------
    model : GSgnnModel
        The GraphStorm GNN model
    emb : dict of Tensor
        The GNN embeddings
    loader : GSgnnEdgeDataLoader
        The GraphStorm dataloader
    return_proba: bool
        Whether to return all the predictions or the maximum prediction
    return_label : bool
        Whether or not to return labels

    Returns
    -------
    dict of Tensor : GNN prediction results. Return all the results when return_proba is true
        otherwise return the maximum result.
    dict of Tensor : labels if return_labels is True
    """
    # find the target src and dst ntypes
    model.eval()
    decoder = model.decoder
    device = model.device

    preds, labels = run_edge_mini_batch_predict(decoder,
                                                emb,
                                                loader,
                                                device,
                                                return_proba,
                                                return_label)
    model.train()
    return preds, labels

def run_edge_mini_batch_predict(decoder, emb, loader, device,
                                return_proba=True, return_label=False):
    """ Perform mini-batch edge prediction with the given decoder.

    This function usually follows full-graph GNN embedding inference. After having
    the GNN embeddings, we need to perform mini-batch computation to make predictions
    on the GNN embeddings.

    Note: callers should call model.eval() before calling this function
    and call model.train() after when doing training.

    Parameters
    ----------
    decoder : GSEdgeDecoder
        The GraphStorm edge decoder
    emb : dict of Tensor
        The GNN embeddings
    loader : GSgnnEdgeDataLoader
        The GraphStorm dataloader
    device: th.device
        Device used to compute prediction result
    return_proba: bool
        Whether to return all the predictions or the maximum prediction
    return_label : bool
        Whether or not to return labels

    Returns
    -------
    dict of Tensor : GNN prediction results. Return all the results when return_proba is true
        otherwise return the maximum result.
    dict of Tensor : labels if return_labels is True
    """
    data = loader.data
    g = data.g
    preds = {}
    labels = {}

    if return_label:
        assert loader.label_field is not None, \
            "Return label is required, but the label field is not provided when" \
            "initlaizing the inference dataloader."

    len_dataloader = max_num_batch = len(loader)
    num_batch = th.tensor([len_dataloader], device=device)
    if is_distributed():
        th.distributed.all_reduce(num_batch, op=th.distributed.ReduceOp.MAX)
        max_num_batch = num_batch[0]
    dataloader_iter = iter(loader)

    with th.no_grad():
        # save preds and labels together in order not to shuffle
        # the order when gather tensors from other trainers
        for iter_l in range(max_num_batch):
            tmp_edge_keys = []
            input_edges = {} if loader.decoder_edge_feat_fields is not None else None
            # TODO suppport multiple target edge types
            if iter_l < len_dataloader:
                input_nodes, target_edge_graph, _ = next(dataloader_iter)
                if loader.decoder_edge_feat_fields is not None:
                    input_edges = {etype: target_edge_graph.edges[etype].data[dgl.EID] \
                        for etype in target_edge_graph.canonical_etypes}
            if is_wholegraph() and loader.decoder_edge_feat_fields is not None:
                tmp_edge_keys = [etype for etype in g.canonical_etypes \
                    if etype not in input_edges]
                prepare_for_wholegraph(g, None, input_edges)
            assert len(target_edge_graph.etypes) == 1
            target_etype = target_edge_graph.canonical_etypes[0]
            batch_embs = {}
            for ntype, in_nodes in input_nodes.items():
                batch_embs[ntype] = emb[ntype][in_nodes].to(device)
            target_edge_graph = target_edge_graph.to(device)
            if loader.decoder_edge_feat_fields is not None:
                edge_decoder_feats = data.get_edge_feats(input_edges,
                                                         loader.decoder_edge_feat_fields,
                                                         target_edge_graph.device)
                # Remove additional keys (etypes) added for WholeGraph compatibility
                for etype in tmp_edge_keys:
                    del input_edges[etype]
                edge_decoder_feats = {etype: feat.to(th.float32) \
                    for etype, feat in edge_decoder_feats.items()}
            else:
                edge_decoder_feats = None

            if return_proba:
                pred = decoder.predict_proba(target_edge_graph, batch_embs, edge_decoder_feats)
            else:
                pred = decoder.predict(target_edge_graph, batch_embs, edge_decoder_feats)
            append_to_dict({target_etype: pred}, preds)

            # TODO(zhengda) we need to have the data loader reads everything,
            # instead of reading labels here.
            if return_label:
                label_field = loader.label_field
                lbl = data.get_edge_feats(
                    {target_etype: target_edge_graph.edges[target_etype].data[dgl.EID]},
                    label_field)

                append_to_dict(lbl, labels)
    barrier()

    for target_etype, pred in preds.items():
        preds[target_etype] = th.cat(pred)
    if return_label:
        for target_etype, label in labels.items():
            labels[target_etype] = th.cat(label)
        return preds, labels
    else:
        return preds, None
