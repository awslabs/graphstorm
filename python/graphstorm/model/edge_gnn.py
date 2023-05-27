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
import torch as th
import dgl

from .gnn import GSgnnModel, GSgnnModelBase

class GSgnnEdgeModelInterface:
    """ The interface for GraphStorm edge prediction model.

    This interface defines two main methods for training and inference.
    """
    @abc.abstractmethod
    def forward(self, blocks, batch_graph, node_feats, edge_feats,
        labels, input_nodes=None):
        """ The forward function for edge prediction.

        This method is used for training. It takes a mini-batch, including
        the graph structure, node features, edge features and edge labels and
        computes the loss of the model in the mini-batch.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        batch_graph : a DGLGraph
            The graph where we run edge classification.
        node_feats : dict of Tensors
            The input node features of the message passing graphs.
        edge_feats : dict of Tensors
            The input edge features of the message passing graphs.
        labels: dict of Tensor
            The labels of the predicted edges.
        input_nodes: dict of Tensors
            The input nodes of a mini-batch.

        Returns
        -------
        The loss of prediction.
        """

    @abc.abstractmethod
    def predict(self, blocks, batch_graph, node_feats, edge_feats, input_nodes, return_proba):
        """ Make prediction on the edges.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        batch_graph : a DGLGraph
            The graph where we run edge classification.
        node_feats : dict of Tensors
            The node features of the message passing graphs.
        edge_feats : dict of Tensors
            The edge features of the message passing graphs.
        input_nodes: dict of Tensors
            The input nodes of a mini-batch.
        return_proba : bool
            Whether or not to return all the predicted results or only the maximum one

        Returns
        -------
        Tensor : the prediction results. Return all the results when return_proba
            is true otherwise return the maximum value.
        """

class GSgnnEdgeModelBase(GSgnnModelBase,  # pylint: disable=abstract-method
                         GSgnnEdgeModelInterface):
    """ The base class for edge-prediction GNN

    When a user wants to define an edge prediction GNN model and train the model
    in GraphStorm, the model class needs to inherit from this base class.
    A user needs to implement some basic methods including `forward`, `predict`,
    `save_model`, `restore_model` and `create_optimizer`.
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

    def forward(self, blocks, batch_graph, node_feats, _,
        labels, input_nodes=None):
        """ The forward function for edge prediction.

        This GNN model doesn't support edge features right now.
        """
        alpha_l2norm = self.alpha_l2norm
        if blocks is None or len(blocks) == 0:
            # no GNN message passing
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            encode_embs = self.compute_embed_step(blocks, node_feats)
        # TODO(zhengda) we only support prediction on one edge type now
        assert len(labels) == 1, "We only support prediction on one edge type for now."
        target_etype = list(labels.keys())[0]

        logits = self.decoder(batch_graph, encode_embs)
        pred_loss = self.loss_func(logits, labels[target_etype])

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        return pred_loss + alpha_l2norm * reg_loss

    def predict(self, blocks, batch_graph, node_feats, _, input_nodes, return_proba=False):
        """ Make prediction on edges.
        """
        if blocks is None or len(blocks) == 0:
            # no GNN message passing in encoder
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            encode_embs = self.compute_embed_step(blocks, node_feats)
        if return_proba:
            return self.decoder.predict_proba(batch_graph, encode_embs)
        return self.decoder.predict(batch_graph, encode_embs)

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
    Tensor : GNN prediction results. Return all the results when return_proba is true
        otherwise return the maximum result.
    Tensor : labels if return_labels is True
    """
    device = model.device
    data = loader.data
    g = data.g
    preds = []
    labels = []
    model.eval()
    with th.no_grad():
        for input_nodes, batch_graph, blocks in loader:
            if not isinstance(input_nodes, dict):
                assert len(g.ntypes) == 1
                input_nodes = {g.ntypes[0]: input_nodes}
            input_feats = data.get_node_feats(input_nodes, device)
            blocks = [block.to(device) for block in blocks]
            pred = model.predict(blocks, batch_graph, input_feats, None, input_nodes,
                                 return_proba)
            preds.append(pred.cpu())

            if return_label:
                # retrieving seed edge id from the graph to find labels
                # TODO(zhengda) expand code for multiple edge types
                assert len(batch_graph.etypes) == 1
                target_etype = batch_graph.canonical_etypes[0]
                # TODO(zhengda) the data loader should return labels directly.
                seeds = batch_graph.edges[target_etype].data[dgl.EID]
                lbl = data.get_labels({target_etype: seeds})
                assert len(lbl) == 1
                labels.append(lbl[target_etype])
    model.train()
    preds = th.cat(preds)
    if return_label:
        return preds, th.cat(labels)
    else:
        return preds

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
    Tensor : GNN prediction results. Return all the results when return_proba is true
        otherwise return the maximum result.
    Tensor : labels if return_labels is True
    """
    # find the target src and dst ntypes
    model.eval()
    decoder = model.decoder
    data = loader.data
    with th.no_grad():
        # save preds and labels together in order not to shuffle
        # the order when gather tensors from other trainers
        preds_list = []
        labels_list = []
        device = model.device
        for input_nodes, batch_graph, _ in loader:
            assert len(batch_graph.etypes) == 1
            etype = batch_graph.canonical_etypes[0]
            batch_embs = {}
            for ntype, in_nodes in input_nodes.items():
                batch_embs[ntype] = emb[ntype][in_nodes].to(device)
            batch_graph = batch_graph.to(device)
            # TODO(zhengda) how to deal with edge features?
            if return_proba:
                preds_list.append(decoder.predict_proba(batch_graph, batch_embs))
            else:
                preds_list.append(decoder.predict(batch_graph, batch_embs))
            # TODO(zhengda) we need to have the data loader reads everything,
            # instead of reading labels here.
            if return_label:
                labels = data.get_labels({etype: batch_graph.edges[etype].data[dgl.EID]})
                labels_list.append(labels[etype])
        # can't use torch.stack here becasue the size of last tensor is different
        preds = th.cat(preds_list)
    th.distributed.barrier()

    model.train()
    if return_label:
        return preds, th.cat(labels_list)
    else:
        return preds
