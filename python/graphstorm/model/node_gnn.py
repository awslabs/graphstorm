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

    GNN model for node prediction task in GraphStorm.
"""
import abc
import torch as th

from .gnn import GSgnnModel, GSgnnModelBase

class GSgnnNodeModelInterface:
    """ The interface for GraphStorm node prediction model.

    This interface defines two main methods for training and inference.
    """
    @abc.abstractmethod
    def forward(self, blocks, node_feats, edge_feats,
        labels, input_nodes=None):
        """ The forward function for node prediction.

        This method is used for training. It takes a mini-batch, including
        the graph structure, node features, edge features and node labels and
        computes the loss of the model in the mini-batch.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
        node_feats : dict of Tensors
            The input node features of the message passing graphs.
        edge_feats : dict of Tensors
            The input edge features of the message passing graphs.
        labels: dict of Tensor
            The labels of the predicted nodes.
        input_nodes: dict of Tensors
            The input nodes of a mini-batch.

        Returns
        -------
        The loss of prediction.
        """

    @abc.abstractmethod
    def predict(self, blocks, node_feats, edge_feats, input_nodes, return_proba):
        """ Make prediction on the nodes with GNN.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message passing graph for computing GNN embeddings.
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
        Tensor : GNN prediction results. Return all the results when return_proba is true
            otherwise return the maximum result.
        Tensor : the GNN embeddings.
        """

class GSgnnNodeModelBase(GSgnnModelBase,  # pylint: disable=abstract-method
                         GSgnnNodeModelInterface):
    """ The base class for node-prediction GNN

    When a user wants to define a node prediction GNN model and train the model
    in GraphStorm, the model class needs to inherit from this base class.
    A user needs to implement some basic methods including `forward`, `predict`,
    `save_model`, `restore_model` and `create_optimizer`.
    """


class GSgnnNodeModel(GSgnnModel, GSgnnNodeModelInterface):
    """ GraphStorm GNN model for node prediction tasks

    Parameters
    ----------
    alpha_l2norm : float
        The alpha for L2 normalization.
    """
    def __init__(self, alpha_l2norm):
        super(GSgnnNodeModel, self).__init__()
        self.alpha_l2norm = alpha_l2norm

    def forward(self, blocks, node_feats, _, labels, input_nodes=None):
        """ The forward function for node prediction.

        This GNN model doesn't support edge features for now.
        """
        alpha_l2norm = self.alpha_l2norm
        if blocks is None or len(blocks) == 0:
            # no GNN message passing
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            encode_embs = self.compute_embed_step(blocks, node_feats)
        # TODO(zhengda) we only support node prediction on one node type now
        assert len(labels) == 1, "We only support prediction on one node type for now."
        target_ntype = list(labels.keys())[0]
        assert target_ntype in encode_embs
        emb = encode_embs[target_ntype]
        labels = labels[target_ntype]
        logits = self.decoder(emb)
        pred_loss = self.loss_func(logits, labels)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        return pred_loss + alpha_l2norm * reg_loss

    def predict(self, blocks, node_feats, _, input_nodes, return_proba):
        """ Make prediction on the nodes with GNN.
        """
        if blocks is None or len(blocks) == 0:
            # no GNN message passing in encoder
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            encode_embs = self.compute_embed_step(blocks, node_feats)
        # TODO(zhengda) we only support node prediction on one node type.
        assert len(encode_embs) == 1, \
            f'There are {len(encode_embs)} node types: {list(encode_embs.keys())}'
        target_ntype = list(encode_embs.keys())[0]
        if return_proba:
            return self.decoder.predict_proba(encode_embs[target_ntype]), \
                encode_embs[target_ntype]
        return self.decoder.predict(encode_embs[target_ntype]), encode_embs[target_ntype]

def node_mini_batch_gnn_predict(model, loader, return_proba=True, return_label=False):
    """ Perform mini-batch prediction on a GNN model.

    Parameters
    ----------
    model : GSgnnModel
        The GraphStorm GNN model
    loader : GSgnnNodeDataLoader
        The GraphStorm dataloader
    return_proba : bool
        Whether or not to return all the predictions or the maximum prediction
    return_label : bool
        Whether or not to return labels

    Returns
    -------
    Tensor : GNN prediction results. Return all the results when return_proba is true
        otherwise return the maximum result.
    Tensor : GNN embeddings.
    Tensor : labels if return_labels is True
    """
    device = model.device
    data = loader.data
    g = data.g
    preds = []
    embs = []
    labels = []
    model.eval()
    with th.no_grad():
        for input_nodes, seeds, blocks in loader:
            if not isinstance(input_nodes, dict):
                assert len(g.ntypes) == 1
                input_nodes = {g.ntypes[0]: input_nodes}
            input_feats = data.get_node_feats(input_nodes, device)
            blocks = [block.to(device) for block in blocks]
            pred, emb = model.predict(blocks, input_feats, None, input_nodes, return_proba)
            preds.append(pred.cpu())
            embs.append(emb.cpu())

            if return_label:
                lbl = data.get_labels(seeds)
                assert len(lbl) == 1
                labels.append(list(lbl.values())[0])
    model.train()
    preds = th.cat(preds)
    embs = th.cat(embs)
    if return_label:
        labels = th.cat(labels)
        return preds, embs, labels
    else:
        return preds, embs

def node_mini_batch_predict(model, emb, loader, return_proba=True, return_label=False):
    """ Perform mini-batch prediction.

    Parameters
    ----------
    model : GSgnnModel
        The GraphStorm GNN model
    emb : dict of Tensor
        The GNN embeddings
    loader : GSgnnNodeDataLoader
        The GraphStorm dataloader
    return_proba : bool
        Whether or not to return all the predictions or the maximum prediction
    return_label : bool
        Whether or not to return labels.

    Returns
    -------
    Tensor : GNN prediction results.
    Tensor : labels if return_labels is True
    """
    device = model.device
    data = loader.data
    preds = []
    labels = []
    # TODO(zhengda) I need to check if the data loader only returns target nodes.
    model.eval()
    with th.no_grad():
        for input_nodes, seeds, _ in loader:
            assert len(input_nodes) == 1, "Currently we only support one node type"
            ntype = list(input_nodes.keys())[0]
            in_nodes = input_nodes[ntype]
            if return_proba:
                pred = model.decoder.predict_proba(emb[ntype][in_nodes].to(device))
            else:
                pred = model.decoder.predict(emb[ntype][in_nodes].to(device))
            preds.append(pred.cpu())
            if return_label:
                lbl = data.get_labels(seeds)
                labels.append(lbl[ntype])
    model.train()

    preds = th.cat(preds)
    if return_label:
        labels = th.cat(labels)
        return preds, labels
    else:
        return preds
