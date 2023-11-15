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
import time
import logging
import abc
import torch as th

from .gnn import GSgnnModel, GSgnnModelBase
from .gnn_encoder_base import prepare_for_wholegraph
from .utils import append_to_dict
from ..utils import is_distributed, get_rank, is_wholegraph

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
        Tensor or dict of Tensor:
            GNN prediction results. Return all the results when return_proba is true
            otherwise return the maximum result.
        Tensor or dict of Tensor:
            The GNN embeddings.
        """

# pylint: disable=abstract-method
class GSgnnNodeModelBase(GSgnnNodeModelInterface,
                         GSgnnModelBase):
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
            encode_embs = self.compute_embed_step(blocks, node_feats, input_nodes)
        # Call emb normalization.
        # the default behavior is doing nothing.
        encode_embs = self.normalize_node_embs(encode_embs)

        target_ntypes = list(labels.keys())
        # compute loss for each node type and aggregate per node type loss
        pred_loss = 0
        for target_ntype in target_ntypes:
            assert target_ntype in encode_embs, f"Node type {target_ntype} not in encode_embs"
            assert target_ntype in labels, f"Node type {target_ntype} not in labels"
            emb = encode_embs[target_ntype]
            ntype_labels = labels[target_ntype]
            if isinstance(self.decoder, th.nn.ModuleDict):
                assert target_ntype in self.decoder, f"Node type {target_ntype} not in decoder"
                decoder = self.decoder[target_ntype]
            else:
                decoder = self.decoder
            ntype_logits = decoder(emb)
            if isinstance(self.loss_func, th.nn.ModuleDict):
                assert target_ntype in self.loss_func, \
                    f"Node type {target_ntype} not in loss function"
                loss_func = self.loss_func[target_ntype]
            else:
                loss_func = self.loss_func
            pred_loss += loss_func(ntype_logits, ntype_labels)
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
            encode_embs = self.compute_embed_step(blocks, node_feats, input_nodes)
        # Call emb normalization.
        # the default behavior is doing nothing.
        encode_embs = self.normalize_node_embs(encode_embs)

        target_ntypes = list(encode_embs.keys())
        # predict for each node type
        predicts = {}
        for target_ntype in target_ntypes:
            if isinstance(self.decoder, th.nn.ModuleDict):
                assert target_ntype in self.decoder, \
                    f"Node type {target_ntype} not in decoder"
                decoder = self.decoder[target_ntype]
            else:
                decoder = self.decoder
            if return_proba:
                predicts[target_ntype] = decoder.predict_proba(encode_embs[target_ntype])
            else:
                predicts[target_ntype] = decoder.predict(encode_embs[target_ntype])
        return predicts, encode_embs

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
    dict of Tensor :
        GNN prediction results. Return all the results when return_proba is true
        otherwise return the maximum result.
    dict of Tensor : GNN embeddings.
    dict of Tensor : labels if return_labels is True
    """
    if get_rank() == 0:
        logging.debug("Perform mini-batch inference for node prediction.")
    device = model.device
    data = loader.data
    g = data.g
    preds = {}

    if return_label:
        assert data.labels is not None, \
            "Return label is required, but the label field is not provided whem" \
            "initlaizing the inference dataset."

    embs = {}
    labels = {}
    model.eval()

    len_dataloader = max_num_batch = len(loader)
    tensor = th.tensor([len_dataloader], device=device)
    if is_distributed():
        th.distributed.all_reduce(tensor, op=th.distributed.ReduceOp.MAX)
        max_num_batch = tensor[0]

    dataloader_iter = iter(loader)

    with th.no_grad():
        # WholeGraph does not support imbalanced batch numbers across processes/trainers
        # TODO (IN): Fix dataloader to have the same number of minibatches
        for iter_l in range(max_num_batch):
            iter_start = time.time()
            tmp_keys = []
            blocks = None
            if iter_l < len_dataloader:
                input_nodes, seeds, blocks = next(dataloader_iter)
                if not isinstance(input_nodes, dict):
                    assert len(g.ntypes) == 1
                    input_nodes = {g.ntypes[0]: input_nodes}
            if is_wholegraph():
                tmp_keys = [ntype for ntype in g.ntypes if ntype not in input_nodes]
                prepare_for_wholegraph(g, input_nodes)
            input_feats = data.get_node_feats(input_nodes, device)
            if blocks is None:
                continue
            # Remove additional keys (ntypes) added for WholeGraph compatibility
            for ntype in tmp_keys:
                del input_nodes[ntype]
            blocks = [block.to(device) for block in blocks]
            pred, emb = model.predict(blocks, input_feats, None, input_nodes, return_proba)
            label = data.get_labels(seeds)
            if return_label:
                append_to_dict(label, labels)

            # pred can be a Tensor or a dict of Tensor
            # emb can be a Tensor or a dict of Tensor
            if isinstance(pred, dict):
                append_to_dict(pred, preds)
            else:
                assert len(seeds) == 1, \
                    f"Expect prediction results of multiple node types {label.keys()}" \
                    f"But only get results of one node type"
                ntype = list(seeds.keys())[0]
                append_to_dict({ntype: pred}, preds)

            if isinstance(emb, dict):
                append_to_dict(emb, embs)
            else: # in case model (e.g., llm encoder) only output a tensor without ntype
                ntype = list(seeds.keys())[0]
                append_to_dict({ntype: emb}, embs)
            if get_rank() == 0 and iter_l % 20 == 0:
                logging.debug("iter %d out of %d: takes %.3f seconds",
                              iter_l, max_num_batch, time.time() - iter_start)

    model.train()
    for ntype, ntype_pred in preds.items():
        preds[ntype] = th.cat(ntype_pred)
    for ntype, ntype_emb in embs.items():
        embs[ntype] = th.cat(ntype_emb)
    if return_label:
        for ntype, ntype_label in labels.items():
            labels[ntype] = th.cat(ntype_label)
        return preds, embs, labels
    else:
        return preds, embs, None

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
    dict of Tensor :
        Prediction results.
    dict of Tensor :
        Labels if return_labels is True
    """
    device = model.device
    data = loader.data

    if return_label:
        assert data.labels is not None, \
            "Return label is required, but the label field is not provided whem" \
            "initlaizing the inference dataset."

    preds = {}
    labels = {}
    # TODO(zhengda) I need to check if the data loader only returns target nodes.
    model.eval()
    with th.no_grad():
        for input_nodes, seeds, _ in loader:
            for ntype, in_nodes in input_nodes.items():
                if isinstance(model.decoder, th.nn.ModuleDict):
                    assert ntype in model.decoder, f"Node type {ntype} not in decoder"
                    decoder = model.decoder[ntype]
                else:
                    decoder = model.decoder
                if return_proba:
                    pred = decoder.predict_proba(emb[ntype][in_nodes].to(device))
                else:
                    pred = decoder.predict(emb[ntype][in_nodes].to(device))
                if ntype in preds:
                    preds[ntype].append(pred.cpu())
                else:
                    preds[ntype] = [pred.cpu()]
                if return_label:
                    lbl = data.get_labels(seeds)
                    if ntype in labels:
                        labels[ntype].append(lbl[ntype])
                    else:
                        labels[ntype] = [lbl[ntype]]
    model.train()

    for ntype, ntype_pred in preds.items():
        preds[ntype] = th.cat(ntype_pred)
    if return_label:
        for ntype, ntype_label in labels.items():
            labels[ntype] = th.cat(ntype_label)
        return preds, labels
    else:
        return preds, None
