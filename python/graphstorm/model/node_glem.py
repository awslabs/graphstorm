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

    GLEM model for node prediction task in GraphStorm.
"""
import os
from copy import deepcopy
import torch as th
import dgl

from .gnn import GSOptimizer
from .node_gnn import GSgnnNodeModel, GSgnnNodeModelBase

class GLEM(GSgnnNodeModelBase):
    """
    GLEM model (https://arxiv.org/abs/2210.14709) for node-level tasks.
    Parameters
    ----------
    alpha_l2norm: float
        Coef for l2 norm of unused weights
    em_order_gnn_first: bool
        In the EM training, set true to train GNN first, train LM first otherwise
    inference_using_gnn: bool
        Set true to use GNN in inference time, otherwise inference with LM
    pl_weight: float
        Weight for the pseudo-likelihood loss in GLEM's loss function.
    """
    def __init__(self,
                 alpha_l2norm,
                 em_order_gnn_first=False,
                 inference_using_gnn=True,
                 pl_weight=0.5
                 ):
        super(GLEM, self).__init__()
        self.alpha_l2norm = alpha_l2norm
        self.em_order_gnn_first = em_order_gnn_first
        self.inference_using_gnn = inference_using_gnn
        self.pl_weight = pl_weight
        self.lm = GSgnnNodeModel(alpha_l2norm)
        self.gnn = GSgnnNodeModel(alpha_l2norm)
        self.training_lm = not em_order_gnn_first

    def init_optimizer(self, lr, sparse_optimizer_lr, weight_decay, lm_lr=None):
        """Initialize optimzer, which will be stored in self.lm._optimizer, self.gnn._optimizer
        """
        sparse_params = self.gnn.get_sparse_params()
        if len(sparse_params) > 0:
            emb_optimizer = dgl.distributed.optim.SparseAdam(sparse_params, lr=sparse_optimizer_lr)
            sparse_opts = [emb_optimizer]
        else:
            sparse_opts = []

        dense_params = self.gnn.get_dense_params() + self.lm.get_dense_params()
        if len(dense_params) > 0:
            optimizer = th.optim.Adam(dense_params, lr=lr,
                                      weight_decay=weight_decay)
            dense_opts = [optimizer]
        else:
            dense_opts = []

        lm_params = self.lm.get_lm_params()
        if len(lm_params) > 0:
            lm_optimizer = th.optim.Adam(lm_params, \
                                         lr=lm_lr if lm_lr is not None else lr,
                                         weight_decay=weight_decay)
            lm_opts = [lm_optimizer]
        else:
            lm_opts = []
        self._optimizer = GSOptimizer(dense_opts=dense_opts,
                                      lm_opts=lm_opts,
                                      sparse_opts=sparse_opts)

    def create_optimizer(self):
        """Create the optimizer that optimizes the model."""
        return self._optimizer

    def save_model(self, model_path):
        """Save either the LM and GNN models.
        `training_lm` flag determine which actively training model to save
        """
        self.lm.save_model(os.path.join(model_path, 'LM'))
        self.gnn.save_model(os.path.join(model_path, 'GNN'))

    def restore_model(self, restore_model_path, model_layer_to_load=None):
        """Restore models from checkpoints."""
        self.lm.restore_model(os.path.join(restore_model_path, 'LM'), model_layer_to_load)
        self.gnn.restore_model(os.path.join(restore_model_path, 'GNN'), model_layer_to_load)

    def set_node_input_encoder(self, encoder):
        """Set the node input LM encoder for lm, shared with gnn. 
        """
        self.lm.set_node_input_encoder(encoder)

    def set_gnn_encoder(self, gnn_encoder):
        """Set gnn encoder only for gnn.
        """
        self.gnn.set_gnn_encoder(gnn_encoder)

    @property
    def gnn_encoder(self):
        """Alias for accessing the gnn_encoder"""
        return self.gnn.gnn_encoder

    def set_decoder(self, decoder):
        """Set the same decoder for both, since lm needs to be able to 
        predict node labels during training
        """
        self.lm.set_decoder(deepcopy(decoder))
        self.gnn.set_decoder(decoder)

    def set_loss_func(self, loss_fn):
        """Set the final loss function based on the node task 
        """
        self.lm.set_loss_func(loss_fn)
        self.gnn.set_loss_func(loss_fn)

    def freeze_params(self, part='lm'):
        """Freeze parameters in lm or gnn"""
        if part == 'lm':
            params = self.lm.parameters()
        elif part == 'gnn':
            params = self.gnn.parameters()
        for param in params:
            param.requires_grad = False

    def unfreeze_params(self, part='lm'):
        """Unfreeze parameters in lm or gnn"""
        if part == 'lm':
            params = self.lm.parameters()
        elif part == 'lm-input-proj':
            params = self.lm.node_input_encoder.input_projs.parameters()
        elif part == 'gnn':
            params = self.gnn.parameters()
        for param in params:
            param.requires_grad = True

    def toggle(self, part='lm'):
        """The method toggles training between lm and gnn."""
        if part == 'lm':
            self.training_lm = True
            self.freeze_params('gnn')
            self.unfreeze_params('lm')
        elif part == 'gnn':
            self.training_lm = False
            self.freeze_params('lm')
            self.unfreeze_params('gnn')
            self.unfreeze_params('lm-input-proj')
        else:
            raise ValueError(f"Unknown model part: {part}")

    def forward(self, blocks, node_feats, edge_feats, labels, input_nodes, use_gnn=True,
                no_pl=False):
        """ Forward pass for GLEM model.
        Parameters
        ----------        
        blocks : List[dgl.heterograph.DGLBlock]
        node_feats : Dict[ntype: tensors.shape [bs, feat_dim]]
        edge_feats : None
        labels : Dict[target_ntype: tensor.shape [bs]]
        input_nodes : {target_ntype: tensor.shape [bs], other_ntype: []}
        use_gnn : bool
            If True, use GNN's decoder, otherwise, use LM's decoder
        no_pl : bool
            If True, do not calculate pseudo likelihood, use MLE loss only
        """
        if use_gnn:
            total_loss = self.forward_gnn(blocks, node_feats, edge_feats, labels, input_nodes,
                                          no_pl=no_pl)
        else:
            total_loss = self.forward_lm(blocks, node_feats, edge_feats, labels, input_nodes,
                                         no_pl=no_pl)
        return total_loss

    def forward_gnn(self, blocks, node_feats, _, labels, input_nodes, no_pl=False):
        """Forward pass for node prediction using GNN
        """
        emb_lm, emb_gnn = self._embed_nodes(blocks, node_feats, _, input_nodes, do_gnn_encode=True)
        labels = self._process_labels(labels)
        logits = self.gnn.decoder(emb_gnn)
        if no_pl:
            loss = self.gnn.loss_func(logits, labels)
        else:
            batch_size = labels.size(0)
            n_gold_nodes = batch_size // 2
            # compute pseudo labels from LM
            logits_lm = self.lm.decoder(emb_lm[n_gold_nodes:])
            pseudo_labels = logits_lm.argmax(-1)

            # GLEM loss
            loss = compute_loss(self.gnn.loss_func, logits, labels[:n_gold_nodes], pseudo_labels,
                                pl_weight=self.pl_weight)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(loss.device)
        # L2 regularization of dense parameters
        for d_para in self.gnn.get_dense_params():
            reg_loss += d_para.square().sum()
        # weighted addition to the total loss
        return loss + self.alpha_l2norm * reg_loss


    def forward_lm(self, blocks, node_feats, _, labels, input_nodes, no_pl=False):
        """Forward pass for node prediction using LM"""
        emb_lm, emb_gnn = self._embed_nodes(blocks, node_feats, _, input_nodes,
                                            do_gnn_encode=not no_pl)
        labels = self._process_labels(labels)
        logits = self.lm.decoder(emb_lm)
        if no_pl:
            loss = self.lm.loss_func(logits, labels)
        else:
            batch_size = labels.size(0)
            n_gold_nodes = batch_size // 2
            # compute pseudo labels from GNN
            logits_gnn = self.gnn.decoder(emb_gnn[n_gold_nodes:])
            pseudo_labels = logits_gnn.argmax(-1)

            # GLEM loss
            loss = compute_loss(self.gnn.loss_func, logits, labels[:n_gold_nodes], pseudo_labels,
                                pl_weight=self.pl_weight)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(loss.device)
        # L2 regularization of dense parameters
        for d_para in self.gnn.get_dense_params():
            reg_loss += d_para.square().sum()
        # weighted addition to the total loss
        return loss + self.alpha_l2norm * reg_loss


    def predict(self, blocks, node_feats, edge_feats, input_nodes, return_proba):
        """Make prediction on the nodes with the LM or GNN. 
        The model's `inference_using_gnn` flag determines how inference is performed.
        If inference_using_gnn is True, message-passing GNN is used on the LM features,
        Otherwise, LM's decoder is used for inference, no message-passing involved.

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
        emb_lm, emb_gnn = self._embed_nodes(blocks, node_feats, edge_feats, input_nodes,
                                do_gnn_encode=self.inference_using_gnn)
        if self.inference_using_gnn:
            decoder = self.gnn.decoder
            emb = emb_gnn
        else:
            decoder = self.lm.decoder
            emb = emb_lm
        if return_proba:
            preds = decoder.predict_proba(emb)
        else:
            preds = decoder.predict(emb)
        return preds, emb

    def _get_seed_nodes(self, input_nodes, blocks):
        """ Get seed nodes from input nodes and labels of the seed nodes.
        Parameters
        ----------        
        input_nodes : {target_ntype: tensor.shape [bs], other_ntype: []}
        blocks : list[dgl.Block]
        """
        target_ntype = blocks[-1].dsttypes[0]
        n_seed_nodes = blocks[-1].num_dst_nodes()
        return {target_ntype: input_nodes[target_ntype][:n_seed_nodes]}

    def _embed_nodes(self, blocks, node_feats, _, input_nodes=None, do_gnn_encode=True):
        """ Embed and encode nodes with LM, optionally followed by GNN encoder for GLEM model
        """
        if do_gnn_encode:
            # Get the projected LM embeddings without GNN message passing
            encode_embs = self.lm.comput_input_embed(input_nodes, node_feats)
            target_ntype = list(encode_embs.keys())[0]
            # GNN message passing
            encode_embs_gnn = self.gnn.gnn_encoder(blocks, encode_embs)
            n_seed_nodes = blocks[-1].num_dst_nodes()
            return encode_embs[target_ntype][:n_seed_nodes], encode_embs_gnn[target_ntype]
        else:
            # Get the projected LM embeddings for seed nodes:
            seed_nodes = self._get_seed_nodes(input_nodes, blocks)
            encode_embs = self.lm.comput_input_embed(seed_nodes, node_feats)
            target_ntype = list(encode_embs.keys())[0]
            return encode_embs[target_ntype], None

    def _process_labels(self, labels):
        # TODO(zhengda) we only support node prediction on one node type now
        assert len(labels) == 1, "We only support prediction on one node type for now."
        target_ntype = list(labels.keys())[0]
        return labels[target_ntype]


def compute_loss(loss_func, logits, labels, pseudo_labels=None, pl_weight=0.5):
    """Combine two types of losses: (1-alpha)*MLE (CE loss on gold labels) + alpha*Pl_loss 
    (CE loss on pseudo labels) semi-supervised objective for training LM or GNN individually 
    in E-step, M-step, respectively.
    If pseudo_labels is provided, assuming the first segment are for gold, and later segment are 
    for pseudo in logits.
    """
    if pseudo_labels is not None:
        assert logits.size(0) == labels.size(0) + pseudo_labels.size(0), \
            "logits prediction miss matches the provided gold and pseudo labels"
        n_golds = labels.size(0)

        def deal_nan(x):
            return 0 if th.isnan(x) else x
        mle_loss = deal_nan(loss_func(logits[:n_golds], labels))
        pl_loss = deal_nan(loss_func(logits[n_golds:], pseudo_labels))
        loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
    else:
        loss = loss_func(logits, labels)
    return loss
